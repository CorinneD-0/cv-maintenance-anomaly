"""
PatchCore: anomaly detection unsupervised basato su EfficientNet-B0.
Si addestra SOLO su immagini normali — nessun difetto necessario nel training.
"""

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights
from torch.utils.data import DataLoader
from tqdm import tqdm
import faiss

faiss.omp_set_num_threads(1)  # evita conflitto OpenMP con torch e sklearn

PATCH_SIZE = 3
STRIDE = 1
CORESET_RATIO = 0.1
DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)
# FAISS inference forzata su CPU — evita segfault MPS + Python 3.14
INFERENCE_DEVICE = "cpu"


class FeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        try:
            backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        except Exception:
            print("⚠️  Pesi ImageNet non scaricabili — uso pesi random (solo per test)")
            backbone = models.efficientnet_b0(weights=None)
        self.features = backbone.features
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = x
        layer2_out = None
        for i, block in enumerate(self.features):
            out = block(out)
            if i == 2:
                layer2_out = out
            if i == 4:
                layer4_out = out
                break
        return layer2_out, layer4_out


def aggregate_patch_features(
    feat_low: torch.Tensor,
    feat_high: torch.Tensor,
    patch_size: int = PATCH_SIZE,
    stride: int = STRIDE,
) -> Tuple[torch.Tensor, int, int]:
    target_size = feat_low.shape[-2:]
    feat_high_up = F.interpolate(
        feat_high, size=target_size, mode="bilinear", align_corners=False
    )
    combined = torch.cat([feat_low, feat_high_up], dim=1)
    unfold = nn.Unfold(kernel_size=patch_size, stride=stride, padding=patch_size // 2)
    B, C, H, W = combined.shape
    patches = unfold(combined)
    patches = patches.reshape(B, C, patch_size, patch_size, -1)
    patches = patches.permute(0, 4, 1, 2, 3)
    patch_features = patches.reshape(B, patches.shape[1], -1)
    return patch_features, H, W


def to_numpy_f32(t: torch.Tensor) -> np.ndarray:
    """Converte tensore → numpy float32 contiguo, sempre su CPU."""
    return np.ascontiguousarray(t.detach().cpu().numpy(), dtype=np.float32)


class PatchCoreDetector:
    def __init__(self, coreset_ratio: float = CORESET_RATIO) -> None:
        self.train_device = DEVICE
        self.infer_device = INFERENCE_DEVICE
        self.extractor = FeatureExtractor().to(self.train_device).eval()
        self.coreset_ratio = coreset_ratio
        self.memory_bank: Optional[np.ndarray] = None
        self.index: Optional[faiss.Index] = None
        self.feature_map_size: Optional[Tuple[int, int]] = None
        self.is_fitted = False
        print(f"PatchCore inizializzato — train device: {self.train_device}")

    @torch.no_grad()
    def _extract_features(
        self, dataloader: DataLoader, device: str
    ) -> Tuple[np.ndarray, Tuple]:
        all_features = []
        map_size = None
        extractor = self.extractor.to(device)
        for imgs, _, _ in tqdm(dataloader, desc="Estrazione feature"):
            imgs = imgs.to(device)
            feat_low, feat_high = extractor(imgs)
            patch_feats, H, W = aggregate_patch_features(feat_low, feat_high)
            if map_size is None:
                map_size = (H, W)
            B, N, C = patch_feats.shape
            all_features.append(to_numpy_f32(patch_feats.reshape(B * N, C)))
        return np.concatenate(all_features, axis=0), map_size

    def _coreset_subsampling(self, features: np.ndarray) -> np.ndarray:
        """Greedy coreset su tutto il feature set — nessun pre-subsampling."""
        n_select = max(1, int(len(features) * self.coreset_ratio))
        print(f"Coreset: {len(features)} → {n_select} feature selezionate")
        indices = [np.random.randint(0, len(features))]
        distances = np.full(len(features), np.inf)
        for _ in tqdm(range(n_select - 1), desc="Coreset subsampling"):
            last = features[indices[-1]]
            new_dist = np.linalg.norm(features - last, axis=1)
            distances = np.minimum(distances, new_dist)
            indices.append(int(np.argmax(distances)))
        return np.ascontiguousarray(features[indices], dtype=np.float32)

    def fit(self, dataloader: DataLoader) -> None:
        print("Estrazione feature dal training set...")
        features, self.feature_map_size = self._extract_features(
            dataloader, self.train_device
        )
        print(f"Feature estratte: {features.shape}")
        self.memory_bank = self._coreset_subsampling(features)
        dim = self.memory_bank.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.memory_bank)
        self.is_fitted = True
        print(f"Memory bank costruito: {self.memory_bank.shape[0]} vettori")

    @torch.no_grad()
    def predict(
        self, dataloader: DataLoader
    ) -> Tuple[np.ndarray, list, np.ndarray]:
        assert self.is_fitted, "Chiama fit() prima di predict()"
        all_scores, all_heatmaps, all_labels = [], [], []

        # Sposta extractor su CPU per inference — evita segfault MPS+FAISS
        extractor = self.extractor.to(self.infer_device).eval()
        print(f"Inference su device: {self.infer_device}")

        for imgs, labels, _ in tqdm(dataloader, desc="Inference PatchCore"):
            imgs = imgs.to(self.infer_device)
            feat_low, feat_high = extractor(imgs)
            patch_feats, fH, fW = aggregate_patch_features(feat_low, feat_high)
            B, N, C = patch_feats.shape
            flat = to_numpy_f32(patch_feats.reshape(B * N, C))
            distances, _ = self.index.search(flat, k=1)
            distances = distances.reshape(B, N)
            for i in range(B):
                score_map = distances[i].reshape(fH, fW)
                all_scores.append(float(score_map.max()))
                all_heatmaps.append(score_map.copy())
                all_labels.append(int(labels[i]))

        return np.array(all_scores), all_heatmaps, np.array(all_labels)

    def predict_single_image(
        self, img_tensor: torch.Tensor
    ) -> Tuple[float, np.ndarray]:
        assert self.is_fitted
        extractor = self.extractor.to(self.infer_device).eval()
        img_tensor = img_tensor.unsqueeze(0).to(self.infer_device)
        with torch.no_grad():
            feat_low, feat_high = extractor(img_tensor)
            patch_feats, fH, fW = aggregate_patch_features(feat_low, feat_high)
            flat = to_numpy_f32(patch_feats.reshape(-1, patch_feats.shape[-1]))
            distances, _ = self.index.search(flat, k=1)
            score_map = distances.reshape(fH, fW)
            return float(score_map.max()), score_map.copy()

    def save(self, save_dir: Path) -> None:
        import pickle
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / "memory_bank.npy", self.memory_bank)
        with open(save_dir / "meta.pkl", "wb") as f:
            pickle.dump({"feature_map_size": self.feature_map_size}, f)
        faiss.write_index(self.index, str(save_dir / "faiss.index"))
        print(f"PatchCore salvato in {save_dir}")

    def load(self, save_dir: Path) -> None:
        import pickle
        save_dir = Path(save_dir)
        self.memory_bank = np.ascontiguousarray(
            np.load(save_dir / "memory_bank.npy"), dtype=np.float32
        )
        with open(save_dir / "meta.pkl", "rb") as f:
            meta = pickle.load(f)
        self.feature_map_size = meta["feature_map_size"]
        self.index = faiss.read_index(str(save_dir / "faiss.index"))
        self.is_fitted = True
        print(f"PatchCore caricato da {save_dir}")
