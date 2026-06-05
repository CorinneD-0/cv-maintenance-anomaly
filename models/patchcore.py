"""
── PATCHCORE MODEL ─────────────────────────────────────────────────
Questo è il cuore del progetto — il modello deep learning per anomaly detection.
L'idea è semplice: imparo come sono fatti i pezzi NORMALI, e poi tutto
quello che si discosta troppo lo chiamo anomalia.
Non ho mai bisogno di foto di pezzi rotti per addestrarlo.
────────────────────────────────────────────────────────────────────
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

# ── CONFIGURAZIONE DEVICE ────────────────────────────────────────────────────
# uso MPS se sono su Mac con chip Apple, altrimenti CUDA o CPU
# il training gira su MPS (veloce), l'inference su CPU per evitare crash con FAISS
PATCH_SIZE = 3      # dimensione delle patch che analizzo nell'immagine
STRIDE = 1          # quanto sposto la finestra ogni volta
CORESET_RATIO = 0.1 # tengo il 10% delle feature nel memory bank
DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)
INFERENCE_DEVICE = "cpu"  # FAISS + MPS crashano su Python 3.14 — uso CPU per l'inference


# ── FEATURE EXTRACTOR (EFFICIENTNET-B0) ─────────────────────────────────────
# uso EfficientNet-B0 pre-addestrato su ImageNet come "occhio" del sistema
# il backbone è CONGELATO — non lo alleno, uso solo le sue feature generali
# prendo i layer 2 e 4 perché catturano info a scale diverse (texture fine + forme)
class FeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        try:
            # carico i pesi pre-addestrati su ImageNet
            backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        except Exception:
            # se non c'è connessione uso pesi casuali (solo per test)
            print("⚠️  Pesi ImageNet non scaricabili — uso pesi random (solo per test)")
            backbone = models.efficientnet_b0(weights=None)
        self.features = backbone.features
        # TRANSFER LEARNING — congelo tutti i parametri, non alleno niente
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # passo l'immagine attraverso i layer e salvo l'output a due profondità diverse
        out = x
        layer2_out = None
        for i, block in enumerate(self.features):
            out = block(out)
            if i == 2:
                layer2_out = out   # layer 2 → texture fine, dettagli piccoli
            if i == 4:
                layer4_out = out   # layer 4 → forme più complesse, struttura generale
                break
        return layer2_out, layer4_out


# ── AGGREGAZIONE PATCH FEATURES ──────────────────────────────────────────────
# unisco le feature dei due layer e le organizzo in "patch" dell'immagine
# ogni patch è una piccola zona dell'immagine → ho un vettore per ogni zona
# questo mi permette poi di fare la HEATMAP (so esattamente dove c'è l'anomalia)
def aggregate_patch_features(
    feat_low: torch.Tensor,   # feature dal layer 2 (texture fine)
    feat_high: torch.Tensor,  # feature dal layer 4 (forme)
    patch_size: int = PATCH_SIZE,
    stride: int = STRIDE,
) -> Tuple[torch.Tensor, int, int]:
    # porto le feature del layer 4 alla stessa risoluzione del layer 2
    target_size = feat_low.shape[-2:]
    feat_high_up = F.interpolate(feat_high, size=target_size, mode="bilinear", align_corners=False)
    # concateno i due livelli → descrizione multi-scala di ogni zona
    combined = torch.cat([feat_low, feat_high_up], dim=1)
    # HOG-like: divido in patch e prendo le feature per ognuna
    unfold = nn.Unfold(kernel_size=patch_size, stride=stride, padding=patch_size // 2)
    B, C, H, W = combined.shape
    patches = unfold(combined)
    patches = patches.reshape(B, C, patch_size, patch_size, -1)
    patches = patches.permute(0, 4, 1, 2, 3)
    patch_features = patches.reshape(B, patches.shape[1], -1)
    return patch_features, H, W


# ── CONVERSIONE NUMPY PER FAISS ──────────────────────────────────────────────
# FAISS su Apple Silicon vuole array contigui in float32 — senza questo crasha
# questa funzione garantisce sempre il formato giusto
def to_numpy_f32(t: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(t.detach().cpu().numpy(), dtype=np.float32)


# ── PATCHCORE DETECTOR ───────────────────────────────────────────────────────
# questa è la classe principale — racchiude tutto il flusso:
# fit() → guarda le immagini normali e le memorizza
# predict() → confronta una nuova immagine con la memoria → anomaly score
class PatchCoreDetector:
    def __init__(self, coreset_ratio: float = CORESET_RATIO) -> None:
        self.train_device = DEVICE
        self.infer_device = INFERENCE_DEVICE
        self.extractor = FeatureExtractor().to(self.train_device).eval()
        self.coreset_ratio = coreset_ratio
        self.memory_bank: Optional[np.ndarray] = None  # qui salvo i vettori "normali"
        self.index: Optional[faiss.Index] = None        # FAISS index per ricerca veloce
        self.feature_map_size: Optional[Tuple[int, int]] = None
        self.is_fitted = False
        print(f"PatchCore inizializzato — train device: {self.train_device}")

    # ── ESTRAZIONE FEATURE DAL TRAINING SET ──────────────────────────────────
    # per ogni immagine normale estraggo le feature con EfficientNet
    # risultato: un array enorme di vettori che descrivono le zone normali
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

    # ── CORESET SUBSAMPLING ───────────────────────────────────────────────────
    # il training genera milioni di vettori — troppi da tenere tutti
    # con il coreset greedy scelgo i più rappresentativi (quelli più "lontani" tra loro)
    # è come scegliere i delegati di una classe: vuoi che ogni tipo sia rappresentato
    # NOTA: senza pre-subsampling → più lento ma memory bank più ricco e risultati migliori
    def _coreset_subsampling(self, features: np.ndarray) -> np.ndarray:
        n_select = max(1, int(len(features) * self.coreset_ratio))
        print(f"Coreset: {len(features)} → {n_select} feature selezionate")
        indices = [np.random.randint(0, len(features))]
        distances = np.full(len(features), np.inf)
        # algoritmo greedy: ogni iterazione prendo il punto più lontano da tutti quelli già scelti
        for _ in tqdm(range(n_select - 1), desc="Coreset subsampling"):
            last = features[indices[-1]]
            new_dist = np.linalg.norm(features - last, axis=1)
            distances = np.minimum(distances, new_dist)
            indices.append(int(np.argmax(distances)))
        return np.ascontiguousarray(features[indices], dtype=np.float32)

    # ── TRAINING (FIT) ────────────────────────────────────────────────────────
    # qui costruisco la "memoria" del sistema — guardo solo immagini normali
    # UNSUPERVISED: non ho mai bisogno di esempi difettosi per addestrare
    def fit(self, dataloader: DataLoader) -> None:
        print("Estrazione feature dal training set...")
        features, self.feature_map_size = self._extract_features(
            dataloader, self.train_device
        )
        print(f"Feature estratte: {features.shape}")
        # MEMORY BANK — tengo solo i vettori più rappresentativi
        self.memory_bank = self._coreset_subsampling(features)
        # FAISS INDEX — struttura dati per trovare il nearest neighbor velocemente
        dim = self.memory_bank.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.memory_bank)
        self.is_fitted = True
        print(f"Memory bank costruito: {self.memory_bank.shape[0]} vettori")

    # ── INFERENCE / PREDIZIONE ────────────────────────────────────────────────
    # per ogni immagine nuova: estraggo le feature → cerco il vicino più simile nel memory bank
    # ANOMALY SCORE = distanza dal nearest neighbor → più è alta, più è anomala
    # giro tutto su CPU perché FAISS + MPS crashano su Python 3.14 (fix OpenMP conflict)
    @torch.no_grad()
    def predict(
        self, dataloader: DataLoader
    ) -> Tuple[np.ndarray, list, np.ndarray]:
        assert self.is_fitted, "Chiama fit() prima di predict()"
        all_scores, all_heatmaps, all_labels = [], [], []
        extractor = self.extractor.to(self.infer_device).eval()
        print(f"Inference su device: {self.infer_device}")

        for imgs, labels, _ in tqdm(dataloader, desc="Inference PatchCore"):
            imgs = imgs.to(self.infer_device)
            feat_low, feat_high = extractor(imgs)
            patch_feats, fH, fW = aggregate_patch_features(feat_low, feat_high)
            B, N, C = patch_feats.shape
            flat = to_numpy_f32(patch_feats.reshape(B * N, C))
            # NEAREST NEIGHBOR SEARCH — cerco la patch normale più simile per ognuna
            distances, _ = self.index.search(flat, k=1)
            distances = distances.reshape(B, N)
            for i in range(B):
                score_map = distances[i].reshape(fH, fW)
                # ANOMALY SCORE — prendo il massimo della mappa (zona più anomala)
                all_scores.append(float(score_map.max()))
                # HEATMAP — la mappa completa serve per la visualizzazione
                all_heatmaps.append(score_map.copy())
                all_labels.append(int(labels[i]))

        return np.array(all_scores), all_heatmaps, np.array(all_labels)

    # ── PREDIZIONE SU SINGOLA IMMAGINE (per la Gradio app) ───────────────────
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

    # ── SALVATAGGIO MODELLO ───────────────────────────────────────────────────
    # salvo tutto su disco così non devo riaddestrare da capo ogni volta
    # 3 file: memory bank (numpy), metadati (pickle), FAISS index
    def save(self, save_dir: Path) -> None:
        import pickle
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / "memory_bank.npy", self.memory_bank)
        with open(save_dir / "meta.pkl", "wb") as f:
            pickle.dump({"feature_map_size": self.feature_map_size}, f)
        faiss.write_index(self.index, str(save_dir / "faiss.index"))
        print(f"PatchCore salvato in {save_dir}")

    # ── CARICAMENTO MODELLO ───────────────────────────────────────────────────
    # ricarico il modello già addestrato — usato con --skip-train
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
