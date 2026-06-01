"""
Preprocessing pipeline.
- Resize 224x224
- Normalizzazione ImageNet
- Augmentation per training
- DataLoader PyTorch per train (solo normali) e test
"""

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


def get_train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_test_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class MVTecDataset(Dataset):
    """
    Dataset MVTec AD per una singola categoria.
    mode='train' → solo immagini normali (cartella good)
    mode='test'  → normali (label=0) + difettose (label=1)
    """

    def __init__(
        self,
        root: Path,
        category: str,
        mode: str = "train",
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        assert mode in ("train", "test"), "mode deve essere 'train' o 'test'"
        self.root = Path(root) / category
        self.mode = mode
        self.transform = transform or (
            get_train_transform() if mode == "train" else get_test_transform()
        )
        self.samples, self.labels = self._load_samples()

    def _load_samples(self) -> Tuple[list, list]:
        samples, labels = [], []
        if self.mode == "train":
            good_path = self.root / "train" / "good"
            for img_path in sorted(good_path.glob("*.png")):
                samples.append(img_path)
                labels.append(0)
        else:
            test_path = self.root / "test"
            for subdir in sorted(test_path.iterdir()):
                if not subdir.is_dir():
                    continue
                label = 0 if subdir.name == "good" else 1
                for img_path in sorted(subdir.glob("*.png")):
                    samples.append(img_path)
                    labels.append(label)
        return samples, labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img = Image.open(self.samples[idx]).convert("RGB")
        img_tensor = self.transform(img)
        return img_tensor, self.labels[idx], str(self.samples[idx])


class BallScrewDataset(Dataset):
    """
    Dataset Ball Screw per anomaly detection su componenti meccanici.
    Struttura attesa: root/normal/*.png e root/defect/*.png (o simile)
    """

    def __init__(
        self,
        root: Path,
        mode: str = "train",
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.root = Path(root)
        self.mode = mode
        self.transform = transform or (
            get_train_transform() if mode == "train" else get_test_transform()
        )
        self.samples, self.labels = self._load_samples()

    def _load_samples(self) -> Tuple[list, list]:
        samples, labels = [], []
        normal_dirs = ["normal", "good", "Normal", "Good"]
        defect_dirs = ["defect", "anomaly", "Defect", "Anomaly", "worn", "Worn"]

        for dir_name in normal_dirs:
            candidate = self.root / dir_name
            if candidate.exists():
                for ext in ["*.png", "*.jpg", "*.bmp"]:
                    for img_path in sorted(candidate.glob(ext)):
                        if self.mode == "train" or True:
                            samples.append(img_path)
                            labels.append(0)
                break

        if self.mode == "test":
            for dir_name in defect_dirs:
                candidate = self.root / dir_name
                if candidate.exists():
                    for ext in ["*.png", "*.jpg", "*.bmp"]:
                        for img_path in sorted(candidate.glob(ext)):
                            samples.append(img_path)
                            labels.append(1)
                    break

        return samples, labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img = Image.open(self.samples[idx]).convert("RGB")
        img_tensor = self.transform(img)
        return img_tensor, self.labels[idx], str(self.samples[idx])


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Riporta un tensore normalizzato a immagine visualizzabile."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = img.permute(1, 2, 0).numpy()
    return np.clip(img, 0, 1)


if __name__ == "__main__":
    print("Test preprocessing pipeline...")
    data_root = Path(__file__).parent.parent / "data" / "mvtec"
    if not data_root.exists():
        print(f"Dataset non trovato in {data_root}")
        print("Esegui prima: python3 data/setup_datasets.py")
    else:
        ds = MVTecDataset(data_root, category="metal_nut", mode="train")
        print(f"Train samples: {len(ds)}")
        img, label, path = ds[0]
        print(f"Shape: {img.shape} | Label: {label} | Min: {img.min():.2f} | Max: {img.max():.2f}")
