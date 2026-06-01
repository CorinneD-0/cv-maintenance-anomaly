"""
Approccio classico: HOG features + OneClassSVM.
Si addestra solo su immagini normali (unsupervised).
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import joblib
from PIL import Image
from skimage.feature import hog
from skimage.transform import resize
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (16, 16),
    "cells_per_block": (2, 2),
    "channel_axis": -1,
}
IMAGE_SIZE = (224, 224)


def extract_hog_features(img_array: np.ndarray) -> np.ndarray:
    """Estrae feature HOG da un'immagine numpy (H, W, C) float [0,1]."""
    img_resized = resize(img_array, IMAGE_SIZE, anti_aliasing=True)
    features = hog(img_resized, **HOG_PARAMS)
    return features


def load_image_as_array(img_path: str | Path) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    return np.array(img).astype(np.float32) / 255.0


def extract_features_from_paths(
    image_paths: list, desc: str = "Extracting HOG"
) -> np.ndarray:
    features = []
    for path in tqdm(image_paths, desc=desc):
        try:
            img = load_image_as_array(path)
            feat = extract_hog_features(img)
            features.append(feat)
        except Exception as e:
            print(f"Errore su {path}: {e}")
    return np.array(features)


class HOGSVMDetector:
    """
    Detector basato su HOG + OneClassSVM.
    Addestrato solo su immagini normali → rileva anomalie come outlier.
    """

    def __init__(self, nu: float = 0.05, kernel: str = "rbf", gamma: str = "scale") -> None:
        self.scaler = StandardScaler()
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        self.is_fitted = False

    def fit(self, train_paths: list) -> None:
        print("Estrazione feature HOG dal training set...")
        features = extract_features_from_paths(train_paths, desc="HOG train")
        features_scaled = self.scaler.fit_transform(features)
        print(f"Training OneClassSVM su {len(features)} campioni...")
        self.model.fit(features_scaled)
        self.is_fitted = True
        print("HOG+SVM addestrato.")

    def predict(self, image_paths: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ritorna:
        - predictions: 0=normale, 1=anomalia
        - scores: anomaly score (più alto = più anomalo)
        """
        assert self.is_fitted, "Chiama fit() prima di predict()"
        features = extract_features_from_paths(image_paths, desc="HOG test")
        features_scaled = self.scaler.transform(features)
        raw_pred = self.model.predict(features_scaled)
        predictions = np.where(raw_pred == -1, 1, 0)
        scores = -self.model.decision_function(features_scaled)
        return predictions, scores

    def predict_single(self, img_path: str | Path) -> Tuple[int, float]:
        pred, score = self.predict([img_path])
        return int(pred[0]), float(score[0])

    def save(self, save_dir: Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, save_dir / "hog_scaler.pkl")
        joblib.dump(self.model, save_dir / "hog_svm.pkl")
        print(f"Modello salvato in {save_dir}")

    def load(self, save_dir: Path) -> None:
        save_dir = Path(save_dir)
        self.scaler = joblib.load(save_dir / "hog_scaler.pkl")
        self.model = joblib.load(save_dir / "hog_svm.pkl")
        self.is_fitted = True
        print(f"Modello caricato da {save_dir}")


if __name__ == "__main__":
    print("Test HOG+SVM detector...")
    data_root = Path(__file__).parent.parent / "data" / "mvtec" / "metal_nut"
    if not data_root.exists():
        print("Dataset non trovato. Esegui prima setup_datasets.py")
    else:
        train_paths = list((data_root / "train" / "good").glob("*.png"))
        print(f"Training su {len(train_paths)} immagini normali...")
        detector = HOGSVMDetector()
        detector.fit(train_paths)
        test_good = list((data_root / "test" / "good").glob("*.png"))[:3]
        test_bad = [
            p for d in (data_root / "test").iterdir()
            if d.is_dir() and d.name != "good"
            for p in list(d.glob("*.png"))[:3]
        ]
        print("\nTest su immagini buone:")
        for p in test_good:
            pred, score = detector.predict_single(p)
            print(f"  {p.name}: pred={'ANOMALIA' if pred else 'OK'} score={score:.3f}")
        print("\nTest su immagini difettose:")
        for p in test_bad:
            pred, score = detector.predict_single(p)
            print(f"  {p.name}: pred={'ANOMALIA' if pred else 'OK'} score={score:.3f}")
