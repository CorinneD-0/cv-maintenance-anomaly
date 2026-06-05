"""
── HOG + ONECLASS SVM (approccio classico) ─────────────────────────
Questo è il modello "tradizionale" del progetto — senza reti neurali.
HOG trasforma ogni immagine in un vettore di numeri che descrive bordi e texture.
OneClassSVM impara com'è fatto un pezzo normale e segnala tutto il resto come anomalia.
Confrontato con PatchCore nella valutazione finale per vedere chi va meglio e dove.
────────────────────────────────────────────────────────────────────
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

# ── PARAMETRI HOG ────────────────────────────────────────────────────────────
# orientations=9 → divido i 180° in 9 direzioni (ogni 20°)
# pixels_per_cell → ogni cella è 16x16 pixel
# cells_per_block → normalizzo in blocchi 2x2 celle per robustezza alla luce
HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (16, 16),
    "cells_per_block": (2, 2),
    "channel_axis": -1,
}
IMAGE_SIZE = (224, 224)  # tutte le immagini vengono portate a questa dimensione


# ── HOG FEATURE EXTRACTION ───────────────────────────────────────────────────
# HOG = Histogram of Oriented Gradients
# in pratica: per ogni zona dell'immagine calcola in che direzione vanno i bordi
# il risultato è un vettore lungo ~6000 numeri che descrive la "forma" dell'immagine
# un pezzo usurato ha bordi diversi da uno normale → il vettore cambia
def extract_hog_features(img_array: np.ndarray) -> np.ndarray:
    img_resized = resize(img_array, IMAGE_SIZE, anti_aliasing=True)
    features = hog(img_resized, **HOG_PARAMS)
    return features


def load_image_as_array(img_path: str | Path) -> np.ndarray:
    # carico l'immagine e la normalizzo in [0,1] — serve per HOG
    img = Image.open(img_path).convert("RGB")
    return np.array(img).astype(np.float32) / 255.0


def extract_features_from_paths(
    image_paths: list, desc: str = "Extracting HOG"
) -> np.ndarray:
    # estraggo HOG da una lista di immagini — usato sia in training che in test
    features = []
    for path in tqdm(image_paths, desc=desc):
        try:
            img = load_image_as_array(path)
            feat = extract_hog_features(img)
            features.append(feat)
        except Exception as e:
            print(f"Errore su {path}: {e}")
    return np.array(features)


# ── ONECLASS SVM DETECTOR ────────────────────────────────────────────────────
# ONECLASS SVM = macchina a vettori di supporto per una sola classe
# impara a costruire una "bolla" attorno ai dati normali
# tutto quello che cade fuori dalla bolla → ANOMALIA
# UNSUPERVISED: si allena solo su immagini normali, non serve mai un difetto
class HOGSVMDetector:
    def __init__(self, nu: float = 0.05, kernel: str = "rbf", gamma: str = "scale") -> None:
        # STANDARDSCALER — normalizza le feature prima di darle all'SVM
        # necessario perché HOG produce valori su scale diverse
        self.scaler = StandardScaler()
        # nu=0.05 → al massimo il 5% dei dati di training può essere outlier
        # kernel RBF → mappa i dati in uno spazio dove la "bolla" funziona meglio
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        self.is_fitted = False

    # ── TRAINING HOG+SVM ─────────────────────────────────────────────────────
    # 1. estraggo HOG da tutte le immagini normali
    # 2. normalizzo con StandardScaler
    # 3. alleno OneClassSVM — impara dove sta la "normalità"
    def fit(self, train_paths: list) -> None:
        print("Estrazione feature HOG dal training set...")
        features = extract_features_from_paths(train_paths, desc="HOG train")
        features_scaled = self.scaler.fit_transform(features)
        print(f"Training OneClassSVM su {len(features)} campioni...")
        self.model.fit(features_scaled)
        self.is_fitted = True
        print("HOG+SVM addestrato.")

    # ── PREDIZIONE ────────────────────────────────────────────────────────────
    # per ogni immagine: estraggo HOG → normalizzo → chiedo all'SVM se è normale
    # predictions: 0=normale, 1=anomalia
    # scores: anomaly score (più è alto, più è anomalo — usato per AUROC)
    def predict(self, image_paths: list) -> Tuple[np.ndarray, np.ndarray]:
        assert self.is_fitted, "Chiama fit() prima di predict()"
        features = extract_features_from_paths(image_paths, desc="HOG test")
        features_scaled = self.scaler.transform(features)
        # l'SVM ritorna +1 (normale) o -1 (anomalia) → converto in 0/1
        raw_pred = self.model.predict(features_scaled)
        predictions = np.where(raw_pred == -1, 1, 0)
        # decision_function → quanto è lontano dal confine (negato = anomaly score)
        scores = -self.model.decision_function(features_scaled)
        return predictions, scores

    def predict_single(self, img_path: str | Path) -> Tuple[int, float]:
        pred, score = self.predict([img_path])
        return int(pred[0]), float(score[0])

    # ── SALVATAGGIO E CARICAMENTO ─────────────────────────────────────────────
    # salvo scaler + modello con joblib (più veloce di pickle per array numpy grandi)
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
