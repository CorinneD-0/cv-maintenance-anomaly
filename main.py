"""
Pipeline principale — orchestratore del progetto.

Fasi:
1. Preprocessing e DataLoader
2. Training HOG+SVM
3. Training PatchCore
4. Evaluation e confronto
5. Spiegazione Gemma4 sui casi anomali
6. Salvataggio risultati
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from preprocessing.preprocess import MVTecDataset, get_dataloader, denormalize
from models.hog_svm import HOGSVMDetector
from models.patchcore import PatchCoreDetector
from models.gemma_explainer import explain_anomaly, check_ollama_running
from evaluation.metrics import (
    compute_metrics, plot_roc_curves, plot_confusion_matrix,
    failure_analysis, save_results, plot_heatmap, RESULTS_DIR,
)

DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "results" / "models"


def run_pipeline(
    category: str = "metal_nut",
    skip_train: bool = False,
    explain: bool = True,
    n_explain: int = 3,
) -> None:
    print(f"\n{'='*60}")
    print(f" INDUSTRIAL ANOMALY DETECTION — {category.upper()}")
    print(f"{'='*60}\n")

    mvtec_root = DATA_DIR / "mvtec"
    if not mvtec_root.exists():
        print("❌ Dataset non trovato. Esegui: python3 data/setup_datasets.py")
        return

    print("📦 Caricamento dataset...")
    train_ds = MVTecDataset(mvtec_root, category=category, mode="train")
    test_ds = MVTecDataset(mvtec_root, category=category, mode="test")
    train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)
    test_loader = get_dataloader(test_ds, batch_size=16, shuffle=False)
    print(f"   Train: {len(train_ds)} immagini normali")
    print(f"   Test:  {len(test_ds)} immagini ({sum(test_ds.labels)} anomalie)")

    train_paths = [str(p) for p in train_ds.samples]
    test_paths = [str(p) for p in test_ds.samples]
    y_true = np.array(test_ds.labels)

    print("\n📊 Approccio classico: HOG + OneClassSVM")
    hog_detector = HOGSVMDetector()
    hog_save_dir = MODELS_DIR / "hog_svm" / category
    if skip_train and hog_save_dir.exists():
        hog_detector.load(hog_save_dir)
    else:
        hog_detector.fit(train_paths)
        hog_detector.save(hog_save_dir)
    _, hog_scores = hog_detector.predict(test_paths)
    hog_metrics = compute_metrics(y_true, hog_scores, model_name="HOG+SVM")

    print("\n🧠 Approccio deep learning: PatchCore")
    patchcore = PatchCoreDetector()
    pc_save_dir = MODELS_DIR / "patchcore" / category
    if skip_train and pc_save_dir.exists():
        patchcore.load(pc_save_dir)
    else:
        patchcore.fit(train_loader)
        patchcore.save(pc_save_dir)
    pc_scores, pc_heatmaps, _ = patchcore.predict(test_loader)
    pc_metrics = compute_metrics(y_true, pc_scores, model_name="PatchCore")

    print("\n📈 Generazione grafici...")
    results = {
        "HOG+SVM": (hog_scores, None),
        "PatchCore": (pc_scores, None),
    }
    plot_roc_curves(results, y_true, save_path=RESULTS_DIR / f"{category}_roc.png")

    all_metrics = {
        "HOG+SVM": hog_metrics,
        "PatchCore": pc_metrics,
    }
    plot_confusion_matrix(all_metrics, save_path=RESULTS_DIR / f"{category}_cm.png")

    print("\n🔍 Failure analysis...")
    hog_fa = failure_analysis(
        y_true, hog_scores, test_paths,
        threshold=hog_metrics["threshold"], model_name="HOG+SVM"
    )
    pc_fa = failure_analysis(
        y_true, pc_scores, test_paths,
        threshold=pc_metrics["threshold"], model_name="PatchCore"
    )

    print("\n🔥 Generazione heatmap su campioni anomali...")
    anomaly_indices = np.where(y_true == 1)[0][:3]
    for idx in anomaly_indices:
        img_orig = np.array(Image.open(test_paths[idx]).convert("RGB"))
        heatmap_path = RESULTS_DIR / f"{category}_heatmap_{idx}.png"
        plot_heatmap(
            img_orig, pc_heatmaps[idx], pc_scores[idx],
            y_true[idx], test_paths[idx], save_path=heatmap_path
        )

    if explain and check_ollama_running():
        print(f"\n💬 Spiegazione Gemma4 su {n_explain} anomalie...")
        anomaly_paths = [test_paths[i] for i in anomaly_indices[:n_explain]]
        anomaly_scores_sample = [pc_scores[i] for i in anomaly_indices[:n_explain]]
        for i, (path, score) in enumerate(zip(anomaly_paths, anomaly_scores_sample)):
            explanation = explain_anomaly(img_path=path, anomaly_score=score)
            print(f"\n  [{Path(path).name}] score={score:.3f}")
            print(f"  Gemma4: {explanation}")
            all_metrics.setdefault("gemma4_explanations", {})[Path(path).name] = explanation
    elif explain:
        print("\n⚠️  Gemma4 non disponibile — Ollama non in esecuzione")
        print("   Avvia con: ollama serve")

    all_metrics["failure_analysis"] = {
        "HOG+SVM": hog_fa,
        "PatchCore": pc_fa,
    }
    save_results(all_metrics, RESULTS_DIR / f"{category}_metrics.json")

    print(f"\n{'='*60}")
    print(f" COMPLETATO — risultati in: {RESULTS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Industrial Anomaly Detection Pipeline")
    parser.add_argument("--category", default="metal_nut",
                        choices=["metal_nut", "screw", "grid"],
                        help="Categoria MVTec AD")
    parser.add_argument("--skip-train", action="store_true",
                        help="Carica modelli già addestrati")
    parser.add_argument("--no-explain", action="store_true",
                        help="Disabilita spiegazione Gemma4")
    args = parser.parse_args()

    run_pipeline(
        category=args.category,
        skip_train=args.skip_train,
        explain=not args.no_explain,
    )
