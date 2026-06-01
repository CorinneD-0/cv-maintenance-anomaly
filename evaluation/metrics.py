"""
Modulo di valutazione: metriche, grafici e failure analysis.
Confronta HOG+SVM vs PatchCore su test set.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    precision_recall_curve, confusion_matrix,
    classification_report, f1_score,
)
import json


RESULTS_DIR = Path(__file__).parent.parent / "results"


def compute_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float = None,
    model_name: str = "model",
) -> dict:
    """Calcola tutte le metriche richieste dall'esame."""
    auroc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    if threshold is None:
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        optimal_idx = np.argmax(tpr - fpr)
        threshold = float(thresholds[optimal_idx])
    y_pred = (y_scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics = {
        "model": model_name,
        "auroc": float(auroc),
        "average_precision": float(ap),
        "f1_score": float(f1),
        "threshold": float(threshold),
        "precision": float(report.get("1", {}).get("precision", 0)),
        "recall": float(report.get("1", {}).get("recall", 0)),
        "confusion_matrix": cm.tolist(),
    }
    print(f"\n[{model_name}]")
    print(f"  AUROC:     {auroc:.4f}")
    print(f"  Avg Prec:  {ap:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    return metrics


def plot_roc_curves(
    results: dict,
    y_true: np.ndarray,
    save_path: Path = None,
) -> None:
    """Plotta curve ROC per confronto modelli."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"HOG+SVM": "#7F77DD", "PatchCore": "#D85A30"}

    for model_name, (y_scores, _) in results.items():
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auroc = roc_auc_score(y_true, y_scores)
        color = colors.get(model_name, "#888780")
        axes[0].plot(fpr, tpr, label=f"{model_name} (AUROC={auroc:.3f})", color=color, lw=2)

    axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("Curva ROC — Confronto modelli")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for model_name, (y_scores, _) in results.items():
        prec, rec, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        color = colors.get(model_name, "#888780")
        axes[1].plot(rec, prec, label=f"{model_name} (AP={ap:.3f})", color=color, lw=2)

    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_path or RESULTS_DIR / "roc_curves.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Curve ROC salvate in {save_path}")


def plot_confusion_matrix(
    metrics: dict,
    save_path: Path = None,
) -> None:
    """Plotta confusion matrix side-by-side per i modelli."""
    models = [m for m in metrics if "confusion_matrix" in metrics[m]]
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    if len(models) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        cm = np.array(metrics[model_name]["confusion_matrix"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normale", "Anomalia"],
            yticklabels=["Normale", "Anomalia"],
            ax=ax,
        )
        ax.set_title(f"Confusion Matrix — {model_name}")
        ax.set_ylabel("Ground Truth")
        ax.set_xlabel("Predizione")

    plt.tight_layout()
    save_path = save_path or RESULTS_DIR / "confusion_matrices.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix salvata in {save_path}")


def plot_heatmap(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    score: float,
    label: int,
    img_path: str = "",
    save_path: Path = None,
) -> None:
    """Visualizza immagine originale + heatmap anomalia."""
    import cv2
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original_img)
    axes[0].set_title("Immagine originale")
    axes[0].axis("off")

    hm_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    axes[1].imshow(hm_norm, cmap="RdYlGn_r")
    axes[1].set_title(f"Anomaly heatmap\nscore={score:.3f}")
    axes[1].axis("off")

    hm_resized = cv2.resize(hm_norm, (original_img.shape[1], original_img.shape[0]))
    hm_colored = plt.cm.RdYlGn_r(hm_resized)[:, :, :3]
    overlay = 0.6 * original_img / 255.0 + 0.4 * hm_colored
    overlay = np.clip(overlay, 0, 1)
    axes[2].imshow(overlay)
    label_str = "🔴 ANOMALIA" if label else "✅ NORMALE"
    axes[2].set_title(f"Overlay — {label_str}")
    axes[2].axis("off")

    plt.suptitle(Path(img_path).name, fontsize=10)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()


def failure_analysis(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    image_paths: list,
    threshold: float,
    model_name: str = "model",
    n_samples: int = 5,
) -> dict:
    """Identifica false positives e false negatives per failure analysis."""
    y_pred = (y_scores >= threshold).astype(int)
    fp_mask = (y_pred == 1) & (y_true == 0)
    fn_mask = (y_pred == 0) & (y_true == 1)
    analysis = {
        "model": model_name,
        "false_positives": {
            "count": int(fp_mask.sum()),
            "paths": [image_paths[i] for i in np.where(fp_mask)[0][:n_samples]],
            "scores": [float(y_scores[i]) for i in np.where(fp_mask)[0][:n_samples]],
        },
        "false_negatives": {
            "count": int(fn_mask.sum()),
            "paths": [image_paths[i] for i in np.where(fn_mask)[0][:n_samples]],
            "scores": [float(y_scores[i]) for i in np.where(fn_mask)[0][:n_samples]],
        },
    }
    print(f"\n[Failure Analysis — {model_name}]")
    print(f"  False Positives: {analysis['false_positives']['count']}")
    print(f"  False Negatives: {analysis['false_negatives']['count']}")
    return analysis


def save_results(metrics: dict, save_path: Path = None) -> None:
    save_path = save_path or RESULTS_DIR / "metrics.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metriche salvate in {save_path}")
