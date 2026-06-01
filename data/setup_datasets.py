"""
Dataset setup script.
MVTec AD: richiede download manuale da https://www.mvtec.com/company/research/datasets/mvtec-ad
Ball Screw: richiede download manuale da https://doi.org/10.35097/489

Questo script organizza i dati dopo il download e verifica la struttura.
"""

import os
import zipfile
import tarfile
from pathlib import Path

DATA_DIR = Path(__file__).parent
MVTEC_CATEGORIES = ["metal_nut", "screw", "grid"]


def extract_mvtec(archive_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    if archive_path.suffix == ".gz":
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(target_dir)
    elif archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(target_dir)
    print(f"Estratto: {archive_path.name} → {target_dir}")


def verify_mvtec_structure(mvtec_root: Path) -> dict:
    report = {}
    for category in MVTEC_CATEGORIES:
        cat_path = mvtec_root / category
        if not cat_path.exists():
            report[category] = {"status": "MISSING"}
            continue
        train_normal = list((cat_path / "train" / "good").glob("*.png"))
        test_good = list((cat_path / "test" / "good").glob("*.png"))
        test_defect = []
        test_path = cat_path / "test"
        for subdir in test_path.iterdir():
            if subdir.is_dir() and subdir.name != "good":
                test_defect.extend(list(subdir.glob("*.png")))
        report[category] = {
            "status": "OK",
            "train_normal": len(train_normal),
            "test_good": len(test_good),
            "test_defect": len(test_defect),
        }
    return report


def verify_ballscrew_structure(bs_root: Path) -> dict:
    report = {}
    if not bs_root.exists():
        return {"status": "MISSING", "path": str(bs_root)}
    images = list(bs_root.rglob("*.png")) + list(bs_root.rglob("*.jpg"))
    report["status"] = "OK"
    report["total_images"] = len(images)
    return report


def print_report(mvtec_report: dict, bs_report: dict) -> None:
    print("\n=== VERIFICA DATASET ===")
    print("\n[MVTec AD]")
    for cat, info in mvtec_report.items():
        if info["status"] == "MISSING":
            print(f"  ❌ {cat}: NON TROVATO")
        else:
            print(f"  ✅ {cat}: train={info['train_normal']} normali | "
                  f"test={info['test_good']} OK + {info['test_defect']} difettosi")

    print("\n[Ball Screw Dataset]")
    if bs_report.get("status") == "MISSING":
        print(f"  ❌ NON TROVATO in {bs_report.get('path')}")
    else:
        print(f"  ✅ {bs_report.get('total_images', 0)} immagini totali")

    print("\n=== ISTRUZIONI DOWNLOAD ===")
    print("1. MVTec AD: https://www.mvtec.com/company/research/datasets/mvtec-ad")
    print("   → Scarica le categorie: metal_nut, screw, grid")
    print("   → Estrai in: data/mvtec/")
    print("2. Ball Screw: https://doi.org/10.35097/489")
    print("   → Estrai in: data/ballscrew/")


if __name__ == "__main__":
    mvtec_root = DATA_DIR / "mvtec"
    bs_root = DATA_DIR / "ballscrew"

    mvtec_report = verify_mvtec_structure(mvtec_root)
    bs_report = verify_ballscrew_structure(bs_root)
    print_report(mvtec_report, bs_report)
