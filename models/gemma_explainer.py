"""
Gemma4:e4b explainer — layer di spiegazione in linguaggio naturale.
Chiama il modello localmente via Ollama REST API.
Privacy-first: le immagini non escono mai dalla macchina locale.
"""

import base64
import json
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from PIL import Image


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma4:e4b"
TIMEOUT = 60

SYSTEM_PROMPT = """You are an industrial maintenance expert AI assistant.
Your task is to analyze images of mechanical components from packaging machines
and identify visible signs of wear, damage, or anomalies.
Be concise and specific. Focus on observable defects."""

INSPECTION_PROMPT = """Analyze this industrial component image.
Describe any visible signs of:
- Surface wear or scratches
- Cracks or fractures
- Corrosion or contamination
- Deformation or misalignment
- Any other anomaly

If the component looks normal, say "Component appears normal — no visible defects."
Keep your analysis under 3 sentences."""


def image_to_base64(img_path: str | Path) -> str:
    """Converte immagine in base64 per Ollama API."""
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def numpy_to_base64(arr: np.ndarray) -> str:
    """Converte array numpy in base64 PNG."""
    from io import BytesIO
    if arr.max() <= 1.0:
        arr = (arr * 255).astype(np.uint8)
    img = Image.fromarray(arr.astype(np.uint8))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def check_ollama_running() -> bool:
    """Verifica che Ollama sia in esecuzione."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def check_model_available(model_name: str = MODEL_NAME) -> bool:
    """Verifica che il modello sia scaricato."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        return any(model_name in m for m in models)
    except Exception:
        return False


def explain_anomaly(
    img_path: Optional[str | Path] = None,
    img_array: Optional[np.ndarray] = None,
    anomaly_score: Optional[float] = None,
    custom_prompt: Optional[str] = None,
) -> str:
    """
    Analizza un'immagine con Gemma4 e restituisce spiegazione testuale.

    Args:
        img_path: percorso immagine su disco
        img_array: array numpy dell'immagine (alternativa a img_path)
        anomaly_score: score PatchCore (aggiunto al prompt per contesto)
        custom_prompt: prompt personalizzato (opzionale)

    Returns:
        Stringa con la spiegazione di Gemma4
    """
    if not check_ollama_running():
        return "[Gemma4 non disponibile] Ollama non è in esecuzione. Avvia con: ollama serve"

    if not check_model_available():
        return f"[Gemma4 non disponibile] Modello {MODEL_NAME} non trovato. Scarica con: ollama pull {MODEL_NAME}"

    if img_path is not None:
        image_b64 = image_to_base64(img_path)
    elif img_array is not None:
        image_b64 = numpy_to_base64(img_array)
    else:
        return "[Errore] Fornire img_path o img_array"

    prompt = custom_prompt or INSPECTION_PROMPT
    if anomaly_score is not None:
        severity = "HIGH" if anomaly_score > 0.7 else ("MEDIUM" if anomaly_score > 0.4 else "LOW")
        prompt = f"[Anomaly score: {anomaly_score:.3f} — Severity: {severity}]\n\n{prompt}"

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": [image_b64],
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 150,
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "[Risposta vuota da Gemma4]").strip()
    except requests.exceptions.Timeout:
        return "[Timeout] Gemma4 ha impiegato troppo. Riprova."
    except Exception as e:
        return f"[Errore Gemma4] {str(e)}"


def explain_batch(
    image_paths: list,
    anomaly_scores: Optional[list] = None,
) -> list:
    """Spiegazione batch — ritorna lista di stringhe."""
    results = []
    for i, path in enumerate(image_paths):
        score = anomaly_scores[i] if anomaly_scores else None
        explanation = explain_anomaly(img_path=path, anomaly_score=score)
        results.append(explanation)
    return results


if __name__ == "__main__":
    print("Test Gemma4 Explainer...")
    print(f"Ollama running: {check_ollama_running()}")
    print(f"Modello {MODEL_NAME} disponibile: {check_model_available()}")

    test_img = Path(__file__).parent.parent / "data" / "mvtec" / "metal_nut" / "test"
    test_images = list(test_img.rglob("*.png"))
    if test_images:
        print(f"\nTest su: {test_images[0].name}")
        result = explain_anomaly(img_path=test_images[0], anomaly_score=0.75)
        print(f"Risposta Gemma4:\n{result}")
    else:
        print("Nessuna immagine trovata per il test.")
