"""
── GEMMA4 EXPLAINER (layer di spiegazione testuale) ─────────────────
Dopo che PatchCore trova un'anomalia, Gemma4 spiega cosa ha visto
in linguaggio naturale — tipo un esperto di manutenzione che guarda la foto.
Gira in locale via Ollama → PRIVACY FIRST, nessun dato esce dalla macchina.
Questo è il componente MULTIMODALE del progetto (immagine + testo).
────────────────────────────────────────────────────────────────────
"""

import base64
import json
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from PIL import Image

# ── CONFIGURAZIONE OLLAMA ─────────────────────────────────────────────────────
# Ollama gira in locale sulla porta 11434 — zero connessione internet necessaria
# scelta deliberata per DATA SOVEREIGNTY: le foto dei componenti Sidel non escono
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma4:e4b"
TIMEOUT = 60  # secondi prima di rinunciare

# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
# dico a Gemma4 che ruolo deve fare — esperto di manutenzione industriale
# temperature=0.1 → risposta deterministica, non creativa (voglio fatti, non fantasia)
SYSTEM_PROMPT = """You are an industrial maintenance expert AI assistant.
Your task is to analyze images of mechanical components from packaging machines
and identify visible signs of wear, damage, or anomalies.
Be concise and specific. Focus on observable defects."""

# ── USER PROMPT ───────────────────────────────────────────────────────────────
# cosa chiedo esattamente a Gemma4 quando vede l'immagine
# aggiungo anche l'anomaly score di PatchCore come contesto
INSPECTION_PROMPT = """Analyze this industrial component image.
Describe any visible signs of:
- Surface wear or scratches
- Cracks or fractures
- Corrosion or contamination
- Deformation or misalignment
- Any other anomaly

If the component looks normal, say "Component appears normal — no visible defects."
Keep your analysis under 3 sentences."""


# ── CONVERSIONE IMMAGINE IN BASE64 ────────────────────────────────────────────
# Ollama vuole le immagini in formato base64 — le converto qui
def image_to_base64(img_path: str | Path) -> str:
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def numpy_to_base64(arr: np.ndarray) -> str:
    # se arriva un array numpy invece di un file, lo converto in PNG e poi base64
    from io import BytesIO
    if arr.max() <= 1.0:
        arr = (arr * 255).astype(np.uint8)
    img = Image.fromarray(arr.astype(np.uint8))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── CHECK OLLAMA DISPONIBILE ──────────────────────────────────────────────────
# verifico che Ollama sia acceso prima di fare la chiamata
# se non c'è → risposta di fallback invece di crash
def check_ollama_running() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

def check_model_available(model_name: str = MODEL_NAME) -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        return any(model_name in m for m in models)
    except Exception:
        return False


# ── SPIEGAZIONE ANOMALIA CON GEMMA4 ──────────────────────────────────────────
# MULTIMODAL INFERENCE — passo immagine + testo a Gemma4
# aggiungo l'anomaly score di PatchCore nel prompt → Gemma4 sa già che c'è qualcosa
# ritorna una stringa con la spiegazione del difetto in inglese tecnico
def explain_anomaly(
    img_path: Optional[str | Path] = None,
    img_array: Optional[np.ndarray] = None,
    anomaly_score: Optional[float] = None,
    custom_prompt: Optional[str] = None,
) -> str:
    # controllo che Ollama sia attivo
    if not check_ollama_running():
        return "[Gemma4 non disponibile] Ollama non è in esecuzione. Avvia con: ollama serve"
    if not check_model_available():
        return f"[Gemma4 non disponibile] Modello {MODEL_NAME} non trovato. Scarica con: ollama pull {MODEL_NAME}"

    # preparo l'immagine in base64
    if img_path is not None:
        image_b64 = image_to_base64(img_path)
    elif img_array is not None:
        image_b64 = numpy_to_base64(img_array)
    else:
        return "[Errore] Fornire img_path o img_array"

    # aggiungo l'anomaly score di PatchCore come contesto nel prompt
    prompt = custom_prompt or INSPECTION_PROMPT
    if anomaly_score is not None:
        severity = "HIGH" if anomaly_score > 0.7 else ("MEDIUM" if anomaly_score > 0.4 else "LOW")
        prompt = f"[Anomaly score: {anomaly_score:.3f} — Severity: {severity}]\n\n{prompt}"

    # ── CHIAMATA API OLLAMA ───────────────────────────────────────────────────
    # passo immagine + prompt + system prompt a Gemma4 in locale
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": [image_b64],
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "temperature": 0.1,   # quasi deterministico — voglio fatti, non creatività
            "num_predict": 150,   # risposta corta — il tecnico vuole 2-3 frasi
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


# ── SPIEGAZIONE BATCH ─────────────────────────────────────────────────────────
# versione batch — chiamo Gemma4 su una lista di immagini
# usata in main.py dopo l'inference per i casi più anomali
def explain_batch(
    image_paths: list,
    anomaly_scores: Optional[list] = None,
) -> list:
    results = []
    for i, path in enumerate(image_paths):
        score = anomaly_scores[i] if anomaly_scores else None
        explanation = explain_anomaly(img_path=path, anomaly_score=score)
        results.append(explanation)
    return results
