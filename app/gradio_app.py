"""
Gradio web app — interfaccia per tecnici di manutenzione.
Upload foto componente → risultato + heatmap + spiegazione Gemma4.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import gradio as gr
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io

from preprocessing.preprocess import get_test_transform
from models.patchcore import PatchCoreDetector
from models.gemma_explainer import explain_anomaly, check_ollama_running

MODELS_DIR = Path(__file__).parent.parent / "results" / "models"
CATEGORIES = ["metal_nut", "screw", "grid"]

_detectors: dict = {}


def load_detector(category: str) -> PatchCoreDetector | None:
    if category not in _detectors:
        save_dir = MODELS_DIR / "patchcore" / category
        if not save_dir.exists():
            return None
        detector = PatchCoreDetector()
        try:
            detector.load(save_dir)
            _detectors[category] = detector
        except Exception as e:
            print(f"Errore caricamento {category}: {e}")
            return None
    return _detectors[category]


def score_to_label(score: float, threshold: float = 0.5) -> tuple:
    if score < threshold * 0.6:
        return "✅ NORMALE", "#1D9E75"
    elif score < threshold:
        return "⚠️ USURA INIZIALE", "#BA7517"
    else:
        return "🔴 SOSTITUIRE", "#D85A30"


def build_heatmap_image(
    original: np.ndarray,
    heatmap: np.ndarray,
) -> np.ndarray:
    import cv2
    hm_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    hm_resized = cv2.resize(hm_norm, (original.shape[1], original.shape[0]))
    hm_colored = (plt.cm.RdYlGn_r(hm_resized)[:, :, :3] * 255).astype(np.uint8)
    overlay = (0.55 * original + 0.45 * hm_colored).astype(np.uint8)
    return overlay


def analyze_component(image: Image.Image, category: str) -> tuple:
    if image is None:
        return "Carica un'immagine", None, "—", 0.0

    detector = load_detector(category)
    if detector is None:
        msg = f"Modello '{category}' non trovato. Esegui prima: python3 main.py --category {category}"
        return msg, None, "—", 0.0

    transform = get_test_transform()
    img_array = np.array(image.convert("RGB"))
    img_tensor = transform(image.convert("RGB"))

    score, heatmap = detector.predict_single_image(img_tensor)
    score_normalized = float(np.clip(score / 200.0, 0.0, 1.0))
    label, color = score_to_label(score_normalized)
    heatmap_img = build_heatmap_image(img_array, heatmap)

    gemma_status = "disponibile" if check_ollama_running() else "non disponibile (Ollama offline)"
    explanation = explain_anomaly(img_array=img_array, anomaly_score=score_normalized)

    result_text = f"""**Risultato:** {label}
**Anomaly score:** {score_normalized:.3f}
**Categoria:** {category}
**Gemma4:** {gemma_status}

---
**Analisi Gemma4:**
{explanation}"""

    return result_text, Image.fromarray(heatmap_img), label, score_normalized


# Colori Sidel: navy #002B5C, arancio #F47920, grigio chiaro #F4F6F9
SIDEL_CSS = """
/* ── Sfondo generale ── */
body, .gradio-container {
    background: #F4F6F9 !important;
    font-family: 'Inter', 'Helvetica Neue', sans-serif !important;
}
.gradio-container h1 {
    color: #002B5C !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
}
.gradio-container p { color: #4A5568 !important; }
button.primary {
    background: #F47920 !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}
button.primary:hover { background: #D4661A !important; }
.block, .panel, .form {
    background: white !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 4px rgba(0,43,92,0.06) !important;
}
label span, .label-wrap span {
    color: #002B5C !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}
.wrap-inner { border-color: #CBD5E0 !important; border-radius: 8px !important; }
input[type=range]::-webkit-slider-thumb { background: #F47920 !important; }
hr { border-color: #E2E8F0 !important; }
.gradio-container p em { font-size: 0.72rem !important; color: #94A3B8 !important; }
textarea, input[type=text] {
    border-color: #E2E8F0 !important;
    border-radius: 8px !important;
    background: #FAFBFC !important;
}
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="Sidel Maintenance Vision Assistant",
        css=SIDEL_CSS,
    ) as app:
        gr.Markdown("""
# 🔧 Sidel Maintenance Vision Assistant
**Anomaly Detection per componenti di macchinari industriali**  
Carica la foto di un componente meccanico per ottenere una valutazione automatica.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Foto componente")
                category_select = gr.Dropdown(
                    choices=CATEGORIES,
                    value="metal_nut",
                    label="Tipo componente",
                )
                analyze_btn = gr.Button("🔍 Analizza", variant="primary")

            with gr.Column(scale=1):
                result_text = gr.Markdown(label="Risultato")
                heatmap_output = gr.Image(label="Heatmap anomalia")

        with gr.Row():
            status_label = gr.Textbox(label="Status", interactive=False)
            score_slider = gr.Slider(0, 1, label="Anomaly score", interactive=False)

        analyze_btn.click(
            fn=analyze_component,
            inputs=[image_input, category_select],
            outputs=[result_text, heatmap_output, status_label, score_slider],
        )

        gr.Markdown("""
---
*Progetto: Industrial Anomaly Detection — Epicode Computer Vision Final Exam · PatchCore + EfficientNet-B0 + HOG+SVM + Gemma4:e4b (locale)*
        """)

    return app


if __name__ == "__main__":
    print("Avvio Gradio app...")
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
