# Industrial Anomaly Detection for Maintenance

> Computer Vision Final Exam — Epicode Institute of Technology  
> Student: Corinne D'Aquino

## Overview

Un sistema di anomaly detection per componenti meccanici industriali, progettato per assistere i tecnici di manutenzione durante i check visivi periodici (piani MMT).

**Problema reale:** I piani di manutenzione Sidel richiedono ispezioni visive periodiche su componenti come cuscinetti, viti, ingranaggi e cinghie. La valutazione soggettiva del tecnico può variare — questo sistema fornisce un supporto oggettivo e automatico.

**Output per ogni immagine:**
- ✅ NORMALE / ⚠️ USURA INIZIALE / 🔴 SOSTITUIRE
- Heatmap della zona anomala
- Spiegazione in linguaggio naturale (Gemma4:e4b locale)

## Pipeline

```
Input immagine
    │
    ├── HOG + OneClassSVM     (approccio classico — baseline)
    │
    ├── PatchCore             (approccio deep learning — principale)
    │   └── EfficientNet-B0 backbone (congelato, pretrained ImageNet)
    │       └── Memory bank feature normali → nearest neighbor distance
    │
    └── Gemma4:e4b (Ollama)   (spiegazione testuale locale, privacy-first)
```

## Dataset

- **MVTec AD** — categorie: `metal_nut`, `screw`, `grid`
  - Download: https://www.mvtec.com/company/research/datasets/mvtec-ad
  - Estrai in: `data/mvtec/`
- **Ball Screw Dataset** (KIT)
  - Download: https://doi.org/10.35097/489
  - Estrai in: `data/ballscrew/`

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/cv-maintenance-anomaly
cd cv-maintenance-anomaly

pip3 install -r requirements.txt

python3 data/setup_datasets.py
```

Per Gemma4 (opzionale):
```bash
ollama pull gemma4:e4b
ollama serve
```

## Esecuzione

```bash
python3 main.py --category metal_nut
```

Opzioni:
```bash
python3 main.py --category screw --skip-train
python3 main.py --category grid --no-explain
```

Web app (bonus):
```bash
python3 app/gradio_app.py
```

## Struttura repository

```
cv-maintenance-anomaly/
├── data/
│   └── setup_datasets.py       # verifica struttura dataset
├── preprocessing/
│   └── preprocess.py           # DataLoader, augmentation, normalizzazione
├── models/
│   ├── hog_svm.py              # HOG + OneClassSVM
│   ├── patchcore.py            # PatchCore + EfficientNet-B0
│   └── gemma_explainer.py      # Gemma4:e4b via Ollama
├── evaluation/
│   └── metrics.py              # AUROC, mAP, confusion matrix, failure analysis
├── app/
│   └── gradio_app.py           # Web app (bonus)
├── results/                    # Output grafici e metriche JSON
├── docs/                       # Technical Analysis Document PDF
├── main.py                     # Pipeline orchestratore
└── requirements.txt
```

## Risultati

| Modello | AUROC | Avg Precision | F1-score |
|---------|-------|---------------|----------|
| HOG+SVM | TBD | TBD | TBD |
| PatchCore | TBD | TBD | TBD |

*Tabella aggiornata dopo esecuzione completa*

## Scelte architetturali

**Perché unsupervised?**  
In contesto industriale reale i difetti sono rari — non si hanno migliaia di immagini difettose per il training. PatchCore e OneClassSVM si addestrano solo su immagini normali.

**Perché Gemma4 locale?**  
Le immagini di componenti industriali contengono informazioni sensibili sull'asset Sidel. Mantenere tutto in locale (via Ollama) garantisce data sovereignty e conformità alle policy aziendali.

## Ethical Considerations

- Nessun dato personale trattato
- Il sistema è un supporto decisionale, non un sostituto del tecnico
- False negative (anomalia non rilevata) = rischio maggiore → soglia conservativa
- Bias potenziale: dataset MVTec non copre tutti i componenti Sidel

## Requirements

- Python 3.10+
- PyTorch 2.2+
- CUDA / Apple MPS (opzionale, migliora performance)
- Ollama con gemma4:e4b (opzionale, per spiegazioni testuali)
