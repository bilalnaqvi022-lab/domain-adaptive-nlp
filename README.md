# 🧠 Domain-Adaptive NLP

> **Aspect-Based Sentiment Analysis on Roman Urdu–English e-commerce reviews using BERT, RoBERTa, DANN, and DAPT.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Transformers-4.x-FFD21E?style=flat-square&logo=huggingface)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## 📌 Overview

This project investigates how NLP models generalize across domains using a suite of adaptation strategies applied to **Roman Urdu–English mixed-language e-commerce reviews** — a low-resource, code-switched setting that presents unique challenges for standard transformer models.

| Strategy | Description |
|----------|-------------|
| **Baseline** | Supervised fine-tuning of BERT / RoBERTa on source domain |
| **DANN** | Domain-Adversarial Neural Network with gradient reversal |
| **DAPT** | Domain-Adaptive Pretraining on unlabeled target domain text |
| **Aspect Sentiment** | Aspect-level sentiment extraction beyond document-level labels |

A **Flask REST API** is included for real-time inference deployment.

---

## 🏗️ Project Structure

```
nlp_project/
│
├── data/
│   ├── raw/                    # Original datasets (unchanged)
│   ├── processed/              # Cleaned and tokenized data
│   └── dataset_loader.py       # Dataset loading and splitting logic
│
├── models/
│   ├── text_classifier.py      # Baseline BERT/RoBERTa classifier
│   ├── aspect_sentiment.py     # Aspect-based sentiment model
│   └── dann.py                 # Domain Adversarial Neural Network
│
├── training/
│   ├── train.py                # Main training loop
│   ├── losses.py               # Loss functions (CE, GRL, adversarial)
│   └── optimizers.py           # Optimizer and scheduler setup
│
├── evaluation/
│   ├── evaluate.py             # Evaluation pipeline
│   └── metrics.py              # Accuracy, F1, domain transfer metrics
│
├── experiments/
│   ├── baseline/               # Baseline run outputs
│   ├── dann/                   # DANN experiment outputs
│   └── dapt/                   # DAPT experiment outputs
│
├── configs/
│   ├── baseline.yaml           # Hyperparameters for baseline
│   ├── dann.yaml               # Hyperparameters for DANN
│   └── dapt.yaml               # Hyperparameters for DAPT
│
├── outputs/                    # Git-ignored
│   ├── checkpoints/
│   ├── logs/
│   └── predictions/
│
├── scripts/
│   ├── run_baseline.py
│   ├── run_dann.py
│   ├── run_dapt.py
│   └── test_model.py
│
├── utils/
│   ├── helpers.py
│   ├── logger.py
│   └── seed.py
│
├── main.py                     # Unified entry point
├── resume.py                   # Resume training from checkpoint
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Clone & Install

```bash
git clone https://github.com/your-username/domain-adaptive-nlp.git
cd domain-adaptive-nlp
pip install -r requirements.txt
```

### Virtual Environment (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

### Virtual Environment (bash/zsh)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

> **Note:** The first run will automatically download model weights and dataset artifacts from HuggingFace Hub.

---

## 🚀 Training

### Baseline (BERT / RoBERTa)

```bash
python scripts/run_baseline.py
```

### DANN — Domain-Adversarial Neural Network

```bash
python scripts/run_dann.py
```

### DAPT — Domain-Adaptive Pretraining

```bash
python scripts/run_dapt.py
```

### Resume from Checkpoint

```bash
python resume.py --checkpoint outputs/checkpoints/dann_epoch5.pt
```

---

## 📊 Evaluation

```bash
python evaluation/evaluate.py --model dann --split test
```

Reported metrics:

- **Accuracy** — overall classification accuracy
- **F1 Score** — macro-averaged across sentiment classes
- **Domain Transfer Performance** — gap between source and target domain accuracy

---

## 📈 Results

| Model    | Source Acc | Target Acc | F1 Score |
|----------|:----------:|:----------:|:--------:|
| Baseline | 82.1%      | 68.4%      | 0.70     |
| DANN     | 83.5%      | 75.2%      | 0.78     |
| **DAPT** | **84.0%**  | **77.1%**  | **0.81** |

> DAPT achieves the best target-domain generalization, narrowing the source–target accuracy gap from **13.7 pp** (baseline) to **6.9 pp**.

---

## 🔬 Key Methods

### 🔹 DANN — Domain Adversarial Neural Network

Uses a **gradient reversal layer (GRL)** to jointly train a label classifier and a domain discriminator. The feature extractor learns representations that are discriminative for sentiment but invariant to domain shift.

### 🔹 DAPT — Domain-Adaptive Pretraining

Continues masked language model pretraining on unlabeled target-domain text (Roman Urdu–English reviews) before task-specific fine-tuning. This aligns the language model's internal representations with the target distribution.

### 🔹 Aspect Sentiment Model

Extends document-level sentiment to **aspect-level granularity** — predicting sentiment separately for entities such as *product quality*, *delivery*, and *price* within a single review.

---

## 🛠️ Requirements

```
Python >= 3.8
torch >= 2.0
transformers >= 4.30
scikit-learn >= 1.3
numpy >= 1.24
flask >= 3.0
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 🌐 Flask API

Start the inference server:

```bash
python app.py
```

Example request:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "delivery bohat slow thi but product quality acha tha"}'
```

Example response:

```json
{
  "sentiment": "mixed",
  "aspects": {
    "delivery": "negative",
    "product_quality": "positive"
  },
  "confidence": 0.87
}
```

---

## 📝 Notes

- The default pipeline uses a small **IMDB subset** for faster local iteration before switching to the full Roman Urdu–English dataset.
- All experiment outputs (checkpoints, logs, predictions) are written to `outputs/` which is excluded from version control via `.gitignore`.
- Reproducibility is handled via `utils/seed.py` — set `SEED=42` in your config or environment.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
