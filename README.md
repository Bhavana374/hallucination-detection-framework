# Evidence-Grounded Hybrid Transformer Framework for Hallucination Detection in Generative AI

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Transformers: HuggingFace](https://img.shields.io/badge/🤗%20Transformers-4.35+-yellow.svg)](https://huggingface.co/transformers/)

A research-grade framework designed to detect, analyze, and explain factual hallucinations in Generative AI outputs using an empirical comparison between classical baselines, standalone Transformer architectures (BERT, DeBERTa), evidence-grounded NLI reasoning, and a multi-dimensional hybrid feature classification model.

---

## 🔬 Core Research Question

> **"Can an evidence-grounded hybrid framework combining Transformer-based claim-evidence reasoning with semantic, factual, and linguistic consistency features improve factual hallucination detection in Generative AI outputs compared with standalone Transformer-based approaches?"**

### Methodological Stance
This framework adopts a strict empirical stance:
- **No bias towards the hybrid model**: We do not assume hybrid superiority.
- **No bias towards BERT/DeBERTa**: We test the judges' hypothesis by fine-tuning and evaluating BERT directly alongside DeBERTa and the Hybrid model under identical leakage-free splits.

---

## 🏗️ System Architecture

```
User Query + Generated Response
              ↓
Response Segmentation & Claim Extraction (Claim IDs, sentence spans)
              ↓
Evidence Retrieval Engine (Dense Sentence Transformer + FAISS Index)
              ↓
Multi-Dimensional Feature Generation
  ├── NLI Posterior Probabilities: P(Entailment), P(Contradiction), P(Unknown)
  ├── Semantic Similarity: Dense Cosine Similarity, Embedding Distances
  ├── Factual & Retrieval Consistency: Evidence ranking scores, Named Entity Overlap
  └── Linguistic Consistency: Lexical Overlap, Length Ratios, Punctuation/Uncertainty cues
              ↓
Hybrid Meta-Classifier (Random Forest / Modular Tree Ensemble)
              ↓
Claim-Level Verdicts & Response-Level Aggregated Hallucination Score
              ↓
Explainability Dashboard & Evidence Attribution (Streamlit UI / FastAPI)
```

---

## 📁 Repository Structure

```
.
├── configs/               # Central YAML configuration files
│   ├── config.yaml        # Global project settings and environment paths
│   ├── data.yaml          # Dataset definitions, split ratios, preprocessing configs
│   ├── model.yaml         # Architecture definitions, hyperparameters, and checkpoints
│   ├── retrieval.yaml     # Embedding models, FAISS indexing parameters, top-k
│   └── experiment.yaml    # Experiment matrix and ablation settings
├── data/                  # Data directories (raw, interim, processed, embeddings)
├── notebooks/             # Exploratory Data Analysis & evaluation walkthroughs
├── src/                   # Core Python package source code
│   ├── data/              # Dataset loaders and validation schemas
│   ├── preprocessing/     # Text normalization, cleaning, and leakage-safe splitters
│   ├── retrieval/         # Dense embedding and FAISS vector retrieval modules
│   ├── features/          # Multi-dimensional feature extraction engine
│   ├── models/            # Baseline models, BERT, DeBERTa, and Hybrid Classifiers
│   ├── training/          # Training pipelines, early stopping, and checkpointing
│   ├── evaluation/        # Metrics computation (F1, PR-AUC, ROC-AUC, latency)
│   ├── explainability/    # Evidence attribution and feature importance visualizers
│   ├── pipeline/          # End-to-end inference and response-level aggregation
│   └── utils/             # Logging, device detection, seed, and config utilities
├── models/                # Checkpoints, serialized classifiers, and exported weights
├── experiments/           # Experiment execution scripts and ablation runs
├── results/               # Raw logs, metrics JSONs, comparison tables, and figures
├── app/                   # Streamlit interactive demonstration application
├── api/                   # FastAPI backend services and schemas
├── tests/                 # Comprehensive unit and integration test suite
├── docs/                  # Architectural documentation and research notes
├── paper/                 # Research paper manuscript drafts (13 sections)
└── presentation/          # PPT slides, demo scripts, and Viva Voce defense Q&A
```

---

## 🚀 Getting Started

### 1. Environment Setup
```bash
# Clone the repository
git clone <repo-url>
cd "Hallucination detection framework"

# Create a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 2. Verify Environment & Run Tests
```bash
# Run unit test suite
pytest tests/unit
```

---

## 🧪 Experiments Matrix

| ID | Experiment Name | Model Architecture | Grounding Input | Output Space |
| :--- | :--- | :--- | :--- | :--- |
| **Exp 1** | Classical Baseline | TF-IDF + Logistic Regression | Claim Only | Binary (Hallucinated / Factual) |
| **Exp 2** | Classical Baseline | TF-IDF + Random Forest | Claim Only | Binary (Hallucinated / Factual) |
| **Exp 3** | Standalone BERT | Fine-tuned `bert-base-uncased` | Claim Only | Binary (Hallucinated / Factual) |
| **Exp 4** | Standalone DeBERTa | Fine-tuned `deberta-v3-base` | Claim Only | Binary (Hallucinated / Factual) |
| **Exp 5** | Evidence BERT | `bert-base-uncased` NLI | Claim + Retrieved Evidence | Entailment / Contradiction / Unknown |
| **Exp 6** | Evidence DeBERTa | `deberta-v3-base` NLI | Claim + Retrieved Evidence | Entailment / Contradiction / Unknown |
| **Exp 7** | Full Hybrid Model | Multi-feature Vector + Random Forest | Claim + Evidence + Multi-Features | Calibrated Hallucination Probability |

---

## 🛡️ Reproducibility & Integrity Guidelines
- **Zero Fabrication**: All results, metrics, and tables are generated directly from code outputs.
- **Leakage Prevention**: Feature vectorizers and scalers are fitted strictly on the training partition.
- **Fixed Random Seeds**: Default seed `42` configured globally across PyTorch, NumPy, and Scikit-Learn.

---

## 👥 Academic Credits & Context
- Final Year Project: "Evidence-Grounded Hybrid Transformer Framework for Hallucination Detection in Generative AI"
- Research Focus: Factual Consistency, Claim-Evidence Grounding, Natural Language Inference, Multi-Feature Hybrid Classification.
