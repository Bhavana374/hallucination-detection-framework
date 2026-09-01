# Cloud Training Guide — Experiments 3–6

> This guide documents how to run full-scale training for Experiments 3–6 on **Google Colab** or another GPU machine. The local laptop is CPU-only and should only be used for smoke tests and 200-sample sanity checks.

---

## Scope Separation

### LOCAL (CPU Laptop) — Already Complete

| Task | Status |
|---|---|
| Model download & caching | Done |
| 200-sample sanity check (all 8 experiments) | Done |
| NLI label mapping fix | Done |
| Evidence-Grounded fusion architecture (Exp 5/6) | Done |
| Config verification | Done |
| Checkpoint loading verification | Done |

### CLOUD/GPU (Google Colab) — To Be Executed

| Task | Status |
|---|---|
| Exp 3: Full BERT fine-tuning (10,000 samples) | Pending |
| Exp 4: Full DeBERTa fine-tuning (10,000 samples) | Pending |
| Exp 5: Full Evidence-Grounded BERT + NLI (10,000 samples) | Pending |
| Exp 6: Full Evidence-Grounded DeBERTa + NLI (10,000 samples) | Pending |

---

## Step 1 — Upload Repository to Colab

### Option A: Upload from local machine

```python
# In a Colab cell:
from google.colab import drive
drive.mount('/content/drive')

# Upload the project folder to Google Drive, then:
!cp -r "/content/drive/MyDrive/Hallucination detection framework" /content/project
%cd /content/project
```

### Option B: Clone from GitHub (if pushed)

```python
!git clone https://github.com/<your-username>/hallucination-detection-framework.git /content/project
%cd /content/project
```

---

## Step 2 — Environment Setup

```python
# Verify GPU is available
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Install dependencies
!pip install -r requirements.txt

# Verify key packages
!python -c "import transformers; print(f'transformers={transformers.__version__}')"
!python -c "import torch; print(f'torch={torch.__version__}, CUDA={torch.cuda.is_available()}')"
```

### Required packages (already in `requirements.txt`)

```
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
scikit-learn>=1.2.0
numpy>=1.24.0
pandas>=2.0.0
faiss-cpu>=1.7.4
PyYAML>=6.0
accelerate>=0.20.0
evaluate>=0.4.0
seaborn>=0.12.0
matplotlib>=3.7.0
```

---

## Step 3 — GPU Configuration

Before running experiments, update `configs/model.yaml` to enable GPU features:

```python
import yaml

with open("configs/model.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Enable fp16 for GPU
cfg["transformers"]["bert"]["fp16"] = True
cfg["transformers"]["deberta"]["fp16"] = True

with open("configs/model.yaml", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print("GPU config applied: fp16=True")
```

---

## Step 4 — Hugging Face Model Loading

Models will auto-download from Hugging Face Hub on first use. No manual download needed.

```python
# Pre-download models to cache (optional, avoids mid-training downloads)
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

print("Downloading BERT...")
AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", num_labels=2)

print("Downloading DeBERTa...")
AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base", num_labels=2)

print("Downloading NLI Cross-Encoder...")
AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-small")
AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-deberta-v3-small")

print("All models downloaded.")
```

---

## Step 5 — Dataset Preparation

The HaluEval dataset (`data/raw/halueval/qa_samples.jsonl`) should be included in the uploaded project. Verify:

```python
import json
from pathlib import Path

data_path = Path("data/raw/halueval/qa_samples.jsonl")
assert data_path.exists(), f"Dataset not found at {data_path}"

with open(data_path) as f:
    records = [json.loads(line) for line in f]

print(f"Total samples: {len(records)}")
print(f"Fields: {list(records[0].keys())}")

# Verify label balance
yes_count = sum(1 for r in records if r["hallucination"] == "yes")
no_count = sum(1 for r in records if r["hallucination"] == "no")
print(f"Hallucinated (yes): {yes_count}")
print(f"Factual (no): {no_count}")
```

Expected output:
```
Total samples: 10000
Fields: ['knowledge', 'question', 'answer', 'hallucination']
Hallucinated (yes): 5010
Factual (no): 4990
```

---

## Step 6 — Train/Validation/Test Split

The benchmark script automatically creates a **70/15/15 stratified split**:

| Split | Approximate Size | Positive (Hallucinated) | Negative (Factual) |
|---|---|---|---|
| Train | ~6,994 | ~3,507 | ~3,487 |
| Validation | ~1,499 | ~752 | ~747 |
| Test | ~1,499 | ~751 | ~748 |

Split is seeded with `random_seed=42` for reproducibility.

---

## Step 7 — Run Experiments

### Exp 3 — Standalone BERT Fine-Tuning

```bash
python scripts/halueval_benchmark.py --full --experiments exp03 --epochs 3 --batch-size 16
```

- **Architecture**: `BertForSequenceClassification` (109M params)
- **Input**: Claim text only (`answer` field)
- **LR**: 2e-5, warmup 10%, AdamW
- **Estimated GPU time**: ~15–20 minutes (T4/V100)

### Exp 4 — Standalone DeBERTa Fine-Tuning

```bash
python scripts/halueval_benchmark.py --full --experiments exp04 --epochs 3 --batch-size 16
```

- **Architecture**: `DebertaV2ForSequenceClassification` (184M params)
- **Input**: Claim text only (`answer` field)
- **LR**: 1.5e-5, warmup 10%, AdamW
- **Estimated GPU time**: ~25–35 minutes (T4/V100)

### Exp 5 — Evidence-Grounded BERT + NLI Fusion

```bash
python scripts/halueval_benchmark.py --full --experiments exp05 --epochs 5 --batch-size 16
```

- **Architecture**: Frozen BERT encoder ([CLS] 768d) + NLI cross-encoder (3d) -> MLP fusion head (771 -> 256 -> 2)
- **Input**: Claim + Gold evidence + NLI scores
- **LR**: 1e-3 (fusion head only)
- **Estimated GPU time**: ~20–30 minutes (T4/V100)

### Exp 6 — Evidence-Grounded DeBERTa + NLI Fusion

```bash
python scripts/halueval_benchmark.py --full --experiments exp06 --epochs 5 --batch-size 16
```

- **Architecture**: Frozen DeBERTa encoder ([CLS] 768d) + NLI cross-encoder (3d) -> MLP fusion head (771 -> 256 -> 2)
- **Input**: Claim + Gold evidence + NLI scores
- **LR**: 1e-3 (fusion head only)
- **Estimated GPU time**: ~30–45 minutes (T4/V100)

### Run All 4 Transformer Experiments Together

```bash
python scripts/halueval_benchmark.py --full --experiments exp03 exp04 exp05 exp06 --epochs 3 --batch-size 16
```

- **Estimated total GPU time**: ~1.5–2.5 hours (T4/V100)

### Run Complete 8-Experiment Benchmark

```bash
python scripts/halueval_benchmark.py --full --epochs 3 --batch-size 16
```

---

## Step 8 — Output & Checkpoint Locations

After training, the following artifacts are generated:

```
results/halueval/full/
|-- exp01_tfidf_lr/
|   |-- config.json
|   |-- metrics.json
|   |-- predictions.csv
|   |-- classification_report.txt
|   |-- confusion_matrix.png
|   |-- run_metadata.json
|-- exp03_bert/
|   |-- checkpoint/
|   |   |-- model.safetensors       # Fine-tuned BERT weights
|   |   |-- config.json
|   |   |-- tokenizer.json
|   |   |-- tokenizer_config.json
|   |-- config.json
|   |-- metrics.json
|   |-- predictions.csv
|   |-- confusion_matrix.png
|-- exp04_deberta/
|   |-- checkpoint/
|   |   |-- model.safetensors       # Fine-tuned DeBERTa weights
|   |-- (same structure)
|-- exp05_evidence_bert/
|   |-- checkpoint/
|   |   |-- fusion_head.pt          # Trained fusion MLP weights
|   |-- (same structure)
|-- exp06_evidence_deberta/
|   |-- checkpoint/
|   |   |-- fusion_head.pt          # Trained fusion MLP weights
|   |-- (same structure)
|-- model_comparison.csv
|-- model_comparison.json
```

Additional outputs:
```
models/checkpoints/hybrid_v2/model.pkl
data/processed/halueval/split_indices.json
data/processed/halueval/validation_report.json
```

---

## Step 9 — Expected Metrics (Full Dataset)

**WARNING**: These are estimated ranges based on published HaluEval benchmarks. The 200-sample sanity check results are NOT final results. Final results require the full 10,000-sample dataset on GPU.

| Experiment | Expected Accuracy | Expected F1 | Expected ROC-AUC |
|---|---|---|---|
| Exp 1 (TF-IDF + LR) | 70-78% | 0.72-0.80 | 0.78-0.85 |
| Exp 2 (TF-IDF + RF) | 65-75% | 0.68-0.78 | 0.75-0.83 |
| Exp 3 (BERT) | 72-82% | 0.74-0.84 | 0.80-0.90 |
| Exp 4 (DeBERTa) | 75-85% | 0.76-0.86 | 0.82-0.92 |
| **Exp 5 (BERT + NLI)** | **82-92%** | **0.84-0.93** | **0.90-0.97** |
| **Exp 6 (DeBERTa + NLI)** | **85-94%** | **0.86-0.95** | **0.92-0.98** |
| Exp 7 (Hybrid RF) | 80-90% | 0.82-0.91 | 0.88-0.96 |
| Exp 8 (Hybrid RF V2) | 80-90% | 0.82-0.91 | 0.88-0.96 |

---

## Step 10 — Copy Results Back to Local Repository

### From Google Drive

```python
# In Colab, after training:
!cp -r /content/project/results/halueval/full "/content/drive/MyDrive/Hallucination detection framework/results/halueval/full"
!cp -r /content/project/models/checkpoints "/content/drive/MyDrive/Hallucination detection framework/models/checkpoints"
!cp /content/project/data/processed/halueval/split_indices.json "/content/drive/MyDrive/Hallucination detection framework/data/processed/halueval/split_indices.json"
```

Then sync from Google Drive to your local machine.

### Direct Download (alternative)

```python
# Zip results for download
!cd /content/project && zip -r /content/training_results.zip results/halueval/full/ models/checkpoints/
from google.colab import files
files.download('/content/training_results.zip')
```

### After Copying Locally

Place the files in their corresponding paths:

```
d:\Hallucination detection framework\
|-- results\halueval\full\          <-- Copy from Colab
|   |-- exp03_bert\
|   |-- exp04_deberta\
|   |-- exp05_evidence_bert\
|   |-- exp06_evidence_deberta\
|   |-- model_comparison.csv
|   |-- model_comparison.json
|-- models\checkpoints\             <-- Copy from Colab
```

---

## Troubleshooting

### Out of Memory (OOM) on Colab Free Tier (T4, 15GB)

Reduce batch size:
```bash
python scripts/halueval_benchmark.py --full --experiments exp04 --epochs 3 --batch-size 8
```

### Colab Disconnects Mid-Training

Run experiments individually to save checkpoints between runs:
```bash
python scripts/halueval_benchmark.py --full --experiments exp03 --epochs 3 --batch-size 16
python scripts/halueval_benchmark.py --full --experiments exp04 --epochs 3 --batch-size 16
python scripts/halueval_benchmark.py --full --experiments exp05 --epochs 5 --batch-size 16
python scripts/halueval_benchmark.py --full --experiments exp06 --epochs 5 --batch-size 16
```

### Verify GPU Is Being Used

```python
import torch
assert torch.cuda.is_available(), "No GPU detected! Go to Runtime > Change runtime type > T4 GPU"
```

---

## Complete Colab Notebook Template

```python
# Cell 1: Mount Drive & Setup
from google.colab import drive
drive.mount('/content/drive')
!cp -r "/content/drive/MyDrive/Hallucination detection framework" /content/project
%cd /content/project

# Cell 2: Install Dependencies
!pip install -r requirements.txt

# Cell 3: Verify GPU
import torch
print(f"CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}")

# Cell 4: Enable fp16
import yaml
with open("configs/model.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["transformers"]["bert"]["fp16"] = True
cfg["transformers"]["deberta"]["fp16"] = True
with open("configs/model.yaml", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

# Cell 5: Run Experiments (one at a time for safety)
!python scripts/halueval_benchmark.py --full --experiments exp03 --epochs 3 --batch-size 16
!python scripts/halueval_benchmark.py --full --experiments exp04 --epochs 3 --batch-size 16
!python scripts/halueval_benchmark.py --full --experiments exp05 --epochs 5 --batch-size 16
!python scripts/halueval_benchmark.py --full --experiments exp06 --epochs 5 --batch-size 16

# Cell 6: Copy Results Back
!cp -r /content/project/results/halueval/full "/content/drive/MyDrive/Hallucination detection framework/results/halueval/full"
!cp -r /content/project/models/checkpoints "/content/drive/MyDrive/Hallucination detection framework/models/checkpoints"
print("Results saved to Google Drive!")
```
