# Standalone & Grounded Transformer Benchmarking Plan

**Document ID:** `DOC-RES-003`  
**Status:** Under Review  
**Subject:** BERT & DeBERTa-v3 Evaluation Protocol against Hybrid V2

---

## 1. Task Definition & Formatting

We will evaluate standalone and grounded Transformer architectures under a controlled, leakage-free setup to directly compare them against the **Hybrid V2 Classifier** (which achieved **96.48% F1-score** without length shortcuts).

### Input Formulations

To ensure a fair comparison, the models will be evaluated under two distinct tasks:

#### Task A: Standalone Claim-Only Binary Classification (Exp 3 & Exp 4)
- **Objective:** Evaluate whether the model can detect hallucinations purely from internal linguistic or semantic patterns within the generated answer, without access to external evidence.
- **Input Representation:** `claim_text` (extracted from the generated `answer` in HaluEval QA).
- **Target Label:** Binary (`0` for Factual, `1` for Hallucinated).

#### Task B: Evidence-Grounded Claim-Evidence Binary Classification (Exp 5 & Exp 6)
- **Objective:** Evaluate whether the model can detect hallucinations by verifying the claim against the retrieved evidence text.
- **Input Representation:** Pair input `[CLS] claim_text [SEP] evidence_context [SEP]` (concatenating the generated answer with the retrieved knowledge passage).
- **Target Label:** Binary (`0` for Factual, `1` for Hallucinated).

---

## 2. Dataset & Split Setup

To guarantee zero data leakage and a mathematically rigorous comparison, all experiments will reuse the exact same data split as Hybrid V1 and V2:

- **Source Dataset:** HaluEval QA (`data/raw/halueval/qa_samples.jsonl`)
- **Total Valid Samples:** 9,992 (after identical deduplication and validation filters)
- **Train/Test Ratio:** 80/20 split
- **Random Seed:** `42` (stratified random partition)
- **Partition Sizes:**
  - **Train partition:** 7,994 samples (4,008 hallucinated, 3,986 factual)
  - **Test partition:** 1,998 samples (1,002 hallucinated, 996 factual)

---

## 3. Candidate Architectures

### 1. BERT (`bert-base-uncased`)
- **Architecture:** 12-layer, 768-hidden, 12-attention heads, 110M parameters.
- **Role:** Baseline contextual representation model.

### 2. DeBERTa-v3 (`microsoft/deberta-v3-base`)
- **Architecture:** 12-layer, 768-hidden, 12-attention heads, 86M parameters. Uses disentangled attention and enhanced masked language modeling.
- **Role:** Advanced contextual representation model with state-of-the-art NLI capability.

---

## 4. Preprocessing & Tokenization

All tokenization will be performed strictly on-the-fly inside the data loader to prevent file serialization leakage:
- **Maximum Sequence Length:** 256 tokens (truncated/padded dynamically).
- **Tokenizer Setup:** Lowercased for `bert-base-uncased` (uncased), cased/lowercased according to default settings for `deberta-v3-base`.
- **Special Tokens:** Standard sequence classification format `[CLS] text_a [SEP] text_b [SEP]` for pairwise inputs.

---

## 5. Hyperparameter & Training Configuration

To maintain training stability and comparability, we will use identical hyperparameters across BERT and DeBERTa:

| Hyperparameter | Value |
|---|---|
| **Optimizer** | AdamW |
| **Learning Rate** | 2.0e-5 |
| **Weight Decay** | 0.01 |
| **Batch Size** | 16 |
| **Training Epochs** | 3 |
| **Learning Rate Scheduler** | Linear warmup with decay (warmup ratio: 0.1) |
| **Loss Function** | Binary Cross-Entropy (implemented via `CrossEntropyLoss`) |
| **FP16 Mixed Precision** | Enabled (on GPU) / Disabled (on CPU) |
| **Early Stopping** | Patience of 2 epochs based on validation loss |

---

## 6. Computational Feasibility & Profiling Results

We ran a controlled profiling test on the current hardware (CPU-only) with a batch size of 8:

### Profiling Summary (Batch Size = 8, CPU)

| Model | Load Time | 1 Training Batch Time | Full Epoch Time (7,994 samples) | Full 3-Epoch Training Time |
|---|---|---|---|---|
| **BERT** | 3.08s | **4.76 seconds** | ~1.32 hours (4,759 seconds) | **~3.96 hours** (14,277s) |
| **DeBERTa-v3** | 3.64s | **84.95 seconds** | ~23.6 hours (84,950 seconds) | **~70.8 hours** (254,850s) |

### Memory Requirements
- **Memory Footprint:** Loading BERT requires ~440MB; DeBERTa requires ~370MB. Peak RAM during training with batch size 16 on CPU requires ~3.0GB to ~4.5GB, which is fully supported by the system.

### Feasibility Conclusion

* **DeBERTa-v3 CPU training is IMPRACTICAL.** Due to the disentangled attention mechanisms being heavily unoptimized on CPU, the backward pass takes ~80 seconds per batch. Running the full training split (3 epochs) on CPU would take approximately **3 days**.
* **BERT CPU training is FEASIBLE but slow.** It can be trained in approximately **4 hours** on CPU.
* **Recommendation:** We should utilize a cloud GPU (e.g., Google Colab, Kaggle, or a cloud VM instance with a CUDA-enabled GPU such as a T4, V100, or A10G) to run these benchmarks. On a standard T4 GPU with mixed precision (FP16), training time is estimated at **3–5 minutes** per epoch for both models.

---

## 7. Fair Comparison Safeguards

1. **Exact Partition Alignment:** The training and testing partitions must be loaded using the same indices as V2 to ensure test samples are never leaked into the training set of any model.
2. **Zero Zero-Shot Comparison:** Pre-trained models without fine-tuning must not be compared to the trained Hybrid V2 model. Zero-shot predictions must only be reported as an untrained baseline reference.
3. **Parameter Matching:** Standalone and Grounded models must use identical hyperparameters (learning rate, optimizer, number of epochs) to ensure the architectural difference is the sole variable being measured.
4. **Metrics Consistency:** All metrics (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC) must be calculated using the identical python functions defined in `src/evaluation/metrics.py`.

---

## 8. Final Evaluation Protocol

The benchmark will compare the following configurations on the test partition:

| Model Config | Grounding Input | Features Used | Target Outputs |
|---|---|---|---|
| **Exp 1: TF-IDF + LR** | None | Lexical n-grams | Factual (0) / Hallucinated (1) |
| **Exp 2: TF-IDF + RF** | None | Lexical n-grams | Factual (0) / Hallucinated (1) |
| **Exp 3: BERT Standalone** | None | Contextual claim tokens | Factual (0) / Hallucinated (1) |
| **Exp 4: DeBERTa Standalone**| None | Contextual claim tokens | Factual (0) / Hallucinated (1) |
| **Exp 5: Evidence BERT** | Retrieved Evidence | Contextual claim + evidence | Factual (0) / Hallucinated (1) |
| **Exp 6: Evidence DeBERTa** | Retrieved Evidence | Contextual claim + evidence | Factual (0) / Hallucinated (1) |
| **Exp 8: Hybrid V2** | Retrieved Evidence | 18 multi-dimensional features | Factual (0) / Hallucinated (1) |
