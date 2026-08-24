# HaluEval Dataset Audit

> **Dataset ID**: `pminervini/HaluEval` (HuggingFace QA subset)  
> **Date Audited**: 2026-08-23  

---

## 1. Dataset Overview and Schema

The official **HaluEval QA subset** is a benchmark designed to evaluate large language model hallucinations on factual question answering. The dataset structure matches the following schema:

| Field Name | Data Type | Description |
|---|---|---|
| `knowledge` | `string` | Wikipedia-sourced reference context containing the ground truth. |
| `question` | `string` | Reasoning question (derived from HotpotQA) based on the context. |
| `answer` | `string` | The generated response text (either factual or hallucinated). |
| `hallucination` | `string` | Label: `"yes"` if the answer contains a hallucination; `"no"` if it is factual. |

---

## 2. Dataset Statistics

| Metric | Value |
|---|---|
| **Total Raw Samples** | 10,000 |
| **Valid Samples** | 9,992 |
| **Removed Samples** | 8 |
| **Class Distribution (Raw)** | `"yes"`: 5,010 (50.1%) / `"no"`: 4,990 (49.9%) |
| **Class Distribution (Valid)** | `"yes"`: 5,010 (50.1%) / `"no"`: 4,982 (49.9%) |

### Length Statistics (Character Counts)

| Field | Minimum | Maximum | Median | Mean |
|---|---|---|---|---|
| `knowledge` | 75 | 1,557 | 321 | 344 |
| `question` | 20 | 630 | 90 | 106 |
| `answer` | 1 | 380 | 28 | 40 |

---

## 3. Data Validation & Preprocessing Report

Before running experiments, a full validation was executed on all 10,000 samples. The findings are summarized below:

### Missing and Malformed Fields
- **Missing fields**: None (100% of samples contain all four keys).
- **Empty strings**: None.
- **Malformed JSON/values**: None.
- **Invalid label values**: None (all labels are strictly either `"yes"` or `"no"`).

### Duplicate Examples
- **Exact duplicate samples**: **8 samples** were identified as exact duplicates where both `answer` and `knowledge` matched an existing record.
- **Duplicate indices removed**:
  - Index `4063` (duplicate of `3901`)
  - Index `4112` (duplicate of `1191`)
  - Index `4601` (duplicate of `1021`)
  - Index `5943` (duplicate of `2460`)
  - Index `7093` (duplicate of `3129`)
  - Index `8877` (duplicate of `3006`)
  - Index `8927` (duplicate of `1899`)
  - Index `9427` (duplicate of `3171`)
- **Action taken**: The 8 duplicate samples were removed from the active dataset. The final processed dataset size is **9,992 samples**.

---

## 4. Leakage Prevention Safeguards

To guarantee scientific validity and prevent data leakage, the following safeguards have been implemented:

1. **Stratified Split Separation**:
   - The valid dataset is split into **80% training (7,994 samples)** and **20% testing (1,998 samples)**, stratified by label.
   - Split indices are saved to `data/processed/halueval/split_indices.json` to ensure consistency across all runs.
2. **Strict Fitting Isolation**:
   - TF-IDF vectorizers for baseline models are fitted **only** on the training split.
   - `StandardScaler` for the hybrid meta-classifier is fitted **only** on the training features.
3. **Retrieval Grounding Separation**:
   - Evidence retrieval uses only the local `knowledge` field from the same example, ensuring no cross-sample context or target label information is leaked.
4. **No Test Tuning**:
   - The test set remains completely untouched until final evaluation of each model. Hyperparameters are selected using validation folds within the training split only.
