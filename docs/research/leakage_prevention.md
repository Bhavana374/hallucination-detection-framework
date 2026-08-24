# Comprehensive Data Leakage Prevention Protocol

**Project Title:** Evidence-Grounded Hybrid Transformer Framework for Hallucination Detection in Generative AI  
**Document ID:** `DOC-RES-003`  
**Status:** Enforced Across All Pipelines  
**Academic Context:** Final-Year Research Project & Review Panel Safeguards

---

## 1. Threat Model & Leakage Failure Modes in NLP Research

Data leakage occurs when information from outside the training partition influences the model training, feature generation, vector indexing, or hyperparameter selection stages. In hallucination detection and claim verification, the primary leakage risks include:

1. **Partition Contamination**: Overlapping claim instances or prompt-response pairs appearing across both training and test partitions.
2. **Preprocessing & Transformation Leakage**: Fitting text normalizers, TF-IDF vectorizers, or feature scalers across the full combined dataset prior to splitting.
3. **Retrieval Knowledge Base Leakage**: Ingesting test set claims or gold target labels into the retrieval index (FAISS) during training.
4. **Near-Duplicate Memorization**: Lexically perturbed variations of training claims appearing in test sets.
5. **Hyperparameter Overfitting**: Tuning decision thresholds or tree depths on test set evaluations.

---

## 2. Implemented Safeguards & Architectural Enforcement

```
Raw Benchmark Dataset
          │
          ▼
[1. Deterministic Stratified Partitioning] (70% Train, 15% Val, 15% Test)
          │
          ├──► Split Leakage Audit (0 ID overlap, 0 Claim overlap)
          │
          ├──► [Train Partition Only] ──────────┐
          │         │                           │
          │         ▼                           ▼
          │    Fit Vectorizers & Scalers   Build FAISS Index (Corpus Only)
          │         │                           │
          │         ▼                           ▼
          └──► [Val & Test Partitions] ◄────────┘
               (Transform Only, Never Fit)
```

### Protocol 1: Strict Temporal Split Separation
- Splitting occurs **before** any feature extraction, tokenization, or vectorization.
- The `split_dataset` routine executes deterministic stratified partitioning using fixed seed `42`.

### Protocol 2: Partition-Isolated Text Cleaning
- `DatasetPreprocessor` processes `train`, `val`, and `test` partitions strictly in isolation.
- `TextCleaner` applies stateless normalizations (HTML unescaping, Unicode standard NFKC, whitespace collapsing) without cross-document frequency statistics.

### Protocol 3: Classical Vectorizer & Scaler Fitting
- All Scikit-Learn vectorizers (`TfidfVectorizer`) and scalers (`StandardScaler`) are fitted strictly on `train`:
  ```python
  vectorizer.fit(train_claims)
  X_train = vectorizer.transform(train_claims)
  X_val = vectorizer.transform(val_claims)
  X_test = vectorizer.transform(test_claims)
  ```
- No `.fit_transform()` is ever called on validation or test sets.

### Protocol 4: Knowledge Base Retrieval Boundary
- The FAISS vector retrieval index is built strictly from the background reference evidence corpus.
- Test set claims and test response annotations are never indexed or exposed during retrieval indexing.

### Protocol 5: Automated Leakage Auditor
- Every pipeline execution calls `check_split_leakage(split)` which computes set intersections:
  - $\text{Train}_{\text{IDs}} \cap \text{Val}_{\text{IDs}} = \emptyset$
  - $\text{Train}_{\text{IDs}} \cap \text{Test}_{\text{IDs}} = \emptyset$
  - $\text{Val}_{\text{IDs}} \cap \text{Test}_{\text{IDs}} = \emptyset$
  - $\text{Train}_{\text{Claims}} \cap \text{Test}_{\text{Claims}} = \emptyset$
- If any overlap is detected, the pipeline halts with a validation error.

---

## 3. Verification & Compliance Matrix

| Leakage Risk | Safeguard Implemented | Verification Mechanism | Status |
| :--- | :--- | :--- | :--- |
| **Train/Test Contamination** | Stratified split by sample ID | `check_split_leakage()` unit test | **ENFORCED** |
| **TF-IDF Vocabulary Leakage** | `fit()` restricted to train set | Model training isolation tests | **ENFORCED** |
| **Near-Duplicate Overlap** | `ClaimDeduplicator` n-gram audit | N-gram Jaccard filtering | **ENFORCED** |
| **Evidence Corpus Leakage** | Separate knowledge corpus indexing | FAISS index builder isolation | **ENFORCED** |
| **Test Set Threshold Tuning** | Validation-set threshold selection | Fixed evaluation pipeline | **ENFORCED** |
