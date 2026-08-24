# Comprehensive Model Selection & Scientific Rationale

**Project Title:** Evidence-Grounded Hybrid Transformer Framework for Hallucination Detection in Generative AI  
**Document ID:** `DOC-RES-002`  
**Status:** Approved for Empirical Comparison  
**Academic Context:** Final-Year Research Project & Guide/Judge Review

---

## 1. Architectural Philosophy & Scientific Neutrality

This research project investigates whether combining evidence-grounded Transformer reasoning with dense semantic similarity, factual consistency, and linguistic features in a **Hybrid Meta-Classifier** improves factual hallucination detection over standalone Transformer models.

### Key Distinction in Architectural Roles
We strictly do **not** treat BERT and Random Forest as direct substitutes for the same task. Each model operates at its scientifically justified level:
- **Transformers (BERT / DeBERTa)**: Contextual representations, cross-encoder sequence-pair modeling, and Natural Language Inference (NLI) reasoning over claims and evidence passages.
- **Sentence Transformers**: Dense semantic embeddings and fast bi-encoder retrieval.
- **FAISS**: Hardware-accelerated local vector similarity indexing.
- **Random Forest**: Non-linear supervised ensemble classification over structured, heterogeneous multi-modal feature vectors.

No model is assumed to be universally superior. The experimental results determine empirical performance.

---

## 2. Rationale for Each Candidate Architecture

### 1. TF-IDF + Logistic Regression
- **Scientific Role:** Primary linear classical NLP baseline.
- **Justification:** Establishes the lower-bound benchmark for surface lexical n-gram statistics without contextual understanding.
- **Expected Contribution:** Demonstrates how much performance is attributable to simple word distribution anomalies versus deep semantic grounding.

### 2. TF-IDF + Random Forest
- **Scientific Role:** Non-linear classical ML baseline.
- **Justification:** Directly addresses the project panel's question regarding the original proposal by evaluating non-linear tree ensembles on classical lexical features.
- **Expected Contribution:** Provides an ablation reference point showing the impact of feature engineering alone vs. deep neural representations.

### 3. Standalone BERT (`bert-base-uncased`)
- **Scientific Role:** Transformer-based contextual baseline (Claim-Only).
- **Justification:** Specifically recommended by the project review panel/judges. Employs bidirectional self-attention to capture deep syntactic and semantic context across tokens.
- **Research Question Addressed:** Can BERT detect hallucinations purely from internal linguistic/contextual representations without external evidence grounding?

### 4. Standalone DeBERTa (`microsoft/deberta-v3-base`)
- **Scientific Role:** Advanced Transformer candidate (Claim-Only).
- **Justification:** Incorporates **Disentangled Attention** (representing words using separate content and relative position vectors) and **Enhanced Masked Decoder** pre-training, making it one of the strongest NLI encoders in NLP literature.
- **Research Question Addressed:** Does DeBERTa's advanced relative-positional attention improve standalone hallucination detection over BERT?

### 5. Evidence-Grounded BERT & DeBERTa (Cross-Encoders)
- **Scientific Role:** Contextual Claim-Evidence NLI Verifiers.
- **Justification:** Concatenates claim and retrieved evidence as `[CLS] Evidence [SEP] Claim [SEP]` to allow cross-attention between every token in the claim and every token in the grounding evidence.
- **Output Space:** Calibrated posterior probabilities $P(\text{Entailment}), P(\text{Contradiction}), P(\text{Neutral})$.

### 6. Sentence Transformers (`sentence-transformers/all-MiniLM-L6-v2`)
- **Scientific Role:** Dense Semantic Embedding & Bi-Encoder Retrieval.
- **Justification:** Generates fixed-dimensional (384-d) dense vector embeddings tuned specifically for cosine semantic similarity and sub-millisecond retrieval.
- **Crucial Clarification:** Semantic similarity alone is **not** equivalence to factual entailment; two contradicting statements ("The sky is blue" vs "The sky is green") exhibit high semantic similarity despite being factually opposed. Dense embeddings provide candidate retrieval and semantic distance features to the hybrid model.

### 7. FAISS (Facebook AI Similarity Search)
- **Scientific Role:** High-performance local vector similarity index.
- **Justification:** Provides sub-millisecond top-$k$ nearest neighbor search using inner product / L2 distance without reliance on external paid cloud services or live network calls.

### 8. Evidence-Grounded Hybrid Model (Random Forest Meta-Classifier)
- **Scientific Role:** Multi-feature decision fusion model.
- **Justification:** Combines:
  1. Transformer NLI posteriors ($P_{\text{entail}}, P_{\text{contradict}}, P_{\text{neutral}}$)
  2. Dense semantic similarity metrics (Cosine similarity, Euclidean distance)
  3. Factual & retrieval consistency signals (FAISS retrieval score, named entity overlap)
  4. Linguistic & surface indicators (claim length, token ratio, uncertainty lexical cues)
- **Why Random Forest for Decision Fusion?**
  - Robust against feature scale differences and multi-collinearity.
  - Highly interpretable via Gini and permutation feature importances.
  - Captures non-linear feature interactions without overfitting small tabular datasets.

---

## 3. Summary Matrix of Model Roles

| Model / Component | Role in Pipeline | Input Representation | Target Output |
| :--- | :--- | :--- | :--- |
| **TF-IDF + LR** | Baseline 1 | Bag-of-words / N-grams | Binary Label (0/1) |
| **TF-IDF + RF** | Baseline 2 | Bag-of-words / N-grams | Binary Label (0/1) |
| **BERT Standalone** | Exp 3 | Token IDs (Claim only) | Binary Label (0/1) |
| **DeBERTa Standalone**| Exp 4 | Token IDs (Claim only) | Binary Label (0/1) |
| **Evidence BERT NLI**| Exp 5 | Token IDs (Evidence + Claim) | NLI Probabilities |
| **Evidence DeBERTa** | Exp 6 | Token IDs (Evidence + Claim) | NLI Probabilities |
| **Sentence-BERT** | Retrieval / Embedder | Raw Claim / Evidence Text | 384-d Dense Vector |
| **FAISS Flat/IVF** | Vector Index | 384-d Dense Vectors | Top-$k$ Candidates |
| **Hybrid RF Ensemble**| Exp 7 (Proposed) | Structured Multi-Feature Vector | Final Calibrated Score |
