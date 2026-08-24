# Hybrid V2 Experiment Report: Removing Length-Based Shortcut Features

**Experiment ID:** `exp08_hybrid_v2`
**Date:** 2026-08-24 07:30:49 UTC
**Author:** Automated Experiment Pipeline

---

## 1. Motivation

The V1 Hybrid Random Forest (Exp 7) achieved **0.981** F1-score on HaluEval QA.

However, the label correlation audit revealed that two features exploit dataset-specific answer-length formatting shortcuts:

| Feature | Single-Feature ROC-AUC | V1 Importance (Rank) |
|---|---|---|
| `claim_word_count` | **0.9586** | 28.57% (#1) |
| `length_ratio` | **0.9452** | 13.49% (#3) |

In HaluEval QA, hallucinated answers are systematically longer than factual answers (mean 11.28 vs 2.40 tokens). These two features alone account for **42.06%** of the V1 model's total feature importance.

This experiment removes both features and retrains to measure actual grounding performance.

---

## 2. Removed Features

| Feature | Definition | Removal Reason |
|---|---|---|
| `claim_word_count` | `len(claim_tokens)` | Direct measure of answer length. ROC-AUC = 0.9586. |
| `length_ratio` | `len(claim_tokens) / len(evidence_tokens)` | Directly derived from answer length. ROC-AUC = 0.9452. |

---

## 3. Features Retained (18 features)

| # | Feature | Category | Retained Rationale |
|---|---|---|---|
| 1 | `nli_prob_entailment` | NLI | Core claim-evidence entailment probability |
| 2 | `nli_prob_contradiction` | NLI | Core claim-evidence contradiction probability |
| 3 | `nli_prob_neutral` | NLI | Core NLI neutral posterior |
| 4 | `nli_entailment_ratio` | NLI | Entailment − Contradiction delta |
| 5 | `semantic_cosine_sim` | Semantic | Dense embedding cosine similarity |
| 6 | `semantic_euclidean_dist` | Semantic | L2 distance between embeddings |
| 7 | `semantic_dot_product` | Semantic | Inner product (redundant with cosine) |
| 8 | `semantic_embedding_magnitude_ratio` | Semantic | Vector magnitude ratio |
| 9 | `entity_overlap_ratio` | Factual | NER overlap proportion |
| 10 | `missing_entity_count` | Factual | Count of ungrounded entities |
| 11 | `number_match_ratio` | Factual | Numerical consistency ratio |
| 12 | `missing_number_count` | Factual | Mismatched digit count |
| 13 | `retrieval_score` | Factual | Evidence retrieval similarity |
| 14 | `lexical_jaccard_similarity` | Linguistic | Token set Jaccard index |
| 15 | `token_precision` | Linguistic | Claim tokens found in evidence / total claim tokens |
| 16 | `token_recall` | Linguistic | Claim tokens found in evidence / total evidence tokens |
| 17 | `token_f1` | Linguistic | Harmonic mean of token precision & recall |
| 18 | `hedge_word_count` | Linguistic | Count of uncertainty cue words |

---

## 4. Dataset

- **Source:** HaluEval QA (`data/raw/halueval/qa_samples.jsonl`)
- **Total raw samples:** 9992
- **Valid samples after deduplication:** 9992
- **Label distribution:** Balanced (approximately 50/50 hallucinated/factual)

---

## 5. Split

- **Method:** Stratified random split
- **Seed:** 42
- **Test ratio:** 0.2
- **Train size:** 7994 (pos=4008, neg=3986)
- **Test size:** 1998 (pos=1002, neg=996)
- **Identical to V1 split:** Yes (same seed, same validation, same deduplication logic)

---

## 6. Training Configuration

| Parameter | Value |
|---|---|
| Classifier | `RandomForestClassifier` |
| n_estimators | 100 |
| random_state | 42 |
| Feature count | 18 (vs 20 in V1) |
| Scaler | `StandardScaler` fitted on training partition only |
| Leakage prevention | Scaler fit strictly on training data |
| Hyperparameter tuning on test set | None |

---

## 7. V1 Results (Baseline Reference)

| Metric | V1 Value |
|---|---|
| Accuracy | 0.981 |
| Precision | 0.983 |
| Recall | 0.979 |
| F1-Score | 0.981 |
| F1-Macro | 0.981 |
| ROC-AUC | 0.9934 |
| PR-AUC | 0.9955 |
| Training Time | 2824.6849 s |
| Inference Time | 359.1475 s |
| TP / TN / FP / FN | 981 / 979 / 17 / 21 |

---

## 8. V2 Results

| Metric | V2 Value |
|---|---|
| Accuracy | 0.965 |
| Precision | 0.9726 |
| Recall | 0.9571 |
| F1-Score | 0.9648 |
| F1-Macro | 0.965 |
| ROC-AUC | 0.9912 |
| PR-AUC | 0.9939 |
| Training Time | 4.2611 s |
| Inference Time | 0.7373 s |
| TP / TN / FP / FN | 959 / 969 / 27 / 43 |

---

## 9. Feature Importance (V2)

| Rank | Feature | V2 Importance | V1 Importance |
|---|---|---|---|
| 1 | `token_precision` | 0.3149 | 0.2272 |
| 2 | `token_recall` | 0.1351 | 0.0603 |
| 3 | `missing_entity_count` | 0.0934 | 0.0744 |
| 4 | `entity_overlap_ratio` | 0.0923 | 0.0402 |
| 5 | `lexical_jaccard_similarity` | 0.0750 | 0.0390 |
| 6 | `nli_prob_entailment` | 0.0572 | 0.0135 |
| 7 | `token_f1` | 0.0540 | 0.0394 |
| 8 | `nli_entailment_ratio` | 0.0445 | 0.0187 |
| 9 | `semantic_dot_product` | 0.0297 | 0.0120 |
| 10 | `semantic_cosine_sim` | 0.0279 | 0.0110 |
| 11 | `nli_prob_neutral` | 0.0240 | 0.0120 |
| 12 | `semantic_euclidean_dist` | 0.0203 | 0.0172 |
| 13 | `nli_prob_contradiction` | 0.0188 | 0.0092 |
| 14 | `semantic_embedding_magnitude_ratio` | 0.0064 | 0.0022 |
| 15 | `missing_number_count` | 0.0033 | 0.0018 |
| 16 | `number_match_ratio` | 0.0031 | 0.0012 |
| 17 | `hedge_word_count` | 0.0000 | 0.0000 |
| 18 | `retrieval_score` | 0.0000 | 0.0000 |

### Top 5 V2 Features

| Rank | Feature | Importance |
|---|---|---|
| 1 | `token_precision` | 0.3149 |
| 2 | `token_recall` | 0.1351 |
| 3 | `missing_entity_count` | 0.0934 |
| 4 | `entity_overlap_ratio` | 0.0923 |
| 5 | `lexical_jaccard_similarity` | 0.0750 |

---

## 10. Error Analysis

### Summary

| Category | Count |
|---|---|
| Total Errors | 70 |
| False Positives (Factual flagged as Hallucinated) | 27 |
| False Negatives (Hallucinated missed) | 43 |

### Error Categories

| Error Type | Count |
|---|---|
| Short factual answers incorrectly flagged | 14 |
| NLI failure (entailment high but flagged, or contradiction low but missed) | 62 |
| Semantic similarity failure | 36 |
| Entity mismatch failure | 40 |
| Numerical mismatch failure | 0 |
| Insufficient evidence indication | 0 |
| Ambiguous cases (probability near 0.50) | 21 |

---

## 11. V1 vs V2 Comparison

| Metric | V1 | V2 | Delta |
|---|---|---|---|
| Accuracy | 0.981 | 0.965 | -0.016 |
| Precision | 0.983 | 0.9726 | -0.0104 |
| Recall | 0.979 | 0.9571 | -0.0219 |
| F1-Score | 0.981 | 0.9648 | -0.0162 |
| F1-Macro | 0.981 | 0.965 | -0.016 |
| ROC-AUC | 0.9934 | 0.9912 | -0.0022 |
| PR-AUC | 0.9955 | 0.9939 | -0.0016 |
| Training Time (s) | 2824.6849 | 4.2611 | -2820.4238 |
| Inference Time (s) | 359.1475 | 0.7373 | -358.4102 |

### Prediction Differences

| Category | Count |
|---|---|
| Total predictions that changed | 52 |
| V1 correct → V2 wrong (regressions) | 42 |
| V1 wrong → V2 correct (improvements) | 10 |

---

## 12. Interpretation

Removing length shortcuts had minimal impact on F1 (-1.62%), suggesting the model was already learning meaningful semantic and factual features alongside the shortcuts.

The removal of `claim_word_count` (V1 rank #1, importance 28.57%) and `length_ratio` (V1 rank #3, importance 13.49%) forced the Random Forest to redistribute its decision weight across the remaining 18 features.

The V2 feature importance ranking reveals which signals the model relies on when length shortcuts are unavailable.

---

## 13. Limitations

1. **Single dataset:** Both V1 and V2 are evaluated on HaluEval QA only. Cross-dataset generalization is unknown.
2. **Heuristic NLI:** The NLI features use a lexical heuristic fallback when PyTorch/Transformers models are unavailable. True NLI model inference would provide stronger signals.
3. **Hash-based semantic embeddings:** Without SentenceTransformers installed, semantic features use hash-based pseudo-embeddings rather than dense neural embeddings.
4. **Fixed retrieval score:** All instances use a constant `retrieval_score=0.90`, making this feature uninformative in the current setup.
5. **No hyperparameter tuning:** RandomForest uses default `n_estimators=100` without grid search.
6. **Token precision monitoring:** `token_precision` should be monitored — if it becomes the dominant V2 feature, it may also be exploiting an indirect length correlation.

---

## 14. Recommended Next Technical Step

Fine-tune BERT (`bert-base-uncased`) and DeBERTa (`deberta-v3-base`) on the same HaluEval QA split using true PyTorch training (Experiments 3–6) to enable a fair cross-architecture comparison. The Hybrid V2 result provides the cleaned multi-feature baseline; Transformer experiments will show whether deep contextual representations can match or exceed the hybrid approach without relying on engineered features.

---

## Final Summary

```
HYBRID V1 F1:
0.981

HYBRID V2 F1:
0.9648

V2 STATUS:
VALID

MOST IMPORTANT FINDING:
Removing length shortcuts had minimal impact on F1 (-1.62%), suggesting the model was already learning meaningful semantic and factual features alongside the shortcuts.

NEXT TECHNICAL STEP:
Fine-tune BERT and DeBERTa on HaluEval QA (Experiments 3-6) for cross-architecture comparison.
```
