# Dataset Exploration & Validation Report

**Framework:** Evidence-Grounded Hybrid Transformer Hallucination Detector  
**Total Samples:** 40  
**Split Proportions:** Train=28, Val=6, Test=6  

---

## 1. Data Leakage & Integrity Audit

- **Leakage Detected:** `NO (CLEAN)`
- **Train-Val Claim Overlap:** 0
- **Train-Test Claim Overlap:** 0
- **Val-Test Claim Overlap:** 0

---

## 2. Partition Class Balance

| Partition | Total Items | Factual (0) | Hallucinated (1) | Hallucinated % |
| :--- | :--- | :--- | :--- | :--- |
| **Overall** | 40 | 20 | 20 | 50.0% |
| **Train** | 28 | 14 | 14 | 50.0% |
| **Val** | 6 | 3 | 3 | 50.0% |
| **Test** | 6 | 3 | 3 | 50.0% |

---

## 3. Sequence Length & Lexical Statistics (Overall)

| Metric | Mean ± Std | Median | Min - Max | 95th Percentile |
| :--- | :--- | :--- | :--- | :--- |
| **Claim Token Length** | 15.7 ± 2.32 | 15.0 | 12 - 21 | 21 |
| **Evidence Token Length** | 27.0 ± 4.76 | 27.0 | 20 - 37 | 37 |
| **Claim-Evidence Jaccard Overlap** | 0.33 ± 0.16 | 0.33 | 0.08 - 0.71 | 0.67 |
| **Claim Unigram Recall in Evidence** | 0.63 ± 0.22 | 0.67 | 0.21 - 1.0 | 0.93 |

---

## 4. Vocabulary Density

- **Total Ingested Tokens:** 1708
- **Unique Vocabulary Size:** 474
- **Type-Token Ratio (TTR):** 0.2775
