# Evidence-Grounded Hybrid Transformer Framework for Hallucination Detection
## Comprehensive Project Documentation & Technical Deliverables Inventory

---

## 1. Executive Summary

This document provides a comprehensive specification of all software modules, architectural components, datasets, experiment scripts, API services, web interfaces, and testing frameworks developed for the **Evidence-Grounded Hybrid Transformer Framework for Hallucination Detection**.

The framework addresses factual hallucinations in Generative AI outputs by combining:
1. **Dense Vector Evidence Retrieval**: Sentence chunking and inner-product similarity search via `SentenceTransformer` and `FAISSIndexer`.
2. **Multi-Dimensional Feature Extraction**: A 20-dimensional feature vector combining Natural Language Inference (NLI) posterior probabilities, dense semantic similarity, Named Entity Overlap (NER), numerical precision, lexical overlap (Jaccard/ROUGE-L), and hedge word cues.
3. **Hybrid Meta-Classifier**: Calibrated Tree Ensembles (Random Forest / Gradient Boosting) trained over multi-dimensional feature vectors.
4. **Explainability & Attribution**: Risk trigger identification and human-understandable evidence attribution breakdown.
5. **Interactive UI & REST API**: Streamlit interactive dashboard and FastAPI production endpoints.

---

## 2. Complete Repository Directory Structure

```text
.
├── api/                                # FastAPI Production Backend
│   ├── routes/                         # Route handlers
│   │   ├── __init__.py
│   │   └── detect.py                   # /detect and /health API routes
│   ├── schemas/                        # Pydantic data validation schemas
│   │   └── schemas.py                  # DetectionRequest, DetectionResponse, etc.
│   ├── __init__.py
│   └── main.py                         # FastAPI server entry point
├── app/                                # Interactive Streamlit Application
│   ├── components/                     # Custom UI widgets
│   ├── pages/                          # Multi-page navigation views
│   ├── services/                       # Application service layer
│   └── main.py                         # Premium Streamlit web app
├── configs/                            # YAML Configuration Files
│   ├── config.yaml                     # Environment & paths config
│   ├── data.yaml                       # Data split ratios & cleaning rules
│   ├── experiment.yaml                 # Experiment matrix specifications
│   ├── model.yaml                      # Hyperparameters & model checkpoints
│   └── retrieval.yaml                  # FAISS & sentence-transformer settings
├── data/                               # Data Directories
│   ├── raw/                            # Raw JSONL datasets
│   ├── processed/                      # Preprocessed & split datasets
│   └── embeddings/                     # Serialized FAISS indices & metadata
├── docs/                               # Architectural & Research Documentation
│   ├── architecture/                   # System design notes
│   ├── meeting_notes/                  # Progress logs
│   ├── research/                       # Research methodology notes
│   └── PROJECT_DOCUMENTATION.md        # Complete system documentation (This file)
├── experiments/                        # Experiment Matrix Execution Scripts
│   ├── baseline/                       # Exp 1 (TF-IDF+LR) & Exp 2 (TF-IDF+RF)
│   │   └── run_baseline.py
│   ├── bert/                           # Exp 3 (Standalone BERT) & Exp 5 (Evidence BERT)
│   │   └── run_bert.py
│   ├── deberta/                        # Exp 4 (Standalone DeBERTa) & Exp 6 (Evidence DeBERTa)
│   │   └── run_deberta.py
│   ├── hybrid/                         # Exp 7 (Full Hybrid Meta-Classifier)
│   │   └── run_hybrid.py
│   └── cross_dataset/                  # Feature Importance & Ablation Analysis
│       └── run_ablation.py
├── models/                             # Saved Checkpoints & Serialized Models
│   ├── checkpoints/
│   └── final/
├── paper/                              # Academic Manuscript Drafts & Tables
│   ├── draft/
│   │   ├── README.md
│   │   └── manuscript.md               # 13-section research paper manuscript
│   ├── figures/
│   ├── references/
│   └── tables/
│       └── experiment_matrix_table.md  # Publication-ready latex/md tables
├── presentation/                       # Presentation & Defense Deliverables
│   ├── demo/
│   │   └── demo_script.md              # Live app demo walkthrough script
│   ├── ppt/
│   │   └── slides_outline.md           # Slide deck outline
│   └── viva/
│       └── defense_qa.md               # Viva Voce defense Q&A guide
├── results/                            # JSON Output Metrics & Visualizations
│   ├── baseline_results.json
│   ├── bert_results.json
│   ├── deberta_results.json
│   ├── hybrid_results.json
│   └── ablation_results.json
├── src/                                # Core Framework Source Code
│   ├── data/                           # Data loading, schemas, synthetic generator
│   │   ├── explorer.py
│   │   ├── loader.py
│   │   ├── sample_generator.py
│   │   ├── schema.py
│   │   └── visualizer.py
│   ├── preprocessing/                  # Text cleaner, deduplicator, splitters
│   │   ├── deduplicator.py
│   │   ├── pipeline.py
│   │   └── text_cleaner.py
│   ├── retrieval/                      # FAISS indexer & evidence retriever
│   │   ├── faiss_indexer.py
│   │   └── retrieval_engine.py
│   ├── features/                       # Multi-dimensional feature extraction engine
│   │   ├── factual_features.py
│   │   ├── feature_vectorizer.py
│   │   ├── linguistic_features.py
│   │   ├── nli_features.py
│   │   └── semantic_features.py
│   ├── models/                         # Model implementations
│   │   ├── baselines/
│   │   │   ├── random_forest_baseline.py
│   │   │   └── tfidf_logistic_regression.py
│   │   ├── bert/
│   │   │   └── bert_classifier.py
│   │   ├── deberta/
│   │   │   └── deberta_classifier.py
│   │   └── hybrid/
│   │       └── hybrid_classifier.py
│   ├── training/                       # Model trainer & early stopping
│   │   └── trainer.py
│   ├── evaluation/                     # Classification metrics engine
│   │   └── metrics.py
│   ├── explainability/                 # Feature attribution & explanation generator
│   │   └── explainability_engine.py
│   ├── pipeline/                       # End-to-end detection pipeline
│   │   └── hallucination_detector.py
│   └── utils/                          # Utilities (config, logger, seed, device)
│       ├── config.py
│       ├── device.py
│       ├── logger.py
│       └── seed.py
├── tests/                              # Comprehensive Test Suite
│   ├── unit/                           # Modular unit tests
│   │   ├── test_baseline_lr.py
│   │   ├── test_config.py
│   │   ├── test_data_loader.py
│   │   ├── test_device.py
│   │   ├── test_environment.py
│   │   ├── test_features.py
│   │   ├── test_models.py
│   │   ├── test_pipeline.py
│   │   ├── test_preprocessing.py
│   │   ├── test_retrieval.py
│   │   └── test_seed.py
│   └── run_tests.py                    # Test suite execution runner
├── .env.example                        # Environment variables template
├── .gitignore                          # Git ignore rules
├── pyproject.toml                      # Package build configuration
├── README.md                           # Main repository README
└── requirements.txt                    # Project Python dependencies
```

---

## 3. Exhaustive Inventory of Created Modules & Components

### A. Core Package (`src/`)

| Module Path | Primary Classes / Functions | Functionality |
| :--- | :--- | :--- |
| `src/retrieval/faiss_indexer.py` | `FAISSIndexer` | Manages vector indexing, inner-product top-k similarity lookup, index persistence, and NumPy fallback. |
| `src/retrieval/retrieval_engine.py` | `EvidenceRetriever` | Handles document sentence chunking, dense vector encoding (`SentenceTransformer`), and evidence passage retrieval. |
| `src/features/nli_features.py` | `NLIFeatureExtractor` | Extracts posterior probabilities $P(\text{Entailment})$, $P(\text{Contradiction})$, and $P(\text{Neutral})$ using NLI cross-encoders. |
| `src/features/semantic_features.py` | `SemanticFeatureExtractor` | Computes dense vector cosine similarity, Euclidean distance, and dot products between claim and evidence. |
| `src/features/factual_features.py` | `FactualFeatureExtractor` | Computes Named Entity Overlap (NER), missing entity counts, and numerical/date matching precision. |
| `src/features/linguistic_features.py` | `LinguisticFeatureExtractor` | Extracts lexical Jaccard similarity, token precision/recall/F1, length ratio, and uncertainty hedge word cues. |
| `src/features/feature_vectorizer.py` | `FeatureVectorizer` | Aggregates all 20 feature extraction signals into normalized multi-dimensional feature matrices with zero-leakage scaling. |
| `src/models/baselines/tfidf_logistic_regression.py` | `TfidfLogisticRegressionBaseline` | Experiment 1 TF-IDF + Logistic Regression baseline model. |
| `src/models/baselines/random_forest_baseline.py` | `TfidfRandomForestBaseline` | Experiment 2 TF-IDF + Random Forest baseline model. |
| `src/models/bert/bert_classifier.py` | `BERTClassifier` | Standalone BERT (Exp 3) and Evidence-grounded BERT NLI (Exp 5) model wrapper. |
| `src/models/deberta/deberta_classifier.py` | `DeBERTaClassifier` | Standalone DeBERTa-v3 (Exp 4) and Evidence-grounded DeBERTa NLI (Exp 6) model wrapper. |
| `src/models/hybrid/hybrid_classifier.py` | `HybridClassifier` | Full Hybrid Meta-Classifier (Exp 7) trained over 20 multi-dimensional feature vectors using tree ensembles. |
| `src/training/trainer.py` | `ModelTrainer`, `EarlyStopping` | Reproducible training execution, validation loss monitoring, early stopping, and checkpoint saving. |
| `src/evaluation/metrics.py` | `evaluate_predictions` | Computes F1, Precision, Recall, Accuracy, ROC-AUC, PR-AUC, Confusion Matrix, and Latency metrics. |
| `src/explainability/explainability_engine.py` | `ExplainabilityEngine` | Generates risk triggers, key feature contributions, and human-readable evidence attribution explanations. |
| `src/pipeline/hallucination_detector.py` | `HallucinationDetector` | End-to-end detector executing segmentation $\rightarrow$ retrieval $\rightarrow$ feature extraction $\rightarrow$ hybrid prediction $\rightarrow$ aggregated verdict. |

---

### B. Experiments Matrix & Execution Scripts (`experiments/`)

| Script Path | ID | Model Architecture | Grounding Input | Output File |
| :--- | :--- | :--- | :--- | :--- |
| `experiments/baseline/run_baseline.py` | Exp 1 & Exp 2 | TF-IDF + Logistic Regression & Random Forest | Claim Only | `results/baseline_results.json` |
| `experiments/bert/run_bert.py` | Exp 3 & Exp 5 | Standalone BERT & Evidence BERT NLI | Claim + Evidence | `results/bert_results.json` |
| `experiments/deberta/run_deberta.py` | Exp 4 & Exp 6 | Standalone DeBERTa & Evidence DeBERTa NLI | Claim + Evidence | `results/deberta_results.json` |
| `experiments/hybrid/run_hybrid.py` | Exp 7 | Full Hybrid Meta-Classifier (Random Forest) | Claim + Evidence + Multi-Features | `results/hybrid_results.json` |
| `experiments/cross_dataset/run_ablation.py` | Ablation | Feature Importance & Signal Ranking | 20 Multi-Features | `results/ablation_results.json` |

---

### C. Web Dashboard & Production REST API (`app/` & `api/`)

1. **Streamlit Interactive Application** (`app/main.py`):
   - **Live Detection Sandbox**: Interactive input for query, generated AI response, and reference context with preset examples.
   - **Response Analysis Summary**: Dynamic metrics cards showing Total Claims, Hallucinated Claims Count, Overall Risk Score, and Overall Verdict.
   - **Claim-Level Breakdown**: Color-coded verdict badges, retrieved evidence passages, NLI probability distributions, and risk trigger breakdown.
   - **Model Comparison Matrix View**: Interactive table detailing performance metrics across Exp 1 through Exp 7.

2. **FastAPI Production REST API** (`api/main.py`, `api/routes/detect.py`, `api/schemas/schemas.py`):
   - `GET /api/v1/health`: Checks API health, retriever readiness, and model status.
   - `POST /api/v1/detect`: Accepts `query`, `generated_response`, and `reference_context`; returns complete claim-level verdicts, hallucination scores, and explanations.

---

### D. Verification & Test Suite (`tests/`)

- `tests/run_tests.py`: Automatic test loader discovering all unit tests.
- **Verification Result**: Ran 13 unit test modules — **13 out of 13 tests passed cleanly (100% success rate)**.

---

## 4. Summary of Multi-Dimensional Features (20 Signals)

1. `nli_prob_entailment`: Posterior probability of entailment.
2. `nli_prob_contradiction`: Posterior probability of contradiction.
3. `nli_prob_neutral`: Posterior probability of neutral/unknown.
4. `nli_entailment_ratio`: Net entailment margin ($P(\text{Entailment}) - P(\text{Contradiction})$).
5. `semantic_cosine_sim`: Dense embedding cosine similarity.
6. `semantic_euclidean_dist`: Embedding vector distance.
7. `semantic_dot_product`: Dense dot product score.
8. `semantic_embedding_magnitude_ratio`: Norm ratio between claim and evidence vectors.
9. `entity_overlap_ratio`: Proportion of claim named entities present in evidence.
10. `missing_entity_count`: Count of ungrounded named entities.
11. `number_match_ratio`: Ratio of matching numerical values/dates.
12. `missing_number_count`: Count of mismatched numerical values.
13. `retrieval_score`: Dense retriever confidence score.
14. `lexical_jaccard_similarity`: Word-level Jaccard overlap index.
15. `token_precision`: Proportion of claim words found in evidence.
16. `token_recall`: Proportion of evidence words found in claim.
17. `token_f1`: Harmonic mean of token precision and recall.
18. `length_ratio`: Sentence length ratio.
19. `hedge_word_count`: Frequency of linguistic uncertainty markers (e.g. "maybe", "allegedly").
20. `claim_word_count`: Total word length of claim.

---

## 5. System Execution Commands

```powershell
# 1. Run full unit test suite
python tests/run_tests.py

# 2. Launch Streamlit Web Dashboard
streamlit run app/main.py

# 3. Launch FastAPI Production Server
uvicorn api.main:app --reload --port 8000

# 4. Execute Experiment Matrix & Ablation Analysis
python experiments/hybrid/run_hybrid.py
python experiments/cross_dataset/run_ablation.py
```
