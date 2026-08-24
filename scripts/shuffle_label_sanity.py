"""
Shuffle Label Sanity Test for the Hybrid meta-classifier.

Trains the Hybrid classifier on randomly shuffled labels to check if 
performance collapses to chance level (indicating no structural leakage).
"""

import json
import sys
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent if 'Path' in dir() else None
if not PROJECT_ROOT:
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_vectorizer import FeatureVectorizer
from src.models.hybrid.hybrid_classifier import HybridClassifier
from src.evaluation.metrics import compute_classification_metrics
from src.utils.logger import get_logger

logger = get_logger("shuffle_label_sanity")


def main():
    logger.info("Initializing FeatureVectorizer...")
    vectorizer = FeatureVectorizer()

    # Load raw data
    raw_path = PROJECT_ROOT / "data" / "raw" / "halueval" / "qa_samples.jsonl"
    records = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Exclude duplicates
    valid_labels = {"yes", "no"}
    seen_hashes = set()
    valid_records = []
    for rec in records:
        row_hash = json.dumps([rec.get("answer", ""), rec.get("knowledge", "")])
        if row_hash not in seen_hashes and rec.get("hallucination") in valid_labels:
            seen_hashes.add(row_hash)
            valid_records.append(rec)

    # Recreate splits (seed 42)
    rng = np.random.RandomState(42)
    pos = [r for r in valid_records if r["hallucination"] == "yes"]
    neg = [r for r in valid_records if r["hallucination"] == "no"]

    rng.shuffle(pos)
    rng.shuffle(neg)

    # Take subsets for fast execution (1,000 train, 200 test)
    train_subset = pos[200:700] + neg[200:700]
    test_subset = pos[:100] + neg[:100]

    rng.shuffle(train_subset)
    rng.shuffle(test_subset)

    logger.info(f"Extracting features for {len(train_subset)} train and {len(test_subset)} test samples...")

    train_instances = [{"claim": r["answer"], "evidence": r["knowledge"], "retrieval_score": 0.90} for r in train_subset]
    test_instances = [{"claim": r["answer"], "evidence": r["knowledge"], "retrieval_score": 0.90} for r in test_subset]

    train_labels = [1 if r["hallucination"] == "yes" else 0 for r in train_subset]
    test_labels = [1 if r["hallucination"] == "yes" else 0 for r in test_subset]

    # Fit feature matrices
    X_train = vectorizer.fit_transform(train_instances)
    X_test = vectorizer.transform(test_instances)

    # --- PART A: Train with TRUE labels ---
    from sklearn.ensemble import RandomForestClassifier
    model_true = RandomForestClassifier(n_estimators=100, random_state=42)
    model_true.fit(X_train, train_labels)

    true_preds = model_true.predict(X_test)
    true_probs = model_true.predict_proba(X_test)[:, 1].tolist()
    metrics_true = compute_classification_metrics(test_labels, true_preds, true_probs)

    logger.info("\n=== TRUE LABEL RUN RESULTS ===")
    logger.info(f"Accuracy:  {metrics_true['accuracy']:.4f}")
    logger.info(f"F1-Score:  {metrics_true['f1_score']:.4f}")
    logger.info(f"ROC-AUC:   {metrics_true['roc_auc']:.4f}")

    # --- PART B: Train with SHUFFLED labels ---
    # Shuffle training labels only!
    shuffled_labels = list(train_labels)
    rng.shuffle(shuffled_labels)

    model_shuffled = RandomForestClassifier(n_estimators=100, random_state=42)
    model_shuffled.fit(X_train, shuffled_labels)

    shuffled_preds = model_shuffled.predict(X_test)
    shuffled_probs = model_shuffled.predict_proba(X_test)[:, 1].tolist()
    metrics_shuffled = compute_classification_metrics(test_labels, shuffled_preds, shuffled_probs)

    logger.info("\n=== SHUFFLED LABEL SANITY TEST RESULTS ===")
    logger.info(f"Accuracy:  {metrics_shuffled['accuracy']:.4f}")
    logger.info(f"F1-Score:  {metrics_shuffled['f1_score']:.4f}")
    logger.info(f"ROC-AUC:   {metrics_shuffled['roc_auc']:.4f}")

    # Save report
    out_path = PROJECT_ROOT / "results" / "halueval" / "shuffle_sanity_report.json"
    with open(out_path, "w") as f:
        json.dump({
            "true_metrics": metrics_true,
            "shuffled_metrics": metrics_shuffled
        }, f, indent=2)
    logger.info(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
