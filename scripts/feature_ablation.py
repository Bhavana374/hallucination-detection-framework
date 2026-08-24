"""
Controlled Feature Ablation Study for the Hybrid Classifier.

Evaluates 8 different feature subsets to isolate the contribution of NLI,
semantic, lexical, linguistic, and factual features to overall performance.
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
from src.evaluation.metrics import compute_classification_metrics
from src.utils.logger import get_logger

logger = get_logger("feature_ablation")


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

    # Take fast subsets (1,000 train, 200 test)
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
    X_train_full = vectorizer.fit_transform(train_instances)
    X_test_full = vectorizer.transform(test_instances)

    feature_names = vectorizer.feature_names
    logger.info(f"Extracted {len(feature_names)} features: {feature_names}")

    # Map groups to column indices
    nli_indices = [i for i, name in enumerate(feature_names) if "nli" in name]
    semantic_indices = [i for i, name in enumerate(feature_names) if "semantic" in name]
    factual_indices = [i for i, name in enumerate(feature_names) if "entity" in name or "number" in name or "retrieval_score" in name]
    lexical_indices = [i for i, name in enumerate(feature_names) if "jaccard" in name or "token" in name]
    linguistic_indices = [i for i, name in enumerate(feature_names) if "length" in name or "word_count" in name or "hedge" in name]
    
    # Combined groups
    lex_ling_indices = lexical_indices + linguistic_indices

    # Define ablation scenarios
    scenarios = {
        "A: Full Hybrid": list(range(len(feature_names))),
        "B: Remove NLI": [i for i in range(len(feature_names)) if i not in nli_indices],
        "C: Remove Semantic": [i for i in range(len(feature_names)) if i not in semantic_indices],
        "D: Remove Lexical": [i for i in range(len(feature_names)) if i not in lexical_indices],
        "E: Remove Linguistic": [i for i in range(len(feature_names)) if i not in linguistic_indices],
        "F: Remove Factual/Numerical/Entity": [i for i in range(len(feature_names)) if i not in factual_indices],
        "G: Use Only NLI": nli_indices,
        "H: Use Only Semantic": semantic_indices
    }

    results = {}
    from sklearn.ensemble import RandomForestClassifier

    logger.info("\n=== RUNNING ABLATION SCENARIOS ===")
    for sc_name, cols in scenarios.items():
        if not cols:
            logger.warning(f"Scenario {sc_name} has empty feature columns. Skipping.")
            continue

        X_tr = X_train_full[:, cols]
        X_te = X_test_full[:, cols]

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_tr, train_labels)

        preds = model.predict(X_te)
        probs = model.predict_proba(X_te)[:, 1].tolist()

        metrics = compute_classification_metrics(test_labels, preds, probs)
        results[sc_name] = metrics

        logger.info(f"{sc_name:<40} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")

    # Save results
    out_path = PROJECT_ROOT / "results" / "halueval" / "ablation_report.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nAblation report saved to: {out_path}")


if __name__ == "__main__":
    main()
