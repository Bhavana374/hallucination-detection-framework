"""
Label Correlation Audit for HaluEval QA features.

Extracts features on a subset of training data and computes statistics 
separately for Factual (no) and Hallucinated (yes) labels to check for 
suspicious correlations or target leakage.
"""

import json
import sys
from pathlib import Path
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_vectorizer import FeatureVectorizer
from src.utils.logger import get_logger

logger = get_logger("label_correlation_audit")


def main():
    logger.info("Initializing FeatureVectorizer...")
    vectorizer = FeatureVectorizer()

    # Load HaluEval dataset raw
    raw_path = PROJECT_ROOT / "data" / "raw" / "halueval" / "qa_samples.jsonl"
    records = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Exclude duplicates in the same order as benchmark script validation
    valid_labels = {"yes", "no"}
    seen_hashes = set()
    valid_records = []
    for rec in records:
        row_hash = json.dumps([rec.get("answer", ""), rec.get("knowledge", "")])
        if row_hash not in seen_hashes and rec.get("hallucination") in valid_labels:
            seen_hashes.add(row_hash)
            valid_records.append(rec)

    # Recreate the train/test splits (using seed 42)
    rng = np.random.RandomState(42)
    pos = [r for r in valid_records if r["hallucination"] == "yes"]
    neg = [r for r in valid_records if r["hallucination"] == "no"]

    rng.shuffle(pos)
    rng.shuffle(neg)

    # Take a representative subset of training data (e.g. 500 factual, 500 hallucinated)
    audit_samples = pos[:500] + neg[:500]
    rng.shuffle(audit_samples)

    logger.info(f"Extracting features for {len(audit_samples)} samples in the audit subset...")

    feature_lists = []
    labels = []

    for i, item in enumerate(audit_samples):
        claim = item["answer"]
        evidence = item["knowledge"]
        label = 1 if item["hallucination"] == "yes" else 0
        
        f_dict = vectorizer.extract_dict(claim, evidence, 0.90)
        feature_lists.append(f_dict)
        labels.append(label)

        if (i + 1) % 100 == 0:
            logger.info(f"  Processed {i + 1}/{len(audit_samples)} samples...")

    # Organize statistics
    feature_names = sorted(list(feature_lists[0].keys()))
    
    factual_features = {name: [] for name in feature_names}
    hallucinated_features = {name: [] for name in feature_names}

    for f_dict, label in zip(feature_lists, labels):
        for name in feature_names:
            val = f_dict[name]
            if label == 0:
                factual_features[name].append(val)
            else:
                hallucinated_features[name].append(val)

    # Compute correlation metrics
    from sklearn.metrics import roc_auc_score

    logger.info("\n=== LABEL CORRELATION AUDIT RESULTS ===")
    print(f"{'Feature Name':<40} | {'Factual Mean':<12} | {'Hallucinated Mean':<17} | {'Diff':<8} | {'ROC-AUC':<8}")
    print("-" * 95)

    audit_results = []
    for name in feature_names:
        fact_vals = np.array(factual_features[name])
        hall_vals = np.array(hallucinated_features[name])

        mean_fact = np.mean(fact_vals)
        mean_hall = np.mean(hall_vals)
        diff = mean_hall - mean_fact

        # Compute ROC-AUC
        all_vals = [f[name] for f in feature_lists]
        try:
            auc = roc_auc_score(labels, all_vals)
            # Center AUC around 0.5 for predictive strength
            auc_print = f"{auc:.4f}"
        except Exception:
            auc = 0.5
            auc_print = "N/A"

        print(f"{name:<40} | {mean_fact:<12.4f} | {mean_hall:<17.4f} | {diff:<8.4f} | {auc_print:<8}")
        audit_results.append({
            "feature": name,
            "mean_factual": float(mean_fact),
            "mean_hallucinated": float(mean_hall),
            "difference": float(diff),
            "roc_auc": float(auc)
        })

    # Save results to json
    out_path = PROJECT_ROOT / "results" / "halueval" / "label_correlation_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(audit_results, f, indent=2)
    logger.info(f"\nAudit results saved to {out_path}")


if __name__ == "__main__":
    main()
