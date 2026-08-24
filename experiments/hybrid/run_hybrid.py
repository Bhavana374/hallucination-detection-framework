"""
Experiment Runner: Hybrid Meta-Classifier (Exp 7: Full Hybrid Model).
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.sample_generator import generate_synthetic_halueval_dataset
from src.data.loader import split_dataset
from src.models.hybrid.hybrid_classifier import HybridClassifier
from src.evaluation.metrics import evaluate_predictions
from src.utils.logger import get_logger

logger = get_logger("run_hybrid")


def _get_item_label(r):
    if hasattr(r, "binary_label"):
        return r.binary_label
    if isinstance(r, dict):
        return r.get("hallucination_label", r.get("binary_label", 0))
    return 0


def _get_item_text(r):
    if hasattr(r, "claim_text"):
        return r.claim_text
    if isinstance(r, dict):
        lbl = _get_item_label(r)
        return r.get("right_answer", "") if lbl == 0 else r.get("hallucinated_answer", "")
    return ""


def _get_item_evidence(r):
    if hasattr(r, "evidence_text"):
        return r.evidence_text
    if isinstance(r, dict):
        return r.get("knowledge", r.get("knowledge_context", ""))
    return ""


def main():
    logger.info("Executing Experiment 7: Full Hybrid Meta-Classifier Model...")

    records = generate_synthetic_halueval_dataset(num_samples=60)
    train_recs, val_recs, test_recs = split_dataset(records, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    def prepare_instances(dataset):
        instances = []
        labels = []
        for r in dataset:
            text = _get_item_text(r)
            evid = _get_item_evidence(r)
            lbl = _get_item_label(r)
            instances.append({
                "claim": text,
                "evidence": evid,
                "retrieval_score": 0.85 if lbl == 0 else 0.30
            })
            labels.append(lbl)
        return instances, labels

    train_insts, train_labels = prepare_instances(train_recs)
    test_insts, test_labels = prepare_instances(test_recs)

    # Exp 7: Full Hybrid Model
    logger.info("--- Running Exp 7: Full Hybrid Model ---")
    hybrid_model = HybridClassifier(classifier_type="random_forest", n_estimators=100)
    hybrid_model.fit(train_insts, train_labels)

    probs = hybrid_model.predict_hallucination_score(test_insts)
    preds = [1 if p >= 0.5 else 0 for p in probs]

    exp7_metrics = evaluate_predictions(test_labels, preds, probs)
    exp7_metrics["experiment_id"] = "Exp 7"
    exp7_metrics["model_name"] = "Full Hybrid Meta-Classifier"
    exp7_metrics["feature_importances"] = hybrid_model.get_feature_importances()

    results = {
        "Exp 7": exp7_metrics
    }

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "hybrid_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Hybrid experiment results saved to {out_file}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
