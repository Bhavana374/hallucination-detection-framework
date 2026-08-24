"""
Experiment Runner: DeBERTa Classifiers (Exp 4: Standalone DeBERTa, Exp 6: Evidence DeBERTa).
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
from src.models.deberta.deberta_classifier import DeBERTaClassifier
from src.evaluation.metrics import evaluate_predictions
from src.utils.logger import get_logger

logger = get_logger("run_deberta")


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
    logger.info("Executing Experiment 4 & 6: DeBERTa Classifiers...")

    records = generate_synthetic_halueval_dataset(num_samples=40)
    train_recs, val_recs, test_recs = split_dataset(records, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    train_texts = [_get_item_text(r) for r in train_recs]
    train_evidences = [_get_item_evidence(r) for r in train_recs]
    train_labels = [_get_item_label(r) for r in train_recs]

    test_texts = [_get_item_text(r) for r in test_recs]
    test_evidences = [_get_item_evidence(r) for r in test_recs]
    test_labels = [_get_item_label(r) for r in test_recs]

    # Exp 4: Standalone DeBERTa
    logger.info("--- Running Exp 4: Standalone DeBERTa ---")
    deberta_standalone = DeBERTaClassifier(num_labels=2)
    probs_exp4 = deberta_standalone.predict_proba(test_texts)
    y_prob4 = probs_exp4[:, 1] if probs_exp4.shape[1] == 2 else probs_exp4[:, 0]
    preds_exp4 = deberta_standalone.predict(test_texts)
    exp4_metrics = evaluate_predictions(test_labels, preds_exp4, y_prob4)
    exp4_metrics["experiment_id"] = "Exp 4"
    exp4_metrics["model_name"] = "Standalone DeBERTa"

    # Exp 6: Evidence DeBERTa NLI
    logger.info("--- Running Exp 6: Evidence DeBERTa ---")
    deberta_evidence = DeBERTaClassifier(num_labels=3)
    probs_exp6 = deberta_evidence.predict_proba(test_texts, test_evidences)
    y_prob6 = probs_exp6[:, 0]
    preds_exp6 = deberta_evidence.predict(test_texts, test_evidences)
    preds_exp6_binary = [1 if p == 0 else 0 for p in preds_exp6]
    exp6_metrics = evaluate_predictions(test_labels, preds_exp6_binary, y_prob6)
    exp6_metrics["experiment_id"] = "Exp 6"
    exp6_metrics["model_name"] = "Evidence DeBERTa NLI"

    results = {
        "Exp 4": exp4_metrics,
        "Exp 6": exp6_metrics
    }

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "deberta_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"DeBERTa experiment results saved to {out_file}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
