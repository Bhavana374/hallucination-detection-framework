"""
Experiment Runner: BERT Classifiers (Exp 3: Standalone BERT, Exp 5: Evidence BERT).
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
from src.models.bert.bert_classifier import BERTClassifier
from src.evaluation.metrics import evaluate_predictions
from src.utils.logger import get_logger

logger = get_logger("run_bert")


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
    logger.info("Executing Experiment 3 & 5: BERT Classifiers...")

    records = generate_synthetic_halueval_dataset(num_samples=40)
    train_recs, val_recs, test_recs = split_dataset(records, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    train_texts = [_get_item_text(r) for r in train_recs]
    train_evidences = [_get_item_evidence(r) for r in train_recs]
    train_labels = [_get_item_label(r) for r in train_recs]

    test_texts = [_get_item_text(r) for r in test_recs]
    test_evidences = [_get_item_evidence(r) for r in test_recs]
    test_labels = [_get_item_label(r) for r in test_recs]

    # Exp 3: Standalone BERT
    logger.info("--- Running Exp 3: Standalone BERT ---")
    bert_standalone = BERTClassifier(num_labels=2)
    probs_exp3 = bert_standalone.predict_proba(test_texts)
    y_prob3 = probs_exp3[:, 1] if probs_exp3.shape[1] == 2 else probs_exp3[:, 0]
    preds_exp3 = bert_standalone.predict(test_texts)
    exp3_metrics = evaluate_predictions(test_labels, preds_exp3, y_prob3)
    exp3_metrics["experiment_id"] = "Exp 3"
    exp3_metrics["model_name"] = "Standalone BERT"

    # Exp 5: Evidence BERT NLI
    logger.info("--- Running Exp 5: Evidence BERT ---")
    bert_evidence = BERTClassifier(num_labels=3)
    probs_exp5 = bert_evidence.predict_proba(test_texts, test_evidences)
    y_prob5 = probs_exp5[:, 0]
    preds_exp5 = bert_evidence.predict(test_texts, test_evidences)
    preds_exp5_binary = [1 if p == 0 else 0 for p in preds_exp5]
    exp5_metrics = evaluate_predictions(test_labels, preds_exp5_binary, y_prob5)
    exp5_metrics["experiment_id"] = "Exp 5"
    exp5_metrics["model_name"] = "Evidence BERT NLI"

    results = {
        "Exp 3": exp3_metrics,
        "Exp 5": exp5_metrics
    }

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "bert_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"BERT experiment results saved to {out_file}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
