"""
Experiment Runner: Baseline Models (Exp 1: TF-IDF + LR, Exp 2: TF-IDF + RF).
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
from src.models.baselines.tfidf_logistic_regression import TfidfLogisticRegressionBaseline
from src.models.baselines.random_forest_baseline import TfidfRandomForestBaseline
from src.evaluation.metrics import evaluate_predictions
from src.utils.logger import get_logger

logger = get_logger("run_baseline")


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


def main():
    logger.info("Executing Experiment 1 & 2: Baseline Models...")

    records = generate_synthetic_halueval_dataset(num_samples=40)
    train_recs, val_recs, test_recs = split_dataset(records, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    train_texts = [_get_item_text(r) for r in train_recs]
    train_labels = [_get_item_label(r) for r in train_recs]

    test_texts = [_get_item_text(r) for r in test_recs]
    test_labels = [_get_item_label(r) for r in test_recs]

    # Exp 1: TF-IDF + Logistic Regression
    logger.info("--- Running Exp 1: TF-IDF + Logistic Regression ---")
    lr_model = TfidfLogisticRegressionBaseline()
    lr_model.fit(train_texts, train_labels)
    lr_probs = lr_model.predict_proba(test_texts)
    lr_preds = lr_model.predict(test_texts)
    exp1_metrics = evaluate_predictions(test_labels, lr_preds, lr_probs)
    exp1_metrics["experiment_id"] = "Exp 1"
    exp1_metrics["model_name"] = "TF-IDF + Logistic Regression"

    # Exp 2: TF-IDF + Random Forest
    logger.info("--- Running Exp 2: TF-IDF + Random Forest ---")
    rf_model = TfidfRandomForestBaseline()
    rf_model.fit(train_texts, train_labels)
    rf_probs = rf_model.predict_proba(test_texts)
    rf_preds = rf_model.predict(test_texts)
    exp2_metrics = evaluate_predictions(test_labels, rf_preds, rf_probs)
    exp2_metrics["experiment_id"] = "Exp 2"
    exp2_metrics["model_name"] = "TF-IDF + Random Forest"

    results = {
        "Exp 1": exp1_metrics,
        "Exp 2": exp2_metrics
    }

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "baseline_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Baseline experiment results saved to {out_file}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
