"""Executable training and evaluation script for Experiment 1: TF-IDF + Logistic Regression Baseline."""

import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_split_from_jsonl
from src.models.baselines.tfidf_logistic_regression import TfidfLogisticRegressionModel
from src.evaluation.metrics import compute_classification_metrics, save_experiment_artifacts
from src.utils.logger import setup_logger
from src.utils.config import load_all_configs

logger = setup_logger("train_baseline_lr")


def main():
    logger.info("============================================================")
    logger.info("STARTING EXPERIMENT 1: TF-IDF + LOGISTIC REGRESSION BASELINE")
    logger.info("============================================================")

    # 1. Load Configurations
    configs = load_all_configs("configs")
    exp_config = {
        "experiment_id": "exp1_tfidf_lr",
        "model_type": "TF-IDF + Logistic Regression",
        "max_features": 5000,
        "ngram_range": [1, 2],
        "c_param": 1.0,
        "random_seed": configs.project.random_seed if "project" in configs else 42,
    }

    # 2. Load Processed Data
    processed_dir = Path("data/processed")
    if not (processed_dir / "train.jsonl").exists():
        logger.error("Processed data not found in data/processed. Running explore_dataset to generate samples...")
        from src.data.sample_generator import generate_benchmark_dataset
        generate_benchmark_dataset()

    split = load_split_from_jsonl(processed_dir)
    logger.info(f"Loaded dataset -> Train: {split.train_size}, Val: {split.val_size}, Test: {split.test_size}")

    train_claims = [it.claim_text for it in split.train]
    train_labels = [it.binary_label for it in split.train]

    val_claims = [it.claim_text for it in split.val]
    val_labels = [it.binary_label for it in split.val]

    test_claims = [it.claim_text for it in split.test]
    test_labels = [it.binary_label for it in split.test]

    # 3. Initialize & Train Model strictly on training split
    model = TfidfLogisticRegressionModel(
        max_features=exp_config["max_features"],
        ngram_range=tuple(exp_config["ngram_range"]),
        c_param=exp_config["c_param"],
        random_state=exp_config["random_seed"],
    )
    model.fit(train_claims, train_labels)

    # 4. Evaluate on Validation Set
    val_start = time.time()
    val_preds = model.predict(val_claims)
    val_probs = model.predict_proba(val_claims)
    val_time = time.time() - val_start

    val_metrics = compute_classification_metrics(
        y_true=val_labels,
        y_pred=val_preds,
        y_prob=val_probs,
        training_time_sec=model.training_time_sec,
        inference_time_sec=val_time,
    )
    logger.info(f"Validation Metrics -> Accuracy: {val_metrics['accuracy']}, F1: {val_metrics['f1_score']}")

    # 5. Evaluate on Test Set
    test_start = time.time()
    test_preds = model.predict(test_claims)
    test_probs = model.predict_proba(test_claims)
    test_time = time.time() - test_start

    test_metrics = compute_classification_metrics(
        y_true=test_labels,
        y_pred=test_preds,
        y_prob=test_probs,
        training_time_sec=model.training_time_sec,
        inference_time_sec=test_time,
    )
    logger.info(f"Test Metrics -> Accuracy: {test_metrics['accuracy']}, F1: {test_metrics['f1_score']}, Macro F1: {test_metrics['f1_macro']}")

    # 6. Format Detailed Test Predictions
    test_prediction_records = []
    for item, pred, prob in zip(split.test, test_preds, test_probs):
        test_prediction_records.append({
            "claim_id": item.claim_id,
            "claim_text": item.claim_text,
            "true_label": item.binary_label,
            "predicted_label": pred,
            "probability_hallucinated": prob,
        })

    # 7. Save Model Checkpoints
    model_ckpt_path = model.save_model("models/checkpoints/exp1_tfidf_lr.pkl")
    model.save_model("models/final/exp1_tfidf_lr.pkl")

    # 8. Save Experiment Artifacts
    exp_output_dir = "experiments/baseline/exp1_tfidf_lr"
    artifacts = save_experiment_artifacts(
        experiment_id="exp1_tfidf_lr",
        output_dir=exp_output_dir,
        config_dict=exp_config,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        test_predictions=test_prediction_records,
    )

    print("\n" + "=" * 60)
    print("EXPERIMENT 1 (TF-IDF + LOGISTIC REGRESSION) SUMMARY")
    print("=" * 60)
    print(f"Test Accuracy:         {test_metrics['accuracy']}")
    print(f"Test Precision:        {test_metrics['precision']}")
    print(f"Test Recall:           {test_metrics['recall']}")
    print(f"Test F1-Score:         {test_metrics['f1_score']}")
    print(f"Test Macro F1:         {test_metrics['f1_macro']}")
    print(f"Training Duration:     {model.training_time_sec:.4f} s")
    print(f"Inference Latency:     {test_metrics['latency_per_sample_ms']:.3f} ms/sample")
    print(f"Model Checkpoint:      {model_ckpt_path}")
    print(f"Artifacts Saved To:    {exp_output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
