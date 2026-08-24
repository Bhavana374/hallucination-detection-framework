"""Comprehensive classification evaluation metrics and experiment artifact tracker."""

import json
import math
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

from src.utils.logger import setup_logger

logger = setup_logger("evaluation_metrics")

try:
    import sklearn.metrics as sk_metrics
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def compute_confusion_matrix(y_true: List[int], y_pred: List[int]) -> Dict[str, int]:
    """Calculate binary confusion matrix counts (TP, FP, TN, FN)."""
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob: Optional[List[float]] = None,
    training_time_sec: float = 0.0,
    inference_time_sec: float = 0.0,
) -> Dict[str, Any]:
    """Compute standard classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC, Latency).

    Args:
        y_true: Ground truth binary labels (0: Factual, 1: Hallucinated).
        y_pred: Predicted binary labels.
        y_prob: Predicted positive class probabilities (optional).
        training_time_sec: Total training execution time in seconds.
        inference_time_sec: Total inference execution time in seconds.

    Returns:
        Structured metrics dictionary.
    """
    n = len(y_true)
    if n == 0:
        return {}

    cm = compute_confusion_matrix(y_true, y_pred)
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]

    accuracy = (tp + tn) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Macro F1 (average across class 0 and class 1)
    precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0.0
    f1_macro = (f1 + f1_0) / 2.0

    # Latency
    latency_per_sample_ms = (inference_time_sec / n) * 1000.0 if n > 0 else 0.0

    # AUC calculation if probabilities are available
    roc_auc = None
    pr_auc = None
    if y_prob is not None and len(y_prob) == n and SKLEARN_AVAILABLE:
        try:
            if len(set(y_true)) > 1:
                roc_auc = round(float(sk_metrics.roc_auc_score(y_true, y_prob)), 4)
                precision_curve, recall_curve, _ = sk_metrics.precision_recall_curve(y_true, y_prob)
                pr_auc = round(float(sk_metrics.auc(recall_curve, precision_curve)), 4)
        except Exception as e:
            logger.warning(f"Error computing AUC scores: {e}")

    return {
        "sample_count": n,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "f1_macro": round(f1_macro, 4),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
        "false_positive_rate": round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0.0,
        "false_negative_rate": round(fn / (fn + tp), 4) if (fn + tp) > 0 else 0.0,
        "training_time_sec": round(training_time_sec, 4),
        "inference_time_sec": round(inference_time_sec, 4),
        "latency_per_sample_ms": round(latency_per_sample_ms, 3),
    }


def save_experiment_artifacts(
    experiment_id: str,
    output_dir: Union[str, Path],
    config_dict: Dict[str, Any],
    val_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
    test_predictions: List[Dict[str, Any]],
) -> Dict[str, str]:
    """Serialize all required research artifacts for an experiment.

    Saves:
      1. experiment_config.json
      2. metrics.json
      3. predictions.csv
      4. classification_report.txt

    Args:
        experiment_id: Unique experiment ID (e.g. 'exp1_tfidf_lr').
        output_dir: Target directory (e.g. 'experiments/baseline/exp1_tfidf_lr').
        config_dict: Hyperparameter & setup configuration.
        val_metrics: Validation metrics dictionary.
        test_metrics: Test metrics dictionary.
        test_predictions: List of sample predictions with IDs, true labels, predicted labels, and probabilities.

    Returns:
        Dictionary of created artifact file paths.
    """
    exp_path = Path(output_dir)
    exp_path.mkdir(parents=True, exist_ok=True)

    artifacts: Dict[str, str] = {}

    # 1. Save experiment_config.json
    cfg_file = exp_path / "experiment_config.json"
    with open(cfg_file, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)
    artifacts["config"] = str(cfg_file)

    # 2. Save metrics.json
    metrics_file = exp_path / "metrics.json"
    metrics_payload = {
        "experiment_id": experiment_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    artifacts["metrics"] = str(metrics_file)

    # 3. Save predictions.csv
    pred_file = exp_path / "predictions.csv"
    with open(pred_file, "w", encoding="utf-8") as f:
        f.write("claim_id,true_label,predicted_label,probability_hallucinated,is_correct\n")
        for pred in test_predictions:
            c_id = pred.get("claim_id", "")
            y_t = pred.get("true_label", 0)
            y_p = pred.get("predicted_label", 0)
            prob = pred.get("probability_hallucinated", 0.0)
            correct = int(y_t == y_p)
            f.write(f"{c_id},{y_t},{y_p},{prob:.4f},{correct}\n")
    artifacts["predictions"] = str(pred_file)

    # 4. Save classification_report.txt
    report_file = exp_path / "classification_report.txt"
    report_lines = [
        f"============================================================",
        f"EXPERIMENT EVALUATION REPORT: {experiment_id}",
        f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"============================================================",
        f"",
        f"--- TEST SET PERFORMANCE ---",
        f"Accuracy:              {test_metrics.get('accuracy', 'N/A')}",
        f"Precision (Class 1):   {test_metrics.get('precision', 'N/A')}",
        f"Recall (Class 1):      {test_metrics.get('recall', 'N/A')}",
        f"F1-Score (Class 1):    {test_metrics.get('f1_score', 'N/A')}",
        f"Macro F1-Score:        {test_metrics.get('f1_macro', 'N/A')}",
        f"ROC-AUC:               {test_metrics.get('roc_auc', 'N/A')}",
        f"PR-AUC:                {test_metrics.get('pr_auc', 'N/A')}",
        f"",
        f"--- COMPUTATIONAL FOOTPRINT ---",
        f"Training Time:         {test_metrics.get('training_time_sec', 'N/A')} s",
        f"Inference Latency:     {test_metrics.get('latency_per_sample_ms', 'N/A')} ms/sample",
        f"",
        f"--- CONFUSION MATRIX ---",
        f"True Positives (TP):   {test_metrics.get('confusion_matrix', {}).get('tp', 'N/A')}",
        f"True Negatives (TN):   {test_metrics.get('confusion_matrix', {}).get('tn', 'N/A')}",
        f"False Positives (FP):  {test_metrics.get('confusion_matrix', {}).get('fp', 'N/A')}",
        f"False Negatives (FN):  {test_metrics.get('confusion_matrix', {}).get('fn', 'N/A')}",
        f"============================================================",
    ]
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    artifacts["classification_report"] = str(report_file)

    logger.info(f"Saved all experiment artifacts to: {exp_path}")
    return artifacts


evaluate_predictions = compute_classification_metrics
