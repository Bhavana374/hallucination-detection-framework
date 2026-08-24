"""
Hybrid V2 Experiment: Remove Length-Based Shortcut Features.

Trains and evaluates a Hybrid Random Forest meta-classifier identical to
Exp 7 (Hybrid V1) but with `claim_word_count` and `length_ratio` excluded
from the 20-dimensional feature vector (resulting in 18 features).

Motivation:
    The V1 label correlation audit found that `claim_word_count` (ROC-AUC=0.9586)
    and `length_ratio` (ROC-AUC=0.9452) exploit a HaluEval QA formatting shortcut
    where hallucinated answers are systematically longer than factual answers.
    Removing these features measures the model's actual grounding capability.

Usage:
    python scripts/run_hybrid_v2.py
"""

import json
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np

# ============================================================
# Project Setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_vectorizer import FeatureVectorizer
from src.models.hybrid.hybrid_classifier import HybridClassifier
from src.evaluation.metrics import compute_classification_metrics
from src.utils.logger import get_logger

logger = get_logger("hybrid_v2")

RANDOM_SEED = 42
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "halueval" / "qa_samples.jsonl"
RESULTS_DIR = PROJECT_ROOT / "results" / "halueval" / "full" / "exp08_hybrid_v2"
V1_RESULTS_DIR = PROJECT_ROOT / "results" / "halueval" / "full" / "exp07_hybrid_rf"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints" / "hybrid_v2"
REPORT_PATH = PROJECT_ROOT / "docs" / "research" / "hybrid_v2_report.md"

# Features to exclude (identified as length-based shortcuts)
EXCLUDED_FEATURES = ["claim_word_count", "length_ratio"]


# ============================================================
# DATA LOADING & VALIDATION (identical to halueval_benchmark.py)
# ============================================================

def load_raw_data(filepath: Path) -> List[Dict[str, str]]:
    """Load raw HaluEval QA JSONL data."""
    if not filepath.exists():
        raise FileNotFoundError(f"Raw data not found at {filepath}.")

    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed line {line_num}: {e}")

    logger.info(f"Loaded {len(records)} raw samples from {filepath}")
    return records


def validate_data(records: List[Dict[str, str]]) -> Tuple[Dict[str, Any], List[Dict]]:
    """Validate dataset integrity — identical logic to halueval_benchmark.py."""
    valid_labels = {"yes", "no"}
    seen_hashes = {}
    valid_records = []
    removed = 0

    for i, rec in enumerate(records):
        issues = []

        # Check required fields
        for field in ["knowledge", "question", "answer", "hallucination"]:
            if field not in rec:
                issues.append(f"missing_{field}")
            elif not rec[field] or not str(rec[field]).strip():
                issues.append(f"empty_{field}")

        # Check label validity
        label = rec.get("hallucination", "")
        if label not in valid_labels:
            issues.append(f"invalid_label:{label}")

        # Check exact duplicates
        row_hash = hashlib.md5(
            (rec.get("answer", "") + rec.get("knowledge", "")).encode()
        ).hexdigest()
        if row_hash in seen_hashes:
            issues.append("exact_duplicate")
        else:
            seen_hashes[row_hash] = i

        if issues:
            removed += 1
        else:
            valid_records.append(rec)

    logger.info(f"Validation: {len(valid_records)} valid / {removed} removed")
    return {"valid": len(valid_records), "removed": removed}, valid_records


def create_stratified_split(
    records: List[Dict[str, str]],
    test_ratio: float = 0.20,
    seed: int = RANDOM_SEED
) -> Tuple[List[Dict], List[Dict], Dict[str, Any]]:
    """Create stratified train/test split — identical logic to halueval_benchmark.py."""
    rng = np.random.RandomState(seed)

    pos = [(i, r) for i, r in enumerate(records) if r["hallucination"] == "yes"]
    neg = [(i, r) for i, r in enumerate(records) if r["hallucination"] == "no"]

    rng.shuffle(pos)
    rng.shuffle(neg)

    n_test_pos = int(len(pos) * test_ratio)
    n_test_neg = int(len(neg) * test_ratio)

    test_pos = pos[:n_test_pos]
    train_pos = pos[n_test_pos:]
    test_neg = neg[:n_test_neg]
    train_neg = neg[n_test_neg:]

    train_data = [r for _, r in train_pos] + [r for _, r in train_neg]
    test_data = [r for _, r in test_pos] + [r for _, r in test_neg]

    rng.shuffle(train_data)
    rng.shuffle(test_data)

    split_meta = {
        "seed": seed,
        "test_ratio": test_ratio,
        "total": len(records),
        "train_size": len(train_data),
        "test_size": len(test_data),
        "train_pos": len(train_pos),
        "train_neg": len(train_neg),
        "test_pos": len(test_pos),
        "test_neg": len(test_neg),
    }

    logger.info(f"Split: train={len(train_data)} (pos={len(train_pos)}, neg={len(train_neg)}), "
                f"test={len(test_data)} (pos={len(test_pos)}, neg={len(test_neg)})")
    return train_data, test_data, split_meta


# ============================================================
# TRAINING & EVALUATION
# ============================================================

def train_hybrid_v2(train_data, test_data):
    """Train Hybrid V2 with excluded length-shortcut features."""
    train_labels = [1 if r["hallucination"] == "yes" else 0 for r in train_data]
    test_labels = [1 if r["hallucination"] == "yes" else 0 for r in test_data]

    # Build instance dicts (same as V1)
    train_instances = [
        {"claim": r["answer"], "evidence": r["knowledge"], "retrieval_score": 0.90}
        for r in train_data
    ]
    test_instances = [
        {"claim": r["answer"], "evidence": r["knowledge"], "retrieval_score": 0.90}
        for r in test_data
    ]

    # Create vectorizer with excluded features
    vectorizer = FeatureVectorizer(excluded_features=EXCLUDED_FEATURES)
    model = HybridClassifier(
        classifier_type="random_forest",
        n_estimators=100,
        random_state=RANDOM_SEED,
        vectorizer=vectorizer
    )

    # Train
    logger.info(f"Training Hybrid V2 (excluded: {EXCLUDED_FEATURES})...")
    t0 = time.time()
    model.fit(train_instances, train_labels)
    train_time = time.time() - t0
    logger.info(f"Training completed in {train_time:.2f}s")

    # Evaluate
    logger.info("Evaluating on test set...")
    t0 = time.time()
    test_probs = model.predict_hallucination_score(test_instances)
    infer_time = time.time() - t0
    test_preds = [1 if p >= 0.5 else 0 for p in test_probs]

    metrics = compute_classification_metrics(test_labels, test_preds, test_probs, train_time, infer_time)

    # Feature importances
    feat_importances = model.get_feature_importances()

    logger.info(f"V2 Results: Acc={metrics['accuracy']}, F1={metrics['f1_score']}, "
                f"ROC-AUC={metrics.get('roc_auc', 'N/A')}, PR-AUC={metrics.get('pr_auc', 'N/A')}")

    return model, metrics, feat_importances, test_labels, test_preds, test_probs


# ============================================================
# ARTIFACT SAVING
# ============================================================

def save_experiment_artifacts(model, metrics, feat_importances, split_meta,
                              test_data, test_labels, test_preds, test_probs):
    """Save all experiment artifacts."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. config.json
    config = {
        "model": "Hybrid Random Forest V2",
        "experiment_id": "exp08_hybrid_v2",
        "description": "Hybrid V2 without length-based shortcut features",
        "input_type": "claim+evidence+features",
        "n_estimators": 100,
        "n_features": len(model.vectorizer.feature_names),
        "excluded_features": EXCLUDED_FEATURES,
        "feature_names": model.vectorizer.feature_names,
        "feature_importances": feat_importances,
        "seed": RANDOM_SEED,
        "train_size": split_meta["train_size"],
        "test_size": split_meta["test_size"],
        "split_meta": split_meta,
    }
    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    # 2. metrics.json
    metrics_payload = {
        "experiment_id": "exp08_hybrid_v2",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_metrics": metrics,
    }
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    # 3. predictions.csv
    with open(RESULTS_DIR / "predictions.csv", "w", encoding="utf-8") as f:
        f.write("index,answer_text,true_label,predicted_label,probability,is_correct\n")
        for i, (rec, yt, yp, prob) in enumerate(zip(test_data, test_labels, test_preds, test_probs)):
            ans = rec["answer"].replace('"', '""')
            f.write(f'{i},"{ans}",{yt},{yp},{prob:.6f},{int(yt == yp)}\n')

    # 4. classification_report.txt
    lines = [
        f"{'='*60}",
        f"EXPERIMENT: exp08_hybrid_v2",
        f"Description: Hybrid V2 without length-based shortcut features",
        f"Excluded Features: {', '.join(EXCLUDED_FEATURES)}",
        f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"{'='*60}",
        "",
        f"Accuracy:            {metrics.get('accuracy', 'N/A')}",
        f"Precision (Class 1): {metrics.get('precision', 'N/A')}",
        f"Recall (Class 1):    {metrics.get('recall', 'N/A')}",
        f"F1-Score (Class 1):  {metrics.get('f1_score', 'N/A')}",
        f"F1-Macro:            {metrics.get('f1_macro', 'N/A')}",
        f"ROC-AUC:             {metrics.get('roc_auc', 'N/A')}",
        f"PR-AUC:              {metrics.get('pr_auc', 'N/A')}",
        "",
        f"Training Time:       {metrics.get('training_time_sec', 'N/A')} s",
        f"Inference Time:      {metrics.get('inference_time_sec', 'N/A')} s",
        f"Latency/Sample:      {metrics.get('latency_per_sample_ms', 'N/A')} ms",
        "",
        f"Confusion Matrix:",
        f"  TP: {metrics.get('confusion_matrix', {}).get('tp', 'N/A')}",
        f"  TN: {metrics.get('confusion_matrix', {}).get('tn', 'N/A')}",
        f"  FP: {metrics.get('confusion_matrix', {}).get('fp', 'N/A')}",
        f"  FN: {metrics.get('confusion_matrix', {}).get('fn', 'N/A')}",
        f"{'='*60}",
    ]
    with open(RESULTS_DIR / "classification_report.txt", "w") as f:
        f.write("\n".join(lines))

    # 5. confusion_matrix.png
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(test_labels, test_preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Factual", "Hallucinated"],
                    yticklabels=["Factual", "Hallucinated"])
        plt.ylabel("Actual Label")
        plt.xlabel("Predicted Label")
        plt.title("Confusion Matrix - Hybrid V2 (No Length Shortcuts)")
        plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved confusion_matrix.png")
    except Exception as e:
        logger.warning(f"Could not generate confusion matrix plot: {e}")

    # 6. feature_importance.csv
    with open(RESULTS_DIR / "feature_importance.csv", "w", encoding="utf-8") as f:
        f.write("rank,feature,importance\n")
        for rank, (feat, imp) in enumerate(feat_importances.items(), 1):
            f.write(f"{rank},{feat},{imp:.6f}\n")

    # 7. run_metadata.json
    try:
        import torch
        pytorch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        device = "cuda" if cuda_available else "cpu"
    except ImportError:
        pytorch_version = "N/A"
        cuda_available = False
        device = "cpu"

    meta = {
        "experiment_id": "exp08_hybrid_v2",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device,
        "pytorch_version": pytorch_version,
        "cuda_available": cuda_available,
        "random_seed": RANDOM_SEED,
        "python_version": sys.version,
        "excluded_features": EXCLUDED_FEATURES,
    }
    with open(RESULTS_DIR / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # 8. Save model checkpoint
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(CHECKPOINT_DIR / "model.pkl")
    logger.info(f"Model checkpoint saved to {CHECKPOINT_DIR / 'model.pkl'}")

    logger.info(f"All experiment artifacts saved to {RESULTS_DIR}")


# ============================================================
# V1 VS V2 COMPARISON
# ============================================================

def generate_v1_v2_comparison(v2_metrics, v2_importances):
    """Generate V1 vs V2 comparison artifacts."""
    # Load V1 metrics
    v1_metrics_path = V1_RESULTS_DIR / "metrics.json"
    with open(v1_metrics_path, "r") as f:
        v1_data = json.load(f)
    v1_metrics = v1_data["test_metrics"]

    # Load V1 config for feature importances
    v1_config_path = V1_RESULTS_DIR / "config.json"
    with open(v1_config_path, "r") as f:
        v1_config = json.load(f)
    v1_importances = v1_config.get("feature_importances", {})

    comparison = {
        "experiment_pair": ["exp07_hybrid_rf (V1)", "exp08_hybrid_v2 (V2)"],
        "methodology_difference": "V2 excludes claim_word_count and length_ratio features",
        "v1_feature_count": 20,
        "v2_feature_count": 18,
        "excluded_features": EXCLUDED_FEATURES,
        "metrics_comparison": {},
    }

    metric_keys = ["accuracy", "precision", "recall", "f1_score", "f1_macro",
                    "roc_auc", "pr_auc", "training_time_sec", "inference_time_sec",
                    "latency_per_sample_ms"]

    for key in metric_keys:
        v1_val = v1_metrics.get(key, "N/A")
        v2_val = v2_metrics.get(key, "N/A")
        delta = None
        if isinstance(v1_val, (int, float)) and isinstance(v2_val, (int, float)):
            delta = round(v2_val - v1_val, 4)
        comparison["metrics_comparison"][key] = {
            "v1": v1_val,
            "v2": v2_val,
            "delta": delta
        }

    # Confusion matrix comparison
    v1_cm = v1_metrics.get("confusion_matrix", {})
    v2_cm = v2_metrics.get("confusion_matrix", {})
    comparison["confusion_matrix_comparison"] = {
        "v1": v1_cm,
        "v2": v2_cm,
    }

    # Feature importance comparison
    importance_comparison = []
    for feat, v2_imp in v2_importances.items():
        v1_imp = v1_importances.get(feat, 0.0)
        importance_comparison.append({
            "feature": feat,
            "v1_importance": round(v1_imp, 6),
            "v2_importance": round(v2_imp, 6),
            "delta": round(v2_imp - v1_imp, 6),
        })
    comparison["feature_importance_comparison"] = importance_comparison

    # Save comparison
    with open(RESULTS_DIR / "v1_v2_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    logger.info("Saved v1_v2_comparison.json")
    return comparison, v1_metrics, v1_importances


# ============================================================
# PREDICTION DIFF ANALYSIS
# ============================================================

def generate_prediction_diff(test_data, test_labels, v2_preds, v2_probs):
    """Find examples where V1 and V2 predictions differ."""
    # Load V1 predictions
    import csv as csv_module
    v1_pred_path = V1_RESULTS_DIR / "predictions.csv"
    v1_predictions = []
    with open(v1_pred_path, newline="", encoding="utf-8") as f:
        reader = csv_module.reader(f)
        next(reader)  # skip header
        for row in reader:
            # Columns: index, answer_text, true_label, predicted_label, probability, is_correct
            v1_predictions.append({
                "true_label": int(row[2]),
                "predicted_label": int(row[3]),
                "probability": float(row[4]),
                "is_correct": int(row[5])
            })

    # Compare
    v1_correct_v2_wrong = []
    v1_wrong_v2_correct = []
    both_wrong = []
    prediction_changes = []

    for i in range(min(len(v1_predictions), len(test_data))):
        v1 = v1_predictions[i]
        v2_pred = v2_preds[i]
        v2_prob = v2_probs[i]
        yt = test_labels[i]
        item = test_data[i]

        v1_correct = (v1["predicted_label"] == yt)
        v2_correct = (v2_pred == yt)

        if v1["predicted_label"] != v2_pred:
            entry = {
                "index": i,
                "question": item["question"],
                "answer": item["answer"][:200],
                "true_label": yt,
                "v1_predicted": v1["predicted_label"],
                "v1_probability": v1["probability"],
                "v2_predicted": v2_pred,
                "v2_probability": round(v2_prob, 6),
            }
            prediction_changes.append(entry)

            if v1_correct and not v2_correct:
                v1_correct_v2_wrong.append(entry)
            elif not v1_correct and v2_correct:
                v1_wrong_v2_correct.append(entry)
            elif not v1_correct and not v2_correct:
                both_wrong.append(entry)

    diff_report = {
        "total_prediction_changes": len(prediction_changes),
        "v1_correct_v2_wrong": len(v1_correct_v2_wrong),
        "v1_wrong_v2_correct": len(v1_wrong_v2_correct),
        "both_wrong_but_different": len(both_wrong),
        "v1_correct_v2_wrong_examples": v1_correct_v2_wrong[:50],
        "v1_wrong_v2_correct_examples": v1_wrong_v2_correct[:50],
        "prediction_changes_all": prediction_changes[:100],
    }

    with open(RESULTS_DIR / "prediction_diff.json", "w", encoding="utf-8") as f:
        json.dump(diff_report, f, indent=2)

    logger.info(f"Prediction diff: {len(prediction_changes)} changes, "
                f"V1ok_V2wrong={len(v1_correct_v2_wrong)}, V1wrong_V2ok={len(v1_wrong_v2_correct)}")
    return diff_report


# ============================================================
# ERROR ANALYSIS
# ============================================================

def generate_error_analysis(test_data, test_labels, test_preds, test_probs, vectorizer):
    """Categorized error analysis for V2."""
    false_positives = []
    false_negatives = []
    correct_positives = []
    correct_negatives = []

    # Categorized error buckets
    error_categories = {
        "short_factual_flagged_as_hallucination": [],
        "nli_failure": [],
        "semantic_similarity_failure": [],
        "entity_mismatch_failure": [],
        "numerical_mismatch_failure": [],
        "insufficient_evidence_indication": [],
        "ambiguous_cases": [],
    }

    for i, (rec, yt, yp, prob) in enumerate(zip(test_data, test_labels, test_preds, test_probs)):
        f_dict = vectorizer.extract_dict(rec["answer"], rec["knowledge"], 0.90)

        info = {
            "index": i,
            "question": rec["question"],
            "answer": rec["answer"],
            "evidence": rec["knowledge"][:300],
            "true_label": yt,
            "predicted_label": yp,
            "probability": round(prob, 6),
            "retrieval_score": 0.90,
            "nli_probabilities": {
                "entailment": round(float(f_dict.get("nli_prob_entailment", 0.0)), 4),
                "contradiction": round(float(f_dict.get("nli_prob_contradiction", 0.0)), 4),
                "neutral": round(float(f_dict.get("nli_prob_neutral", 0.0)), 4),
            },
            "features": {
                "semantic_cosine_sim": round(float(f_dict.get("semantic_cosine_sim", 0.0)), 4),
                "entity_overlap_ratio": round(float(f_dict.get("entity_overlap_ratio", 0.0)), 4),
                "missing_entity_count": round(float(f_dict.get("missing_entity_count", 0.0)), 4),
                "number_match_ratio": round(float(f_dict.get("number_match_ratio", 0.0)), 4),
                "missing_number_count": round(float(f_dict.get("missing_number_count", 0.0)), 4),
                "lexical_jaccard": round(float(f_dict.get("lexical_jaccard_similarity", 0.0)), 4),
                "token_precision": round(float(f_dict.get("token_precision", 0.0)), 4),
                "token_recall": round(float(f_dict.get("token_recall", 0.0)), 4),
                "hedge_word_count": round(float(f_dict.get("hedge_word_count", 0.0)), 4),
            },
        }

        is_error = (yt != yp)

        if yt == 0 and yp == 1:
            false_positives.append(info)
            # Categorize
            answer_words = len(rec["answer"].split())
            if answer_words <= 5:
                error_categories["short_factual_flagged_as_hallucination"].append(info)
            if f_dict.get("nli_prob_entailment", 0) > 0.5 and yp == 1:
                error_categories["nli_failure"].append(info)
            if f_dict.get("semantic_cosine_sim", 0) > 0.3 and yp == 1:
                error_categories["semantic_similarity_failure"].append(info)
            if f_dict.get("entity_overlap_ratio", 1.0) < 0.5:
                error_categories["entity_mismatch_failure"].append(info)
            if f_dict.get("number_match_ratio", 1.0) < 0.5:
                error_categories["numerical_mismatch_failure"].append(info)
            if 0.40 <= prob <= 0.60:
                error_categories["ambiguous_cases"].append(info)

        elif yt == 1 and yp == 0:
            false_negatives.append(info)
            if f_dict.get("nli_prob_contradiction", 0) < 0.3:
                error_categories["nli_failure"].append(info)
            if f_dict.get("semantic_cosine_sim", 0) > 0.5:
                error_categories["semantic_similarity_failure"].append(info)
            if f_dict.get("entity_overlap_ratio", 0) > 0.8:
                error_categories["entity_mismatch_failure"].append(info)
            if 0.40 <= prob <= 0.60:
                error_categories["ambiguous_cases"].append(info)
            if f_dict.get("nli_prob_entailment", 0) < 0.2 and f_dict.get("nli_prob_contradiction", 0) < 0.2:
                error_categories["insufficient_evidence_indication"].append(info)

        elif yt == 1 and yp == 1:
            if len(correct_positives) < 30:
                correct_positives.append(info)
        elif yt == 0 and yp == 0:
            if len(correct_negatives) < 30:
                correct_negatives.append(info)

    error_analysis = {
        "summary": {
            "total_errors": len(false_positives) + len(false_negatives),
            "false_positives_count": len(false_positives),
            "false_negatives_count": len(false_negatives),
        },
        "error_categories": {
            cat: {
                "count": len(examples),
                "examples": examples[:20],
            }
            for cat, examples in error_categories.items()
        },
        "false_positives": false_positives[:50],
        "false_negatives": false_negatives[:50],
        "correct_positives_sample": correct_positives[:20],
        "correct_negatives_sample": correct_negatives[:20],
    }

    with open(RESULTS_DIR / "error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(error_analysis, f, indent=2)

    logger.info(f"Error analysis: {len(false_positives)} FP, {len(false_negatives)} FN")
    return error_analysis


# ============================================================
# RESEARCH REPORT GENERATION
# ============================================================

def generate_report(v2_metrics, v1_metrics, v2_importances, v1_importances,
                    comparison, diff_report, error_analysis, split_meta):
    """Generate docs/research/hybrid_v2_report.md."""

    v1_f1 = v1_metrics.get("f1_score", "N/A")
    v2_f1 = v2_metrics.get("f1_score", "N/A")
    v1_acc = v1_metrics.get("accuracy", "N/A")
    v2_acc = v2_metrics.get("accuracy", "N/A")

    # Top 5 V2 feature importances
    top_features = list(v2_importances.items())[:5]
    top_feat_lines = "\n".join([f"| {i+1} | `{feat}` | {imp:.4f} |" for i, (feat, imp) in enumerate(top_features)])

    # Full V2 feature importance table
    all_feat_lines = "\n".join([f"| {i+1} | `{feat}` | {imp:.4f} | {v1_importances.get(feat, 0.0):.4f} |"
                                 for i, (feat, imp) in enumerate(v2_importances.items())])

    # Determine V2 status
    if isinstance(v2_f1, (int, float)):
        if v2_f1 >= 0.85:
            v2_status = "VALID"
        elif v2_f1 >= 0.70:
            v2_status = "VALID — Moderate performance drop, expected after removing shortcut features"
        else:
            v2_status = "NEEDS INVESTIGATION — Significant performance drop suggests model was heavily reliant on length shortcuts"
    else:
        v2_status = "NEEDS INVESTIGATION"

    # Determine most important finding
    top_feat_name = top_features[0][0] if top_features else "unknown"
    top_feat_imp = top_features[0][1] if top_features else 0.0

    # F1 delta
    f1_delta = None
    if isinstance(v1_f1, (int, float)) and isinstance(v2_f1, (int, float)):
        f1_delta = round(v2_f1 - v1_f1, 4)
        f1_delta_pct = f"{f1_delta * 100:+.2f}%"
    else:
        f1_delta_pct = "N/A"

    # Error analysis summary
    fp_count = error_analysis["summary"]["false_positives_count"]
    fn_count = error_analysis["summary"]["false_negatives_count"]
    total_errors = error_analysis["summary"]["total_errors"]

    # Short factual flagged
    short_flagged = error_analysis["error_categories"].get("short_factual_flagged_as_hallucination", {}).get("count", 0)

    # Prediction diff summary
    total_changes = diff_report["total_prediction_changes"]
    v1_correct_v2_wrong = diff_report["v1_correct_v2_wrong"]
    v1_wrong_v2_correct = diff_report["v1_wrong_v2_correct"]

    # Finding
    if isinstance(v2_f1, (int, float)) and isinstance(v1_f1, (int, float)):
        if abs(v2_f1 - v1_f1) < 0.02:
            most_important_finding = f"Removing length shortcuts had minimal impact on F1 ({f1_delta_pct}), suggesting the model was already learning meaningful semantic and factual features alongside the shortcuts."
        elif v2_f1 < v1_f1:
            most_important_finding = f"Removing length shortcuts caused a {f1_delta_pct} F1 drop, confirming the V1 model was partially reliant on answer-length formatting artifacts. The V2 model now relies primarily on `{top_feat_name}` (importance: {top_feat_imp:.4f}) as its strongest signal."
        else:
            most_important_finding = f"V2 F1 ({v2_f1:.4f}) is higher than V1 ({v1_f1:.4f}), suggesting length features were adding noise."
    else:
        most_important_finding = "Unable to determine — metrics not available."

    report = f"""# Hybrid V2 Experiment Report: Removing Length-Based Shortcut Features

**Experiment ID:** `exp08_hybrid_v2`
**Date:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}
**Author:** Automated Experiment Pipeline

---

## 1. Motivation

The V1 Hybrid Random Forest (Exp 7) achieved **{v1_f1}** F1-score on HaluEval QA.

However, the label correlation audit revealed that two features exploit dataset-specific answer-length formatting shortcuts:

| Feature | Single-Feature ROC-AUC | V1 Importance (Rank) |
|---|---|---|
| `claim_word_count` | **0.9586** | 28.57% (#1) |
| `length_ratio` | **0.9452** | 13.49% (#3) |

In HaluEval QA, hallucinated answers are systematically longer than factual answers (mean 11.28 vs 2.40 tokens). These two features alone account for **42.06%** of the V1 model's total feature importance.

This experiment removes both features and retrains to measure actual grounding performance.

---

## 2. Removed Features

| Feature | Definition | Removal Reason |
|---|---|---|
| `claim_word_count` | `len(claim_tokens)` | Direct measure of answer length. ROC-AUC = 0.9586. |
| `length_ratio` | `len(claim_tokens) / len(evidence_tokens)` | Directly derived from answer length. ROC-AUC = 0.9452. |

---

## 3. Features Retained (18 features)

| # | Feature | Category | Retained Rationale |
|---|---|---|---|
| 1 | `nli_prob_entailment` | NLI | Core claim-evidence entailment probability |
| 2 | `nli_prob_contradiction` | NLI | Core claim-evidence contradiction probability |
| 3 | `nli_prob_neutral` | NLI | Core NLI neutral posterior |
| 4 | `nli_entailment_ratio` | NLI | Entailment − Contradiction delta |
| 5 | `semantic_cosine_sim` | Semantic | Dense embedding cosine similarity |
| 6 | `semantic_euclidean_dist` | Semantic | L2 distance between embeddings |
| 7 | `semantic_dot_product` | Semantic | Inner product (redundant with cosine) |
| 8 | `semantic_embedding_magnitude_ratio` | Semantic | Vector magnitude ratio |
| 9 | `entity_overlap_ratio` | Factual | NER overlap proportion |
| 10 | `missing_entity_count` | Factual | Count of ungrounded entities |
| 11 | `number_match_ratio` | Factual | Numerical consistency ratio |
| 12 | `missing_number_count` | Factual | Mismatched digit count |
| 13 | `retrieval_score` | Factual | Evidence retrieval similarity |
| 14 | `lexical_jaccard_similarity` | Linguistic | Token set Jaccard index |
| 15 | `token_precision` | Linguistic | Claim tokens found in evidence / total claim tokens |
| 16 | `token_recall` | Linguistic | Claim tokens found in evidence / total evidence tokens |
| 17 | `token_f1` | Linguistic | Harmonic mean of token precision & recall |
| 18 | `hedge_word_count` | Linguistic | Count of uncertainty cue words |

---

## 4. Dataset

- **Source:** HaluEval QA (`data/raw/halueval/qa_samples.jsonl`)
- **Total raw samples:** {split_meta['total']}
- **Valid samples after deduplication:** {split_meta['train_size'] + split_meta['test_size']}
- **Label distribution:** Balanced (approximately 50/50 hallucinated/factual)

---

## 5. Split

- **Method:** Stratified random split
- **Seed:** {RANDOM_SEED}
- **Test ratio:** {split_meta['test_ratio']}
- **Train size:** {split_meta['train_size']} (pos={split_meta['train_pos']}, neg={split_meta['train_neg']})
- **Test size:** {split_meta['test_size']} (pos={split_meta['test_pos']}, neg={split_meta['test_neg']})
- **Identical to V1 split:** Yes (same seed, same validation, same deduplication logic)

---

## 6. Training Configuration

| Parameter | Value |
|---|---|
| Classifier | `RandomForestClassifier` |
| n_estimators | 100 |
| random_state | 42 |
| Feature count | 18 (vs 20 in V1) |
| Scaler | `StandardScaler` fitted on training partition only |
| Leakage prevention | Scaler fit strictly on training data |
| Hyperparameter tuning on test set | None |

---

## 7. V1 Results (Baseline Reference)

| Metric | V1 Value |
|---|---|
| Accuracy | {v1_acc} |
| Precision | {v1_metrics.get('precision', 'N/A')} |
| Recall | {v1_metrics.get('recall', 'N/A')} |
| F1-Score | {v1_f1} |
| F1-Macro | {v1_metrics.get('f1_macro', 'N/A')} |
| ROC-AUC | {v1_metrics.get('roc_auc', 'N/A')} |
| PR-AUC | {v1_metrics.get('pr_auc', 'N/A')} |
| Training Time | {v1_metrics.get('training_time_sec', 'N/A')} s |
| Inference Time | {v1_metrics.get('inference_time_sec', 'N/A')} s |
| TP / TN / FP / FN | {v1_metrics.get('confusion_matrix', {}).get('tp', 'N/A')} / {v1_metrics.get('confusion_matrix', {}).get('tn', 'N/A')} / {v1_metrics.get('confusion_matrix', {}).get('fp', 'N/A')} / {v1_metrics.get('confusion_matrix', {}).get('fn', 'N/A')} |

---

## 8. V2 Results

| Metric | V2 Value |
|---|---|
| Accuracy | {v2_acc} |
| Precision | {v2_metrics.get('precision', 'N/A')} |
| Recall | {v2_metrics.get('recall', 'N/A')} |
| F1-Score | {v2_f1} |
| F1-Macro | {v2_metrics.get('f1_macro', 'N/A')} |
| ROC-AUC | {v2_metrics.get('roc_auc', 'N/A')} |
| PR-AUC | {v2_metrics.get('pr_auc', 'N/A')} |
| Training Time | {v2_metrics.get('training_time_sec', 'N/A')} s |
| Inference Time | {v2_metrics.get('inference_time_sec', 'N/A')} s |
| TP / TN / FP / FN | {v2_metrics.get('confusion_matrix', {}).get('tp', 'N/A')} / {v2_metrics.get('confusion_matrix', {}).get('tn', 'N/A')} / {v2_metrics.get('confusion_matrix', {}).get('fp', 'N/A')} / {v2_metrics.get('confusion_matrix', {}).get('fn', 'N/A')} |

---

## 9. Feature Importance (V2)

| Rank | Feature | V2 Importance | V1 Importance |
|---|---|---|---|
{all_feat_lines}

### Top 5 V2 Features

| Rank | Feature | Importance |
|---|---|---|
{top_feat_lines}

---

## 10. Error Analysis

### Summary

| Category | Count |
|---|---|
| Total Errors | {total_errors} |
| False Positives (Factual flagged as Hallucinated) | {fp_count} |
| False Negatives (Hallucinated missed) | {fn_count} |

### Error Categories

| Error Type | Count |
|---|---|
| Short factual answers incorrectly flagged | {short_flagged} |
| NLI failure (entailment high but flagged, or contradiction low but missed) | {error_analysis['error_categories'].get('nli_failure', {}).get('count', 0)} |
| Semantic similarity failure | {error_analysis['error_categories'].get('semantic_similarity_failure', {}).get('count', 0)} |
| Entity mismatch failure | {error_analysis['error_categories'].get('entity_mismatch_failure', {}).get('count', 0)} |
| Numerical mismatch failure | {error_analysis['error_categories'].get('numerical_mismatch_failure', {}).get('count', 0)} |
| Insufficient evidence indication | {error_analysis['error_categories'].get('insufficient_evidence_indication', {}).get('count', 0)} |
| Ambiguous cases (probability near 0.50) | {error_analysis['error_categories'].get('ambiguous_cases', {}).get('count', 0)} |

---

## 11. V1 vs V2 Comparison

| Metric | V1 | V2 | Delta |
|---|---|---|---|
| Accuracy | {v1_acc} | {v2_acc} | {comparison['metrics_comparison'].get('accuracy', {}).get('delta', 'N/A')} |
| Precision | {v1_metrics.get('precision', 'N/A')} | {v2_metrics.get('precision', 'N/A')} | {comparison['metrics_comparison'].get('precision', {}).get('delta', 'N/A')} |
| Recall | {v1_metrics.get('recall', 'N/A')} | {v2_metrics.get('recall', 'N/A')} | {comparison['metrics_comparison'].get('recall', {}).get('delta', 'N/A')} |
| F1-Score | {v1_f1} | {v2_f1} | {comparison['metrics_comparison'].get('f1_score', {}).get('delta', 'N/A')} |
| F1-Macro | {v1_metrics.get('f1_macro', 'N/A')} | {v2_metrics.get('f1_macro', 'N/A')} | {comparison['metrics_comparison'].get('f1_macro', {}).get('delta', 'N/A')} |
| ROC-AUC | {v1_metrics.get('roc_auc', 'N/A')} | {v2_metrics.get('roc_auc', 'N/A')} | {comparison['metrics_comparison'].get('roc_auc', {}).get('delta', 'N/A')} |
| PR-AUC | {v1_metrics.get('pr_auc', 'N/A')} | {v2_metrics.get('pr_auc', 'N/A')} | {comparison['metrics_comparison'].get('pr_auc', {}).get('delta', 'N/A')} |
| Training Time (s) | {v1_metrics.get('training_time_sec', 'N/A')} | {v2_metrics.get('training_time_sec', 'N/A')} | {comparison['metrics_comparison'].get('training_time_sec', {}).get('delta', 'N/A')} |
| Inference Time (s) | {v1_metrics.get('inference_time_sec', 'N/A')} | {v2_metrics.get('inference_time_sec', 'N/A')} | {comparison['metrics_comparison'].get('inference_time_sec', {}).get('delta', 'N/A')} |

### Prediction Differences

| Category | Count |
|---|---|
| Total predictions that changed | {total_changes} |
| V1 correct → V2 wrong (regressions) | {v1_correct_v2_wrong} |
| V1 wrong → V2 correct (improvements) | {v1_wrong_v2_correct} |

---

## 12. Interpretation

{most_important_finding}

The removal of `claim_word_count` (V1 rank #1, importance 28.57%) and `length_ratio` (V1 rank #3, importance 13.49%) forced the Random Forest to redistribute its decision weight across the remaining 18 features.

The V2 feature importance ranking reveals which signals the model relies on when length shortcuts are unavailable.

---

## 13. Limitations

1. **Single dataset:** Both V1 and V2 are evaluated on HaluEval QA only. Cross-dataset generalization is unknown.
2. **Heuristic NLI:** The NLI features use a lexical heuristic fallback when PyTorch/Transformers models are unavailable. True NLI model inference would provide stronger signals.
3. **Hash-based semantic embeddings:** Without SentenceTransformers installed, semantic features use hash-based pseudo-embeddings rather than dense neural embeddings.
4. **Fixed retrieval score:** All instances use a constant `retrieval_score=0.90`, making this feature uninformative in the current setup.
5. **No hyperparameter tuning:** RandomForest uses default `n_estimators=100` without grid search.
6. **Token precision monitoring:** `token_precision` should be monitored — if it becomes the dominant V2 feature, it may also be exploiting an indirect length correlation.

---

## 14. Recommended Next Technical Step

Fine-tune BERT (`bert-base-uncased`) and DeBERTa (`deberta-v3-base`) on the same HaluEval QA split using true PyTorch training (Experiments 3–6) to enable a fair cross-architecture comparison. The Hybrid V2 result provides the cleaned multi-feature baseline; Transformer experiments will show whether deep contextual representations can match or exceed the hybrid approach without relying on engineered features.

---

## Final Summary

```
HYBRID V1 F1:
{v1_f1}

HYBRID V2 F1:
{v2_f1}

V2 STATUS:
{v2_status}

MOST IMPORTANT FINDING:
{most_important_finding}

NEXT TECHNICAL STEP:
Fine-tune BERT and DeBERTa on HaluEval QA (Experiments 3-6) for cross-architecture comparison.
```
"""

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"Research report saved to {REPORT_PATH}")


# ============================================================
# MAIN
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("HYBRID V2 EXPERIMENT: Remove Length Shortcuts")
    logger.info(f"Excluded features: {EXCLUDED_FEATURES}")
    logger.info("=" * 60)

    np.random.seed(RANDOM_SEED)

    # Step 1: Load & validate
    logger.info("Step 1: Loading raw HaluEval data...")
    records = load_raw_data(RAW_DATA_PATH)

    logger.info("Step 2: Validating data...")
    validation_report, valid_records = validate_data(records)

    # Step 3: Split (identical to V1)
    logger.info("Step 3: Creating stratified split (identical to V1)...")
    train_data, test_data, split_meta = create_stratified_split(valid_records)

    # Verify split matches V1
    v1_config_path = V1_RESULTS_DIR / "config.json"
    if v1_config_path.exists():
        with open(v1_config_path, "r") as f:
            v1_config = json.load(f)
        assert split_meta["train_size"] == v1_config["train_size"], \
            f"Train size mismatch: V2={split_meta['train_size']} vs V1={v1_config['train_size']}"
        assert split_meta["test_size"] == v1_config["test_size"], \
            f"Test size mismatch: V2={split_meta['test_size']} vs V1={v1_config['test_size']}"
        logger.info(f"[OK] Split matches V1: train={split_meta['train_size']}, test={split_meta['test_size']}")

    # Step 4: Train Hybrid V2
    logger.info("Step 4: Training Hybrid V2...")
    model, v2_metrics, v2_importances, test_labels, test_preds, test_probs = \
        train_hybrid_v2(train_data, test_data)

    # Verify feature count
    assert len(model.vectorizer.feature_names) == 18, \
        f"Expected 18 features, got {len(model.vectorizer.feature_names)}"
    assert "claim_word_count" not in model.vectorizer.feature_names, \
        "claim_word_count should be excluded"
    assert "length_ratio" not in model.vectorizer.feature_names, \
        "length_ratio should be excluded"
    logger.info(f"[OK] Feature vector: {len(model.vectorizer.feature_names)} features (excluded: {EXCLUDED_FEATURES})")

    # Step 5: Save all artifacts
    logger.info("Step 5: Saving experiment artifacts...")
    save_experiment_artifacts(model, v2_metrics, v2_importances, split_meta,
                              test_data, test_labels, test_preds, test_probs)

    # Step 6: V1 vs V2 comparison
    logger.info("Step 6: Generating V1 vs V2 comparison...")
    comparison, v1_metrics, v1_importances = generate_v1_v2_comparison(v2_metrics, v2_importances)

    # Step 7: Prediction diff analysis
    logger.info("Step 7: Generating prediction diff analysis...")
    diff_report = generate_prediction_diff(test_data, test_labels, test_preds, test_probs)

    # Step 8: Error analysis
    logger.info("Step 8: Generating categorized error analysis...")
    # Use a fresh vectorizer for error analysis feature extraction (full features, not excluded)
    analysis_vectorizer = FeatureVectorizer()
    error_analysis = generate_error_analysis(test_data, test_labels, test_preds, test_probs, analysis_vectorizer)

    # Step 9: Generate research report
    logger.info("Step 9: Generating research report...")
    generate_report(v2_metrics, v1_metrics, v2_importances, v1_importances,
                    comparison, diff_report, error_analysis, split_meta)

    # Final summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("HYBRID V2 EXPERIMENT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"V1 F1: {v1_metrics.get('f1_score', 'N/A')}")
    logger.info(f"V2 F1: {v2_metrics.get('f1_score', 'N/A')}")
    logger.info(f"Features: 18 (excluded: {EXCLUDED_FEATURES})")
    logger.info(f"Results: {RESULTS_DIR}")
    logger.info(f"Model:   {CHECKPOINT_DIR / 'model.pkl'}")
    logger.info(f"Report:  {REPORT_PATH}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
