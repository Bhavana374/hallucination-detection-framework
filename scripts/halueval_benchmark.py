"""
HaluEval Benchmark Evaluation Script.

Runs 7 experiments on the HaluEval QA dataset with full reproducibility,
data leakage prevention, and comprehensive artifact generation.

Usage:
    # Sanity check (200 samples):
    python scripts/halueval_benchmark.py --sanity-check

    # Full evaluation (10,000 samples):
    python scripts/halueval_benchmark.py
"""

import argparse
import json
import os
import sys
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger

logger = get_logger("halueval_benchmark")

RANDOM_SEED = 42
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "halueval" / "qa_samples.jsonl"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "halueval"
RESULTS_DIR = PROJECT_ROOT / "results" / "halueval"


# ============================================================
# DATA LOADING & VALIDATION
# ============================================================

def load_raw_data(filepath: Path) -> List[Dict[str, str]]:
    """Load raw HaluEval QA JSONL data."""
    if not filepath.exists():
        raise FileNotFoundError(f"Raw data not found at {filepath}. Run download first.")

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


def validate_data(records: List[Dict[str, str]]) -> Dict[str, Any]:
    """Validate dataset integrity and generate validation report."""
    report = {
        "total_samples": len(records),
        "missing_fields": {"knowledge": 0, "question": 0, "answer": 0, "hallucination": 0},
        "empty_fields": {"knowledge": 0, "question": 0, "answer": 0, "hallucination": 0},
        "invalid_labels": 0,
        "invalid_label_values": [],
        "exact_duplicates": 0,
        "duplicate_indices": [],
        "class_distribution": {"yes": 0, "no": 0},
        "valid_samples": 0,
        "removed_samples": 0,
        "removal_reasons": [],
    }

    valid_labels = {"yes", "no"}
    seen_hashes = {}
    valid_records = []

    for i, rec in enumerate(records):
        issues = []

        # Check missing fields
        for field in ["knowledge", "question", "answer", "hallucination"]:
            if field not in rec:
                report["missing_fields"][field] += 1
                issues.append(f"missing_{field}")
            elif not rec[field] or not str(rec[field]).strip():
                report["empty_fields"][field] += 1
                issues.append(f"empty_{field}")

        # Check label validity
        label = rec.get("hallucination", "")
        if label not in valid_labels:
            report["invalid_labels"] += 1
            report["invalid_label_values"].append({"index": i, "value": label})
            issues.append(f"invalid_label:{label}")

        # Check exact duplicates
        row_hash = hashlib.md5(
            (rec.get("answer", "") + rec.get("knowledge", "")).encode()
        ).hexdigest()
        if row_hash in seen_hashes:
            report["exact_duplicates"] += 1
            report["duplicate_indices"].append({"index": i, "duplicate_of": seen_hashes[row_hash]})
            issues.append("exact_duplicate")
        else:
            seen_hashes[row_hash] = i

        if issues:
            report["removed_samples"] += 1
            report["removal_reasons"].append({"index": i, "reasons": issues})
        else:
            report["class_distribution"][label] += 1
            valid_records.append(rec)

    report["valid_samples"] = len(valid_records)
    logger.info(f"Validation: {report['valid_samples']} valid / {report['removed_samples']} removed / {report['exact_duplicates']} duplicates")
    return report, valid_records


def create_stratified_split(
    records: List[Dict[str, str]],
    test_ratio: float = 0.20,
    seed: int = RANDOM_SEED
) -> Tuple[List[Dict], List[Dict], Dict[str, Any]]:
    """Create stratified train/test split with leakage prevention."""
    rng = np.random.RandomState(seed)

    # Separate by label
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

    # Shuffle within splits
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
        "train_indices": [i for i, _ in train_pos + train_neg],
        "test_indices": [i for i, _ in test_pos + test_neg],
    }

    logger.info(f"Split: train={len(train_data)} (pos={len(train_pos)}, neg={len(train_neg)}), "
                f"test={len(test_data)} (pos={len(test_pos)}, neg={len(test_neg)})")
    return train_data, test_data, split_meta


# ============================================================
# EXPERIMENT RUNNERS
# ============================================================

def run_exp1_tfidf_lr(train_data, test_data, output_dir: Path) -> Dict:
    """Experiment 1: TF-IDF + Logistic Regression."""
    from src.models.baselines.tfidf_logistic_regression import TfidfLogisticRegressionModel
    from src.evaluation.metrics import compute_classification_metrics

    logger.info("=== EXP 1: TF-IDF + Logistic Regression ===")

    train_texts = [r["answer"] for r in train_data]
    train_labels = [1 if r["hallucination"] == "yes" else 0 for r in train_data]
    test_texts = [r["answer"] for r in test_data]
    test_labels = [1 if r["hallucination"] == "yes" else 0 for r in test_data]

    model = TfidfLogisticRegressionModel(max_features=5000, ngram_range=(1, 2), c_param=1.0, random_state=RANDOM_SEED)

    t0 = time.time()
    model.fit(train_texts, train_labels)
    train_time = time.time() - t0

    t0 = time.time()
    test_probs = model.predict_proba(test_texts)
    test_preds = [1 if p >= 0.5 else 0 for p in test_probs]
    infer_time = time.time() - t0

    metrics = compute_classification_metrics(test_labels, test_preds, test_probs, train_time, infer_time)
    _save_experiment(output_dir, "exp01_tfidf_lr", {
        "model": "TF-IDF + Logistic Regression",
        "input_type": "claim_only",
        "max_features": 5000, "ngram_range": [1, 2], "C": 1.0,
        "seed": RANDOM_SEED, "train_size": len(train_data), "test_size": len(test_data),
    }, metrics, test_data, test_labels, test_preds, test_probs)
    return metrics


def run_exp2_tfidf_rf(train_data, test_data, output_dir: Path) -> Dict:
    """Experiment 2: TF-IDF + Random Forest."""
    from src.models.baselines.random_forest_baseline import TfidfRandomForestBaseline
    from src.evaluation.metrics import compute_classification_metrics

    logger.info("=== EXP 2: TF-IDF + Random Forest ===")

    train_texts = [r["answer"] for r in train_data]
    train_labels = [1 if r["hallucination"] == "yes" else 0 for r in train_data]
    test_texts = [r["answer"] for r in test_data]
    test_labels = [1 if r["hallucination"] == "yes" else 0 for r in test_data]

    model = TfidfRandomForestBaseline(max_features=5000, n_estimators=100, random_state=RANDOM_SEED)

    t0 = time.time()
    model.fit(train_texts, train_labels)
    train_time = time.time() - t0

    t0 = time.time()
    test_probs = model.predict_proba(test_texts)
    test_preds = [1 if p >= 0.5 else 0 for p in test_probs]
    infer_time = time.time() - t0

    metrics = compute_classification_metrics(test_labels, test_preds, test_probs, train_time, infer_time)
    _save_experiment(output_dir, "exp02_tfidf_rf", {
        "model": "TF-IDF + Random Forest",
        "input_type": "claim_only",
        "max_features": 5000, "n_estimators": 100,
        "seed": RANDOM_SEED, "train_size": len(train_data), "test_size": len(test_data),
    }, metrics, test_data, test_labels, test_preds, test_probs)
    return metrics


def _finetune_transformer(model_obj, train_data, test_data, output_dir, exp_id, exp_name,
                          use_evidence=False, epochs=3, batch_size=16, lr=2e-5):
    """Shared fine-tuning logic for BERT/DeBERTa experiments."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from src.evaluation.metrics import compute_classification_metrics

    logger.info(f"=== {exp_id.upper()}: {exp_name} ===")

    tokenizer = model_obj.tokenizer
    model = model_obj.model
    device = model_obj.device
    max_length = model_obj.max_length

    train_labels = [1 if r["hallucination"] == "yes" else 0 for r in train_data]
    test_labels = [1 if r["hallucination"] == "yes" else 0 for r in test_data]

    # Tokenize
    if use_evidence:
        train_enc = tokenizer(
            [r["answer"] for r in train_data],
            [r["knowledge"] for r in train_data],
            padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        test_enc = tokenizer(
            [r["answer"] for r in test_data],
            [r["knowledge"] for r in test_data],
            padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
    else:
        train_enc = tokenizer(
            [r["answer"] for r in train_data],
            padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        test_enc = tokenizer(
            [r["answer"] for r in test_data],
            padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )

    train_dataset = TensorDataset(
        train_enc["input_ids"], train_enc["attention_mask"],
        torch.tensor(train_labels, dtype=torch.long)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Fine-tune
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()

    training_history = []
    t0 = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        for batch_idx, batch in enumerate(train_loader):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * input_ids.size(0)
            preds = torch.argmax(outputs.logits, dim=-1)
            epoch_correct += (preds == labels).sum().item()
            epoch_total += input_ids.size(0)

            if (batch_idx + 1) % 50 == 0:
                logger.info(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, "
                            f"Loss: {loss.item():.4f}")

        epoch_acc = epoch_correct / epoch_total if epoch_total > 0 else 0
        epoch_avg_loss = epoch_loss / epoch_total if epoch_total > 0 else 0
        training_history.append({"epoch": epoch + 1, "loss": epoch_avg_loss, "accuracy": epoch_acc})
        logger.info(f"  Epoch {epoch+1}: Loss={epoch_avg_loss:.4f}, Acc={epoch_acc:.4f}")

    train_time = time.time() - t0

    # Evaluate on test set
    model.eval()
    all_probs = []
    t0 = time.time()

    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            end = min(i + batch_size, len(test_data))
            batch_ids = test_enc["input_ids"][i:end].to(device)
            batch_mask = test_enc["attention_mask"][i:end].to(device)
            outputs = model(input_ids=batch_ids, attention_mask=batch_mask)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            all_probs.extend(probs[:, 1].tolist())

    infer_time = time.time() - t0

    test_preds = [1 if p >= 0.5 else 0 for p in all_probs]
    metrics = compute_classification_metrics(test_labels, test_preds, all_probs, train_time, infer_time)

    config = {
        "model": exp_name,
        "model_name": model_obj.model_name,
        "input_type": "claim+evidence" if use_evidence else "claim_only",
        "epochs": epochs, "batch_size": batch_size, "learning_rate": lr,
        "max_length": max_length, "device": device,
        "seed": RANDOM_SEED, "train_size": len(train_data), "test_size": len(test_data),
        "training_history": training_history,
    }

    _save_experiment(output_dir, exp_id, config, metrics, test_data, test_labels, test_preds, all_probs)

    # Save training history
    history_path = output_dir / exp_id / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)

    return metrics


def run_exp3_bert(train_data, test_data, output_dir: Path, epochs=3, batch_size=16) -> Dict:
    """Experiment 3: Standalone BERT fine-tuning."""
    from src.models.bert.bert_classifier import BERTClassifier
    model = BERTClassifier(model_name="bert-base-uncased", num_labels=2, max_length=256)
    return _finetune_transformer(model, train_data, test_data, output_dir,
                                  "exp03_bert", "Standalone BERT", use_evidence=False,
                                  epochs=epochs, batch_size=batch_size)


def run_exp4_deberta(train_data, test_data, output_dir: Path, epochs=3, batch_size=16) -> Dict:
    """Experiment 4: Standalone DeBERTa fine-tuning."""
    from src.models.deberta.deberta_classifier import DeBERTaClassifier
    model = DeBERTaClassifier(model_name="microsoft/deberta-v3-base", num_labels=2, max_length=256)
    return _finetune_transformer(model, train_data, test_data, output_dir,
                                  "exp04_deberta", "Standalone DeBERTa", use_evidence=False,
                                  epochs=epochs, batch_size=batch_size)


def run_exp5_evidence_bert(train_data, test_data, output_dir: Path, epochs=3, batch_size=16) -> Dict:
    """Experiment 5: Evidence-Grounded BERT fine-tuning."""
    from src.models.bert.bert_classifier import BERTClassifier
    model = BERTClassifier(model_name="bert-base-uncased", num_labels=2, max_length=256)
    return _finetune_transformer(model, train_data, test_data, output_dir,
                                  "exp05_evidence_bert", "Evidence-Grounded BERT", use_evidence=True,
                                  epochs=epochs, batch_size=batch_size)


def run_exp6_evidence_deberta(train_data, test_data, output_dir: Path, epochs=3, batch_size=16) -> Dict:
    """Experiment 6: Evidence-Grounded DeBERTa fine-tuning."""
    from src.models.deberta.deberta_classifier import DeBERTaClassifier
    model = DeBERTaClassifier(model_name="microsoft/deberta-v3-base", num_labels=2, max_length=256)
    return _finetune_transformer(model, train_data, test_data, output_dir,
                                  "exp06_evidence_deberta", "Evidence-Grounded DeBERTa", use_evidence=True,
                                  epochs=epochs, batch_size=batch_size)


def run_exp7_hybrid_rf(train_data, test_data, output_dir: Path) -> Dict:
    """Experiment 7: Hybrid Random Forest over 20-dimensional features."""
    from src.features.feature_vectorizer import FeatureVectorizer
    from src.models.hybrid.hybrid_classifier import HybridClassifier
    from src.evaluation.metrics import compute_classification_metrics

    logger.info("=== EXP 7: Hybrid Random Forest ===")

    train_labels = [1 if r["hallucination"] == "yes" else 0 for r in train_data]
    test_labels = [1 if r["hallucination"] == "yes" else 0 for r in test_data]

    # Build instance dicts
    train_instances = [
        {"claim": r["answer"], "evidence": r["knowledge"], "retrieval_score": 0.90}
        for r in train_data
    ]
    test_instances = [
        {"claim": r["answer"], "evidence": r["knowledge"], "retrieval_score": 0.90}
        for r in test_data
    ]

    vectorizer = FeatureVectorizer()
    model = HybridClassifier(classifier_type="random_forest", n_estimators=100,
                              random_state=RANDOM_SEED, vectorizer=vectorizer)

    t0 = time.time()
    model.fit(train_instances, train_labels)
    train_time = time.time() - t0

    t0 = time.time()
    test_probs = model.predict_hallucination_score(test_instances)
    test_preds = [1 if p >= 0.5 else 0 for p in test_probs]
    infer_time = time.time() - t0

    metrics = compute_classification_metrics(test_labels, test_preds, test_probs, train_time, infer_time)

    # Get feature importances
    feat_importances = model.get_feature_importances()

    config = {
        "model": "Hybrid Random Forest",
        "input_type": "claim+evidence+features",
        "n_estimators": 100, "n_features": 20,
        "feature_names": model.vectorizer.feature_names,
        "feature_importances": feat_importances,
        "seed": RANDOM_SEED, "train_size": len(train_data), "test_size": len(test_data),
    }

    _save_experiment(output_dir, "exp07_hybrid_rf", config, metrics, test_data, test_labels, test_preds, test_probs)
    return metrics


def run_exp8_hybrid_v2(train_data, test_data, output_dir: Path) -> Dict:
    """Experiment 8: Hybrid Random Forest V2 without length shortcuts."""
    from src.features.feature_vectorizer import FeatureVectorizer
    from src.models.hybrid.hybrid_classifier import HybridClassifier
    from src.evaluation.metrics import compute_classification_metrics

    logger.info("=== EXP 8: Hybrid Random Forest V2 ===")

    train_labels = [1 if r["hallucination"] == "yes" else 0 for r in train_data]
    test_labels = [1 if r["hallucination"] == "yes" else 0 for r in test_data]

    # Build instance dicts
    train_instances = [
        {"claim": r["answer"], "evidence": r["knowledge"], "retrieval_score": 0.90}
        for r in train_data
    ]
    test_instances = [
        {"claim": r["answer"], "evidence": r["knowledge"], "retrieval_score": 0.90}
        for r in test_data
    ]

    vectorizer = FeatureVectorizer(excluded_features=["claim_word_count", "length_ratio"])
    model = HybridClassifier(classifier_type="random_forest", n_estimators=100,
                              random_state=RANDOM_SEED, vectorizer=vectorizer)

    t0 = time.time()
    model.fit(train_instances, train_labels)
    train_time = time.time() - t0

    t0 = time.time()
    test_probs = model.predict_hallucination_score(test_instances)
    test_preds = [1 if p >= 0.5 else 0 for p in test_probs]
    infer_time = time.time() - t0

    metrics = compute_classification_metrics(test_labels, test_preds, test_probs, train_time, infer_time)

    # Save model checkpoint
    model_checkpoint_path = PROJECT_ROOT / "models" / "checkpoints" / "hybrid_v2" / "model.pkl"
    model.save(model_checkpoint_path)

    # Get feature importances
    feat_importances = model.get_feature_importances()

    config = {
        "model": "Hybrid Random Forest V2",
        "input_type": "claim+evidence+features",
        "n_estimators": 100,
        "n_features": len(model.vectorizer.feature_names),
        "excluded_features": model.vectorizer.excluded_features,
        "feature_names": model.vectorizer.feature_names,
        "feature_importances": feat_importances,
        "seed": RANDOM_SEED,
        "train_size": len(train_data),
        "test_size": len(test_data),
    }

    _save_experiment(output_dir, "exp08_hybrid_v2", config, metrics, test_data, test_labels, test_preds, test_probs)
    return metrics


# ============================================================
# ARTIFACT SAVING
# ============================================================

def _save_experiment(output_dir: Path, exp_id: str, config: Dict, metrics: Dict,
                     test_data: List[Dict], test_labels: List[int],
                     test_preds: List[int], test_probs: List[float]):
    """Save all experiment artifacts."""
    exp_dir = output_dir / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # config.json
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    # metrics.json
    metrics_payload = {
        "experiment_id": exp_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_metrics": metrics,
    }
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    # predictions.csv
    with open(exp_dir / "predictions.csv", "w", encoding="utf-8") as f:
        f.write("index,answer_text,true_label,predicted_label,probability,is_correct\n")
        for i, (rec, yt, yp, prob) in enumerate(zip(test_data, test_labels, test_preds, test_probs)):
            ans = rec["answer"].replace('"', '""')
            f.write(f'{i},"{ans}",{yt},{yp},{prob:.6f},{int(yt == yp)}\n')

    # classification_report.txt
    lines = [
        f"{'='*60}",
        f"EXPERIMENT: {exp_id}",
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
    with open(exp_dir / "classification_report.txt", "w") as f:
        f.write("\n".join(lines))

    # Plot confusion matrix if matplotlib and seaborn are available
    try:
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
        plt.title(f"Confusion Matrix - {exp_id}")
        plt.savefig(exp_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logger.warning(f"Could not generate confusion matrix plot: {e}")

    # Save feature importances as CSV if present in config
    if "feature_importances" in config:
        try:
            with open(exp_dir / "feature_importance.csv", "w", encoding="utf-8") as f:
                f.write("feature,importance\n")
                for feat, imp in config["feature_importances"].items():
                    f.write(f"{feat},{imp:.6f}\n")
        except Exception as e:
            logger.warning(f"Could not save feature importance CSV: {e}")

    # run_metadata.json
    import torch
    meta = {
        "experiment_id": exp_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "random_seed": RANDOM_SEED,
        "python_version": sys.version,
    }
    with open(exp_dir / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Saved experiment artifacts to {exp_dir}")


def generate_comparison(results: Dict[str, Dict], output_dir: Path):
    """Generate model comparison CSV and JSON."""
    rows = []
    for exp_id, metrics in results.items():
        rows.append({
            "experiment_id": exp_id,
            "accuracy": metrics.get("accuracy", "N/A"),
            "precision": metrics.get("precision", "N/A"),
            "recall": metrics.get("recall", "N/A"),
            "f1_score": metrics.get("f1_score", "N/A"),
            "f1_macro": metrics.get("f1_macro", "N/A"),
            "roc_auc": metrics.get("roc_auc", "N/A"),
            "pr_auc": metrics.get("pr_auc", "N/A"),
            "training_time_sec": metrics.get("training_time_sec", "N/A"),
            "inference_time_sec": metrics.get("inference_time_sec", "N/A"),
            "latency_per_sample_ms": metrics.get("latency_per_sample_ms", "N/A"),
        })

    # CSV
    csv_path = output_dir / "model_comparison.csv"
    with open(csv_path, "w") as f:
        headers = list(rows[0].keys())
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(str(row[h]) for h in headers) + "\n")

    # JSON
    json_path = output_dir / "model_comparison.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    logger.info(f"Saved model comparison to {csv_path} and {json_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="HaluEval Benchmark Evaluation")
    parser.add_argument("--sanity-check", action="store_true",
                        help="Run sanity check on 200 samples only")
    parser.add_argument("--experiments", nargs="*", default=None,
                        help="Specific experiments to run (e.g., exp01 exp02 exp07 exp08)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of fine-tuning epochs for Transformer experiments")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for Transformer experiments")
    args = parser.parse_args()

    np.random.seed(RANDOM_SEED)

    # ---- Load & Validate ----
    logger.info("Loading raw HaluEval data...")
    records = load_raw_data(RAW_DATA_PATH)

    logger.info("Validating data...")
    validation_report, valid_records = validate_data(records)

    # Save validation report
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_DIR / "validation_report.json", "w") as f:
        json.dump(validation_report, f, indent=2)
    logger.info(f"Validation report saved to {PROCESSED_DIR / 'validation_report.json'}")

    # ---- Subset for sanity check ----
    if args.sanity_check:
        logger.info("SANITY CHECK MODE: Using 200 samples only")
        rng = np.random.RandomState(RANDOM_SEED)
        indices = rng.choice(len(valid_records), size=min(200, len(valid_records)), replace=False)
        valid_records = [valid_records[i] for i in indices]
        run_label = "sanity_check"
    else:
        run_label = "full"

    # ---- Split ----
    logger.info("Creating stratified train/test split...")
    train_data, test_data, split_meta = create_stratified_split(valid_records)

    split_meta["run_type"] = run_label
    with open(PROCESSED_DIR / "split_indices.json", "w") as f:
        # Don't save full indices for large datasets — just metadata
        meta_save = {k: v for k, v in split_meta.items() if k not in ("train_indices", "test_indices")}
        json.dump(meta_save, f, indent=2)

    # ---- Determine output directory ----
    output_dir = RESULTS_DIR / run_label
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Run Experiments ----
    all_experiments = {
        "exp01": ("exp01_tfidf_lr", lambda: run_exp1_tfidf_lr(train_data, test_data, output_dir)),
        "exp02": ("exp02_tfidf_rf", lambda: run_exp2_tfidf_rf(train_data, test_data, output_dir)),
        "exp03": ("exp03_bert", lambda: run_exp3_bert(train_data, test_data, output_dir, args.epochs, args.batch_size)),
        "exp04": ("exp04_deberta", lambda: run_exp4_deberta(train_data, test_data, output_dir, args.epochs, args.batch_size)),
        "exp05": ("exp05_evidence_bert", lambda: run_exp5_evidence_bert(train_data, test_data, output_dir, args.epochs, args.batch_size)),
        "exp06": ("exp06_evidence_deberta", lambda: run_exp6_evidence_deberta(train_data, test_data, output_dir, args.epochs, args.batch_size)),
        "exp07": ("exp07_hybrid_rf", lambda: run_exp7_hybrid_rf(train_data, test_data, output_dir)),
        "exp08": ("exp08_hybrid_v2", lambda: run_exp8_hybrid_v2(train_data, test_data, output_dir)),
    }

    # Filter experiments if specified
    if args.experiments:
        selected = {k: v for k, v in all_experiments.items() if k in args.experiments}
    else:
        selected = all_experiments

    results = {}
    for key, (exp_name, runner) in selected.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"RUNNING: {exp_name}")
        logger.info(f"{'='*60}")
        try:
            t0 = time.time()
            metrics = runner()
            elapsed = time.time() - t0
            results[exp_name] = metrics
            logger.info(f"COMPLETED {exp_name} in {elapsed:.1f}s — "
                        f"Acc={metrics.get('accuracy', 'N/A')}, F1={metrics.get('f1_score', 'N/A')}, "
                        f"ROC-AUC={metrics.get('roc_auc', 'N/A')}")
        except Exception as e:
            logger.error(f"FAILED {exp_name}: {e}", exc_info=True)
            results[exp_name] = {"error": str(e)}

    # ---- Comparison ----
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    if valid_results:
        generate_comparison(valid_results, output_dir)

    # ---- Summary ----
    logger.info(f"\n{'='*60}")
    logger.info(f"BENCHMARK COMPLETE ({run_label})")
    logger.info(f"{'='*60}")
    for exp_name, metrics in results.items():
        if "error" in metrics:
            logger.info(f"  {exp_name}: FAILED — {metrics['error']}")
        else:
            logger.info(f"  {exp_name}: Acc={metrics.get('accuracy')}, F1={metrics.get('f1_score')}, ROC-AUC={metrics.get('roc_auc')}")


if __name__ == "__main__":
    main()
