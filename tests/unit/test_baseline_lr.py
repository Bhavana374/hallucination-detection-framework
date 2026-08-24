"""Unit tests for Experiment 1: TF-IDF + Logistic Regression baseline model and evaluation metrics."""

from pathlib import Path
from src.evaluation.metrics import (
    compute_confusion_matrix,
    compute_classification_metrics,
    save_experiment_artifacts,
)
from src.models.baselines.tfidf_logistic_regression import TfidfLogisticRegressionModel


def test_confusion_matrix_and_metrics():
    """Verify classification metric calculations."""
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    # TP: 1, TN: 1, FP: 1, FN: 1
    cm = compute_confusion_matrix(y_true, y_pred)
    assert cm["tp"] == 1
    assert cm["tn"] == 1
    assert cm["fp"] == 1
    assert cm["fn"] == 1

    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1_score"] == 0.5


def test_tfidf_logistic_regression_fit_predict_save_load(tmp_path: Path):
    """Verify TF-IDF + Logistic Regression fit, predict, and checkpoint round-trip."""
    train_texts = [
        "Einstein was a brilliant theoretical physicist.",
        "Einstein developed relativity in physics.",
        "Nero was a ruthless Roman Emperor.",
        "Julius Caesar conquered Gaul in Rome.",
    ]
    train_labels = [0, 0, 1, 1]

    model = TfidfLogisticRegressionModel(max_features=100, ngram_range=(1, 2))
    model.fit(train_texts, train_labels)

    assert model.is_fitted is True

    test_texts = [
        "Einstein worked on physics.",
        "Roman emperor ruled in Rome.",
    ]
    preds = model.predict(test_texts)
    probs = model.predict_proba(test_texts)

    assert len(preds) == 2
    assert len(probs) == 2
    assert all(0.0 <= p <= 1.0 for p in probs)

    # Test serialization
    ckpt_path = tmp_path / "model_test.pkl"
    model.save_model(ckpt_path)
    assert ckpt_path.exists()

    loaded_model = TfidfLogisticRegressionModel.load_model(ckpt_path)
    assert loaded_model.is_fitted is True

    loaded_preds = loaded_model.predict(test_texts)
    assert loaded_preds == preds


def test_save_experiment_artifacts(tmp_path: Path):
    """Verify saving experiment metrics, config, predictions, and report."""
    val_metrics = {"accuracy": 0.8, "f1_score": 0.75}
    test_metrics = {
        "accuracy": 0.85,
        "precision": 0.8,
        "recall": 0.9,
        "f1_score": 0.85,
        "f1_macro": 0.85,
        "training_time_sec": 0.05,
        "latency_per_sample_ms": 1.2,
        "confusion_matrix": {"tp": 3, "tn": 3, "fp": 0, "fn": 0},
    }
    preds = [
        {"claim_id": "c1", "true_label": 0, "predicted_label": 0, "probability_hallucinated": 0.1},
        {"claim_id": "c2", "true_label": 1, "predicted_label": 1, "probability_hallucinated": 0.9},
    ]

    artifacts = save_experiment_artifacts(
        experiment_id="exp1_test",
        output_dir=tmp_path / "exp1",
        config_dict={"param": "value"},
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        test_predictions=preds,
    )

    assert Path(artifacts["config"]).exists()
    assert Path(artifacts["metrics"]).exists()
    assert Path(artifacts["predictions"]).exists()
    assert Path(artifacts["classification_report"]).exists()
