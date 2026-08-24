"""
Unified Model Trainer for Baselines, PyTorch Transformers, and Hybrid Meta-Classifiers.

Handles training loops, validation evaluation, early stopping, deterministic seeding,
and metric logging across the experiment matrix.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import numpy as np
from src.evaluation.metrics import evaluate_predictions
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger("trainer")


class EarlyStopping:
    """Early stopping handler based on validation loss or score monitoring."""

    def __init__(self, patience: int = 3, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.early_stop = False

    def check(self, score: float) -> bool:
        if self.mode == "max":
            improved = score > (self.best_score + self.min_delta)
        else:
            improved = score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class ModelTrainer:
    """
    Unified trainer manager for training and evaluating models in the framework.
    """

    def __init__(
        self,
        seed: int = 42,
        output_dir: str = "models/checkpoints",
        patience: int = 3
    ):
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.patience = patience
        set_seed(seed)

    def train_baseline_or_hybrid(
        self,
        model: Any,
        train_data: Union[List[str], List[Dict[str, Any]]],
        train_labels: List[int],
        val_data: Optional[Union[List[str], List[Dict[str, Any]]]] = None,
        val_labels: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Train a baseline or hybrid classifier and evaluate on validation data.
        """
        start_time = time.time()
        logger.info(f"Starting model training (samples: {len(train_data)})...")

        # Fit model
        model.fit(train_data, train_labels)
        train_time = time.time() - start_time

        val_metrics = {}
        if val_data is not None and val_labels is not None:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(val_data)
                if isinstance(probs, np.ndarray) and probs.ndim == 2:
                    y_prob = probs[:, 1] if probs.shape[1] == 2 else probs[:, 0]
                elif isinstance(probs, list):
                    y_prob = probs
                else:
                    y_prob = probs
            else:
                y_prob = None

            y_pred = model.predict(val_data) if hasattr(model, "predict") else [1 if p >= 0.5 else 0 for p in y_prob]
            val_metrics = evaluate_predictions(val_labels, y_pred, y_prob)
            logger.info(f"Validation Metrics: F1={val_metrics.get('f1', 0):.4f}, Acc={val_metrics.get('accuracy', 0):.4f}")

        return {
            "training_time_seconds": train_time,
            "validation_metrics": val_metrics
        }
