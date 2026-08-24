"""
Hybrid Meta-Classifier for Hallucination Detection (Exp 7).

Combines NLI probabilities, semantic similarity, factual consistency,
and linguistic cues with Random Forest / Gradient Boosting tree ensemble.
"""

import pickle
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

import numpy as np
from src.features.feature_vectorizer import FeatureVectorizer
from src.utils.logger import get_logger

logger = get_logger("hybrid_classifier")

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-Learn not found. Using fallback tree classifier for Hybrid Model.")


class FallbackTreeEnsemble:
    """Fallback classifier for multi-dimensional feature matrix when sklearn is missing."""

    def fit(self, X: np.ndarray, y: np.ndarray):
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float32)
        # Average feature values heuristic as pseudo probability
        mean_vals = np.mean(X, axis=1)
        probs_pos = 1.0 / (1.0 + np.exp(-mean_vals))
        probs_neg = 1.0 - probs_pos
        return np.column_stack([probs_neg, probs_pos]).astype(np.float32)


class HybridClassifier:
    """
    Hybrid Meta-Classifier operating over multi-dimensional feature vectors (Exp 7).
    """

    def __init__(
        self,
        classifier_type: str = "random_forest",
        n_estimators: int = 100,
        random_state: int = 42,
        vectorizer: Optional[FeatureVectorizer] = None
    ):
        self.classifier_type = classifier_type
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.vectorizer = vectorizer or FeatureVectorizer()
        self.is_fitted = False

        if SKLEARN_AVAILABLE:
            if classifier_type == "gradient_boosting":
                self.model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state)
            else:
                self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        else:
            self.model = FallbackTreeEnsemble()

    def fit(self, instances: List[Dict[str, Any]], labels: List[int]) -> "HybridClassifier":
        """
        Fit vectorizer scaler and meta-classifier on training instances.
        Each instance dict must have 'claim', 'evidence', and optional 'retrieval_score'.
        """
        start_time = time.time()
        logger.info(f"Extracting multi-dimensional features for {len(instances)} training samples...")
        X = self.vectorizer.fit_transform(instances)
        y = np.array(labels)

        logger.info(f"Fitting Hybrid {self.classifier_type} meta-classifier on shape {X.shape}...")
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info(f"Hybrid model training completed in {time.time() - start_time:.4f} seconds.")
        return self

    def predict_proba(self, instances: List[Dict[str, Any]]) -> np.ndarray:
        """Predict calibrated hallucination probabilities for input instances."""
        if not self.is_fitted:
            raise RuntimeError("HybridClassifier must be fitted before predict_proba.")

        X = self.vectorizer.transform(instances)
        if SKLEARN_AVAILABLE and hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
            return probs.astype(np.float32)
        else:
            return self.model.predict_proba(X)

    def predict_hallucination_score(self, instances: List[Dict[str, Any]]) -> List[float]:
        """Return list of hallucination probabilities (Class 1 = Hallucinated)."""
        probs = self.predict_proba(instances)
        if probs.shape[1] == 2:
            return probs[:, 1].tolist()
        return probs[:, 0].tolist()

    def get_feature_importances(self) -> Dict[str, float]:
        """Get feature importance dictionary sorted by importance weight."""
        if not self.is_fitted or not SKLEARN_AVAILABLE or not hasattr(self.model, "feature_importances_"):
            return {name: 1.0 / max(1, len(self.vectorizer.feature_names)) for name in self.vectorizer.feature_names}

        importances = self.model.feature_importances_
        result = {}
        for name, imp in zip(self.vectorizer.feature_names, importances):
            result[name] = float(imp)

        # Sort descending
        return dict(sorted(result.items(), key=lambda item: item[1], reverse=True))

    def save(self, filepath: Union[str, Path]) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({
                "classifier_type": self.classifier_type,
                "n_estimators": self.n_estimators,
                "random_state": self.random_state,
                "vectorizer": self.vectorizer,
                "model": self.model,
                "is_fitted": self.is_fitted
            }, f)
        logger.info(f"Saved Hybrid model to {filepath}")

    def load(self, filepath: Union[str, Path]) -> None:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.classifier_type = data["classifier_type"]
            self.n_estimators = data["n_estimators"]
            self.random_state = data["random_state"]
            self.vectorizer = data["vectorizer"]
            self.model = data["model"]
            self.is_fitted = data["is_fitted"]
        logger.info(f"Loaded Hybrid model from {filepath}")
