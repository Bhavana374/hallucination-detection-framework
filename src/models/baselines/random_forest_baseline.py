"""
TF-IDF + Random Forest baseline model for claim-level hallucination detection.
"""

import pickle
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
from src.utils.logger import get_logger

logger = get_logger("tfidf_random_forest")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-Learn not found. Using fallback decision ensemble.")


class FallbackRandomForest:
    """Pure-Python decision ensemble fallback when scikit-learn is absent."""

    def __init__(self, n_estimators: int = 10):
        self.n_estimators = n_estimators
        self.vocab: Dict[str, int] = {}

    def fit(self, texts: List[str], labels: List[int]):
        words = sorted(list(set(w.lower() for t in texts for w in t.split())))
        self.vocab = {w: i for i, w in enumerate(words[:1000])}
        return self

    def predict_proba(self, texts: List[str]) -> List[float]:
        results = []
        for t in texts:
            # Simple lexical frequency score
            hits = sum(1 for w in t.lower().split() if w in self.vocab)
            prob = 1.0 / (1.0 + np.exp(-hits * 0.1))
            results.append(float(prob))
        return results


class TfidfRandomForestBaseline:
    """
    TF-IDF + Random Forest Classifier (Experiment 2 Baseline).
    """

    def __init__(self, max_features: int = 5000, n_estimators: int = 100, random_state: int = 42):
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.is_fitted = False

        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), stop_words="english")
            self.classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        else:
            self.vectorizer = None
            self.classifier = FallbackRandomForest(n_estimators=n_estimators)

    def fit(self, texts: List[str], labels: List[int]) -> "TfidfRandomForestBaseline":
        start_time = time.time()
        logger.info(f"Fitting TF-IDF + Random Forest on {len(texts)} samples...")

        if SKLEARN_AVAILABLE and self.vectorizer is not None and self.classifier is not None:
            X = self.vectorizer.fit_transform(texts)
            self.classifier.fit(X, labels)
        else:
            self.classifier.fit(texts, labels)

        self.is_fitted = True
        logger.info(f"Model training completed in {time.time() - start_time:.4f} seconds.")
        return self

    def predict_proba(self, texts: List[str]) -> List[float]:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba.")

        if SKLEARN_AVAILABLE and self.vectorizer is not None and self.classifier is not None:
            X = self.vectorizer.transform(texts)
            probs = self.classifier.predict_proba(X)
            if probs.shape[1] == 2:
                return probs[:, 1].tolist()
            return probs[:, 0].tolist()
        else:
            return self.classifier.predict_proba(texts)

    def predict(self, texts: List[str], threshold: float = 0.5) -> List[int]:
        probs = self.predict_proba(texts)
        return [1 if p >= threshold else 0 for p in probs]

    def save(self, filepath: Union[str, Path]) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "classifier": self.classifier,
                "is_fitted": self.is_fitted
            }, f)
        logger.info(f"Saved Random Forest model to {filepath}")

    def load(self, filepath: Union[str, Path]) -> None:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.vectorizer = data["vectorizer"]
            self.classifier = data["classifier"]
            self.is_fitted = data["is_fitted"]
        logger.info(f"Loaded Random Forest model from {filepath}")
