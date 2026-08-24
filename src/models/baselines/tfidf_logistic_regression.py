"""TF-IDF + Logistic Regression baseline model for claim-level hallucination detection."""

import math
import pickle
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

from src.utils.logger import setup_logger

logger = setup_logger("tfidf_logistic_regression")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class FallbackTfidfLR:
    """Self-contained pure-Python TF-IDF + Logistic Regression implementation."""

    def __init__(self, max_features: int = 5000, lr: float = 0.1, epochs: int = 100):
        self.max_features = max_features
        self.lr = lr
        self.epochs = epochs
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.weights: List[float] = []
        self.bias: float = 0.0

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _get_ngrams(self, text: str) -> List[str]:
        tokens = self._tokenize(text)
        unigrams = tokens
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
        return unigrams + bigrams

    def fit(self, texts: List[str], labels: List[int]) -> "FallbackTfidfLR":
        # Build vocabulary & compute IDF
        doc_count = len(texts)
        doc_freq: Dict[str, int] = {}
        for text in texts:
            unique_terms = set(self._get_ngrams(text))
            for term in unique_terms:
                doc_freq[term] = doc_freq.get(term, 0) + 1

        sorted_terms = sorted(doc_freq.keys(), key=lambda t: doc_freq[t], reverse=True)[:self.max_features]
        self.vocab = {term: idx for idx, term in enumerate(sorted_terms)}
        self.idf = {term: math.log((1 + doc_count) / (1 + doc_freq[term])) + 1.0 for term in self.vocab}

        # Vectorize
        X = [self._transform_single(t) for t in texts]
        y = labels
        n_features = len(self.vocab)
        self.weights = [0.0] * n_features
        self.bias = 0.0

        # Train with SGD
        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                pred = self._sigmoid(sum(w * x for w, x in zip(self.weights, xi)) + self.bias)
                err = yi - pred
                for j in range(n_features):
                    if xi[j] != 0.0:
                        self.weights[j] += self.lr * err * xi[j]
                self.bias += self.lr * err
        return self

    def _sigmoid(self, z: float) -> float:
        z_clamped = max(min(z, 20.0), -20.0)
        return 1.0 / (1.0 + math.exp(-z_clamped))

    def _transform_single(self, text: str) -> List[float]:
        terms = self._get_ngrams(text)
        counts: Dict[str, int] = {}
        for term in terms:
            if term in self.vocab:
                counts[term] = counts.get(term, 0) + 1

        vec = [0.0] * len(self.vocab)
        norm_sq = 0.0
        for term, cnt in counts.items():
            idx = self.vocab[term]
            tf_idf = cnt * self.idf[term]
            vec[idx] = tf_idf
            norm_sq += tf_idf ** 2

        norm = math.sqrt(norm_sq) if norm_sq > 0 else 1.0
        return [v / norm for v in vec]

    def predict_proba(self, texts: List[str]) -> List[float]:
        probs: List[float] = []
        for text in texts:
            vec = self._transform_single(text)
            z = sum(w * x for w, x in zip(self.weights, vec)) + self.bias
            probs.append(self._sigmoid(z))
        return probs

    def predict(self, texts: List[str], threshold: float = 0.5) -> List[int]:
        probs = self.predict_proba(texts)
        return [1 if p >= threshold else 0 for p in probs]


class TfidfLogisticRegressionModel:
    """TF-IDF + Logistic Regression baseline model wrapper."""

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        c_param: float = 1.0,
        random_state: int = 42,
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.c_param = c_param
        self.random_state = random_state

        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                sublinear_tf=True,
            )
            self.classifier = LogisticRegression(
                C=c_param,
                random_state=random_state,
                max_iter=1000,
                class_weight="balanced",
            )
            self._use_fallback = False
        else:
            logger.warning("Scikit-Learn not found. Using fallback native TF-IDF + LR implementation.")
            self.fallback_model = FallbackTfidfLR(max_features=max_features)
            self._use_fallback = True

        self.is_fitted = False
        self.training_time_sec = 0.0

    def fit(self, train_texts: List[str], train_labels: List[int]) -> "TfidfLogisticRegressionModel":
        """Fit TF-IDF vocabulary and Logistic Regression parameters strictly on training claims.

        Args:
            train_texts: List of claim strings.
            train_labels: List of binary integer labels (0: Factual, 1: Hallucinated).

        Returns:
            self
        """
        start_time = time.time()
        logger.info(f"Fitting TF-IDF + Logistic Regression on {len(train_texts)} samples...")

        if not self._use_fallback:
            X_train = self.vectorizer.fit_transform(train_texts)
            self.classifier.fit(X_train, train_labels)
        else:
            self.fallback_model.fit(train_texts, train_labels)

        self.is_fitted = True
        self.training_time_sec = time.time() - start_time
        logger.info(f"Model training completed in {self.training_time_sec:.4f} seconds.")
        return self

    def predict_proba(self, texts: List[str]) -> List[float]:
        """Predict hallucination probability (class 1 probability) for input claims.

        Args:
            texts: List of claim strings.

        Returns:
            List of float probabilities [0.0 - 1.0].
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict_proba.")

        if not self._use_fallback:
            X = self.vectorizer.transform(texts)
            probs = self.classifier.predict_proba(X)
            # Return probability of class 1 (Hallucinated)
            return [float(p[1]) for p in probs]
        else:
            return self.fallback_model.predict_proba(texts)

    def predict(self, texts: List[str], threshold: float = 0.5) -> List[int]:
        """Predict binary hallucination labels for input claims.

        Args:
            texts: List of claim strings.
            threshold: Decision boundary threshold (default: 0.5).

        Returns:
            List of integer labels (0: Factual, 1: Hallucinated).
        """
        probs = self.predict_proba(texts)
        return [1 if p >= threshold else 0 for p in probs]

    def save_model(self, output_path: Union[str, Path]) -> str:
        """Serialize trained model artifacts to disk.

        Args:
            output_path: Path to pickle file.

        Returns:
            String path where model was saved.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_type": "tfidf_logistic_regression",
            "is_fallback": self._use_fallback,
            "max_features": self.max_features,
            "ngram_range": self.ngram_range,
            "training_time_sec": self.training_time_sec,
            "vectorizer": self.vectorizer if not self._use_fallback else None,
            "classifier": self.classifier if not self._use_fallback else None,
            "fallback_model": self.fallback_model if self._use_fallback else None,
        }

        with open(path, "wb") as f:
            pickle.dump(payload, f)

        logger.info(f"Saved model checkpoint to: {path}")
        return str(path)

    @classmethod
    def load_model(cls, model_path: Union[str, Path]) -> "TfidfLogisticRegressionModel":
        """Load serialized model checkpoint from disk.

        Args:
            model_path: Path to pickle file.

        Returns:
            Instantiated and fitted TfidfLogisticRegressionModel.
        """
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found at: {path.resolve()}")

        with open(path, "rb") as f:
            payload = pickle.load(f)

        instance = cls(
            max_features=payload.get("max_features", 5000),
            ngram_range=payload.get("ngram_range", (1, 2)),
        )
        instance._use_fallback = payload.get("is_fallback", False)
        instance.training_time_sec = payload.get("training_time_sec", 0.0)

        if not instance._use_fallback:
            instance.vectorizer = payload["vectorizer"]
            instance.classifier = payload["classifier"]
        else:
            instance.fallback_model = payload["fallback_model"]

        instance.is_fitted = True
        logger.info(f"Loaded model checkpoint from: {path}")
        return instance


TfidfLogisticRegressionBaseline = TfidfLogisticRegressionModel
