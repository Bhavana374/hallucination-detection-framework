"""
Feature Vectorizer & Scaler Pipeline.

Aggregates NLI, Semantic, Factual, and Linguistic feature extractors into
leakage-safe multi-dimensional feature vectors.
"""

import pickle
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from src.features.nli_features import NLIFeatureExtractor
from src.features.semantic_features import SemanticFeatureExtractor
from src.features.factual_features import FactualFeatureExtractor
from src.features.linguistic_features import LinguisticFeatureExtractor
from src.utils.logger import get_logger

logger = get_logger("feature_vectorizer")

try:
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("Scikit-Learn not found. Using fallback standard scaler.")


class FallbackScaler:
    """Fallback z-score standard scaler when scikit-learn is absent."""

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1e-10
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            return X
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class FeatureVectorizer:
    """
    Main Multi-Dimensional Feature Vectorizer engine for the Hybrid Model.
    """

    def __init__(self, embedder=None, nli_model_name: str = "cross-encoder/nli-deberta-v3-small", excluded_features: Optional[List[str]] = None):
        self.nli_extractor = NLIFeatureExtractor(model_name=nli_model_name)
        self.semantic_extractor = SemanticFeatureExtractor(embedder=embedder)
        self.factual_extractor = FactualFeatureExtractor()
        self.linguistic_extractor = LinguisticFeatureExtractor()

        self.scaler = StandardScaler() if HAS_SKLEARN else FallbackScaler()
        self.feature_names: List[str] = []
        self.excluded_features = excluded_features or []
        self.is_fitted = False

    def extract_dict(self, claim: str, evidence: str, retrieval_score: float = 0.0) -> Dict[str, float]:
        """Extract merged feature dictionary for a single claim-evidence instance."""
        f_nli = self.nli_extractor.extract_features(claim, evidence)
        f_sem = self.semantic_extractor.extract_features(claim, evidence)
        f_fact = self.factual_extractor.extract_features(claim, evidence, retrieval_score)
        f_ling = self.linguistic_extractor.extract_features(claim, evidence)

        merged = {}
        merged.update(f_nli)
        merged.update(f_sem)
        merged.update(f_fact)
        merged.update(f_ling)
        return merged

    def transform_single(self, claim: str, evidence: str, retrieval_score: float = 0.0) -> np.ndarray:
        """Extract and transform feature vector for a single instance."""
        f_dict = self.extract_dict(claim, evidence, retrieval_score)
        if not self.feature_names:
            self.feature_names = [k for k in sorted(list(f_dict.keys())) if k not in self.excluded_features]

        vec = np.array([[f_dict[k] for k in self.feature_names]], dtype=np.float32)
        if self.is_fitted:
            vec = self.scaler.transform(vec)
        return vec.squeeze(0)

    def extract_matrix(self, instances: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
        """
        Extract raw unscaled feature matrix for a list of sample dicts.
        Each sample dict must contain 'claim', 'evidence', and optional 'retrieval_score'.
        """
        if not instances:
            return np.zeros((0, 0), dtype=np.float32), []

        feature_dicts = []
        for item in instances:
            claim = item.get("claim", "")
            evidence = item.get("evidence", "")
            score = item.get("retrieval_score", 0.0)
            feature_dicts.append(self.extract_dict(claim, evidence, score))

        if not self.feature_names:
            self.feature_names = [k for k in sorted(list(feature_dicts[0].keys())) if k not in self.excluded_features]

        rows = []
        for fd in feature_dicts:
            row = [fd.get(k, 0.0) for k in self.feature_names]
            rows.append(row)

        X = np.array(rows, dtype=np.float32)
        return X, self.feature_names

    def fit(self, instances: List[Dict[str, Any]]) -> "FeatureVectorizer":
        """Fit scaler on training data partition."""
        X, _ = self.extract_matrix(instances)
        if X.shape[0] > 0:
            self.scaler.fit(X)
            self.is_fitted = True
            logger.info(f"Fitted FeatureVectorizer scaler on {X.shape[0]} samples across {X.shape[1]} features.")
        return self

    def transform(self, instances: List[Dict[str, Any]]) -> np.ndarray:
        """Transform instances into scaled feature matrix."""
        X, _ = self.extract_matrix(instances)
        if X.shape[0] > 0 and self.is_fitted:
            X = self.scaler.transform(X)
        return X.astype(np.float32)

    def fit_transform(self, instances: List[Dict[str, Any]]) -> np.ndarray:
        """Fit scaler and return transformed matrix, extracting features only once."""
        X, _ = self.extract_matrix(instances)
        if X.shape[0] > 0:
            self.scaler.fit(X)
            self.is_fitted = True
            logger.info(f"Fitted FeatureVectorizer scaler on {X.shape[0]} samples across {X.shape[1]} features.")
            X = self.scaler.transform(X)
        return X.astype(np.float32)

    def save(self, filepath: str) -> None:
        """Serialize fitted vectorizer and scaler."""
        with open(filepath, "wb") as f:
            pickle.dump({
                "feature_names": self.feature_names,
                "excluded_features": self.excluded_features,
                "scaler": self.scaler,
                "is_fitted": self.is_fitted
            }, f)
        logger.info(f"Saved FeatureVectorizer to {filepath}")

    def load(self, filepath: str) -> None:
        """Load serialized vectorizer and scaler."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.feature_names = data["feature_names"]
            self.excluded_features = data.get("excluded_features", [])
            self.scaler = data["scaler"]
            self.is_fitted = data["is_fitted"]
        logger.info(f"Loaded FeatureVectorizer from {filepath}")
