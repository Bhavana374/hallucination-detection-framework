"""
Semantic Similarity Feature Extractor.

Computes semantic distance, cosine similarity, and vector alignments
between claim and retrieved evidence.
"""

import numpy as np
from typing import Dict, List, Any
from src.utils.logger import get_logger

logger = get_logger("semantic_features")


class SemanticFeatureExtractor:
    """
    Extracts dense semantic similarity features between claims and evidence passages.
    """

    def __init__(self, embedder=None):
        self.embedder = embedder

    def _fallback_embedding(self, text: str, dim: int = 128) -> np.ndarray:
        """Compute simple hash-based pseudo embedding if dense embedder is absent."""
        vec = np.zeros(dim, dtype=np.float32)
        words = text.lower().split()
        for w in words:
            idx = abs(hash(w)) % dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def extract_features(self, claim: str, evidence: str, claim_vec: np.ndarray = None, evidence_vec: np.ndarray = None) -> Dict[str, float]:
        """
        Extract semantic similarity features.

        Returns:
            Dict containing:
            - semantic_cosine_sim
            - semantic_euclidean_dist
            - semantic_dot_product
            - semantic_embedding_magnitude_ratio
        """
        if not claim or not evidence:
            return {
                "semantic_cosine_sim": 0.0,
                "semantic_euclidean_dist": 2.0,
                "semantic_dot_product": 0.0,
                "semantic_embedding_magnitude_ratio": 1.0,
            }

        if claim_vec is None or evidence_vec is None:
            if self.embedder is not None and hasattr(self.embedder, "encode"):
                encoded = self.embedder.encode([claim, evidence])
                v1, v2 = encoded[0], encoded[1]
            else:
                v1 = self._fallback_embedding(claim)
                v2 = self._fallback_embedding(evidence)
        else:
            v1, v2 = claim_vec, evidence_vec

        v1 = v1.squeeze()
        v2 = v2.squeeze()

        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)

        if n1 > 0:
            v1_norm = v1 / n1
        else:
            v1_norm = v1

        if n2 > 0:
            v2_norm = v2 / n2
        else:
            v2_norm = v2

        dot = float(np.dot(v1_norm, v2_norm))
        cosine_sim = float(np.clip(dot, -1.0, 1.0))
        euclidean_dist = float(np.linalg.norm(v1_norm - v2_norm))
        mag_ratio = float(n1 / n2) if n2 > 0 else 1.0

        return {
            "semantic_cosine_sim": cosine_sim,
            "semantic_euclidean_dist": euclidean_dist,
            "semantic_dot_product": dot,
            "semantic_embedding_magnitude_ratio": mag_ratio,
        }
