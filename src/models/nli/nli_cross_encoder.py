"""
NLI Cross-Encoder for Evidence-Grounded Hallucination Detection (Exp 5 & Exp 6).

Uses the cross-encoder/nli-deberta-v3-small model to compute
Entailment / Neutral / Contradiction scores between evidence and claims.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False
    logger.warning(
        "sentence-transformers not installed. NLI CrossEncoder running in fallback mode."
    )


class NLICrossEncoder:
    """
    Wrapper around the sentence-transformers CrossEncoder for NLI scoring.

    Produces three scores per (premise, hypothesis) pair:
        - Entailment
        - Neutral
        - Contradiction

    The label ordering is read from the model config at load time, so
    we never hard-code which index corresponds to which class.
    """

    DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-small"
    LABEL_NAMES = ("entailment", "neutral", "contradiction")

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.label_mapping: Dict[int, str] = {}

        if HAS_CROSS_ENCODER:
            try:
                kwargs = {}
                if device is not None:
                    kwargs["device"] = device
                self.model = CrossEncoder(model_name, **kwargs)
                self._resolve_label_mapping()
                logger.info(
                    f"Loaded NLI CrossEncoder '{model_name}' "
                    f"(labels: {self.label_mapping})"
                )
            except Exception as e:
                logger.warning(
                    f"Could not load NLI model '{model_name}': {e}. "
                    "Using fallback NLI scorer."
                )
                self.model = None
                self._set_default_label_mapping()
        else:
            self.model = None
            self._set_default_label_mapping()

    # ------------------------------------------------------------------
    # Label mapping helpers
    # ------------------------------------------------------------------
    def _resolve_label_mapping(self) -> None:
        """Read id2label from the underlying model config if available."""
        try:
            config = self.model.model.config
            if hasattr(config, "id2label") and config.id2label:
                self.label_mapping = {
                    int(k): v.lower() for k, v in config.id2label.items()
                }
                return
        except Exception:
            pass
        self._set_default_label_mapping()

    def _set_default_label_mapping(self) -> None:
        """Fallback label ordering matching cross-encoder/nli-deberta-v3-small."""
        self.label_mapping = {i: name for i, name in enumerate(self.LABEL_NAMES)}

    def get_label_index(self, label_name: str) -> int:
        """Return the score index for a given label name (case-insensitive)."""
        target = label_name.strip().lower()
        for idx, name in self.label_mapping.items():
            if name == target:
                return idx
        raise KeyError(
            f"Label '{label_name}' not found in mapping {self.label_mapping}"
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(
        self, premise_hypothesis_pairs: List[Tuple[str, str]]
    ) -> np.ndarray:
        """
        Compute NLI scores for a list of (premise, hypothesis) pairs.

        Args:
            premise_hypothesis_pairs: list of (premise, hypothesis) tuples.

        Returns:
            np.ndarray of shape (N, 3) with softmax scores ordered by
            the model's id2label mapping.
        """
        if not premise_hypothesis_pairs:
            return np.zeros((0, 3), dtype=np.float32)

        if self.model is not None:
            scores = self.model.predict(
                premise_hypothesis_pairs, apply_softmax=True
            )
            return np.asarray(scores, dtype=np.float32)
        else:
            # Fallback: random-ish scores for smoke-testing without the model
            rng = np.random.default_rng(42)
            return rng.dirichlet([1, 1, 1], size=len(premise_hypothesis_pairs)).astype(
                np.float32
            )

    def predict_single(
        self, premise: str, hypothesis: str
    ) -> Dict[str, float]:
        """
        Convenience method: return a {label: score} dict for one pair.
        """
        scores = self.predict([(premise, hypothesis)])[0]
        return {self.label_mapping[i]: float(s) for i, s in enumerate(scores)}
