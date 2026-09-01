"""
Natural Language Inference (NLI) Feature Extractor.

Computes posterior probabilities P(Entailment), P(Contradiction), P(Neutral)
between claim and retrieved evidence.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from src.utils.logger import get_logger

logger = get_logger("nli_features")

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("PyTorch/Transformers not installed. Using heuristic NLI feature extractor.")


class NLIFeatureExtractor:
    """
    Extracts NLI posterior probabilities for claim-evidence pairs.
    """

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

        if HAS_TRANSFORMERS:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.model.eval()
                logger.info(f"Loaded NLI cross-encoder model: {model_name}")
            except Exception as e:
                logger.warning(f"Could not load HuggingFace NLI model '{model_name}': {e}. Using heuristic NLI.")
                self.tokenizer = None
                self.model = None

    def extract_features(self, claim: str, evidence: str) -> Dict[str, float]:
        """
        Extract NLI probabilities for a single (claim, evidence) pair.

        Returns:
            Dict containing:
            - nli_prob_entailment
            - nli_prob_contradiction
            - nli_prob_neutral
            - nli_entailment_ratio (entailment - contradiction)
        """
        if not claim or not evidence:
            return {
                "nli_prob_entailment": 0.33,
                "nli_prob_contradiction": 0.33,
                "nli_prob_neutral": 0.34,
                "nli_entailment_ratio": 0.0,
            }

        if self.model is not None and self.tokenizer is not None:
            try:
                inputs = self.tokenizer(evidence, claim, return_tensors="pt", truncation=True, max_length=256)
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                
                # Resolve label mapping dynamically from model config if available
                id2label = getattr(self.model.config, "id2label", None)
                if id2label:
                    label_map = {v.lower(): int(k) for k, v in id2label.items()}
                    p_ent = float(probs[label_map.get("entailment", 1)])
                    p_contra = float(probs[label_map.get("contradiction", 0)])
                    p_neut = float(probs[label_map.get("neutral", 2)])
                elif len(probs) == 3:
                    # Default cross-encoder/nli-deberta-v3-small mapping: 0 -> contradiction, 1 -> entailment, 2 -> neutral
                    p_contra, p_ent, p_neut = float(probs[0]), float(probs[1]), float(probs[2])
                else:
                    p_ent, p_contra, p_neut = float(probs[0]), float(probs[1]), 1.0 - float(probs[0] + probs[1])

                return {
                    "nli_prob_entailment": p_ent,
                    "nli_prob_contradiction": p_contra,
                    "nli_prob_neutral": max(0.0, p_neut),
                    "nli_entailment_ratio": p_ent - p_contra,
                }
            except Exception as e:
                logger.debug(f"NLI inference error: {e}. Falling back to heuristic NLI.")

        # Heuristic lexical fallback:
        c_words = set(claim.lower().split())
        e_words = set(evidence.lower().split())
        overlap = len(c_words.intersection(e_words)) / max(1, len(c_words))

        # Check explicit negation / contradiction cues
        negations = {"not", "never", "no", "false", "unlike", "different", "failed"}
        c_neg = len(c_words.intersection(negations)) > 0
        e_neg = len(e_words.intersection(negations)) > 0
        negation_mismatch = (c_neg != e_neg) and (overlap > 0.3)

        if negation_mismatch:
            p_contra = min(0.9, 0.5 + overlap * 0.4)
            p_ent = max(0.05, 0.4 - overlap * 0.3)
            p_neut = 1.0 - (p_contra + p_ent)
        else:
            p_ent = min(0.95, overlap * 0.8 + 0.1)
            p_contra = max(0.05, (1.0 - overlap) * 0.3)
            p_neut = max(0.0, 1.0 - (p_ent + p_contra))

        return {
            "nli_prob_entailment": float(p_ent),
            "nli_prob_contradiction": float(p_contra),
            "nli_prob_neutral": float(p_neut),
            "nli_entailment_ratio": float(p_ent - p_contra),
        }
