"""
Linguistic & Lexical Consistency Feature Extractor.

Extracts lexical overlap (Jaccard, token precision/recall), sentence length ratio,
uncertainty markers, hedge words, and punctuation cues.
"""

import re
from typing import Dict, List, Set, Any
from src.utils.logger import get_logger

logger = get_logger("linguistic_features")

HEDGE_WORDS = {
    "maybe", "perhaps", "possibly", "allegedly", "reportedly", "presumably",
    "unclear", "uncertain", "suggests", "seems", "appears", "might", "could",
    "unknown", "unconfirmed", "supposedly"
}


class LinguisticFeatureExtractor:
    """
    Extracts lexical overlap and linguistic uncertainty cues.
    """

    def extract_features(self, claim: str, evidence: str) -> Dict[str, float]:
        """
        Extract linguistic consistency features.

        Returns:
            Dict containing:
            - lexical_jaccard_similarity
            - token_precision
            - token_recall
            - token_f1
            - length_ratio
            - hedge_word_count
            - claim_word_count
        """
        if not claim:
            return {
                "lexical_jaccard_similarity": 0.0,
                "token_precision": 0.0,
                "token_recall": 0.0,
                "token_f1": 0.0,
                "length_ratio": 0.0,
                "hedge_word_count": 0.0,
                "claim_word_count": 0.0,
            }

        c_tokens = re.findall(r'\w+', claim.lower())
        e_tokens = re.findall(r'\w+', evidence.lower()) if evidence else []

        c_set = set(c_tokens)
        e_set = set(e_tokens)

        # Jaccard
        intersection = c_set.intersection(e_set)
        union = c_set.union(e_set)
        jaccard = len(intersection) / max(1, len(union))

        # Precision, Recall, F1
        precision = len(intersection) / max(1, len(c_set))
        recall = len(intersection) / max(1, len(e_set))
        if precision + recall > 0:
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        # Length ratio
        len_ratio = len(c_tokens) / max(1, len(e_tokens)) if e_tokens else 0.0

        # Hedge words
        hedges = sum(1 for t in c_tokens if t in HEDGE_WORDS)

        return {
            "lexical_jaccard_similarity": float(jaccard),
            "token_precision": float(precision),
            "token_recall": float(recall),
            "token_f1": float(f1),
            "length_ratio": float(min(len_ratio, 10.0)),
            "hedge_word_count": float(hedges),
            "claim_word_count": float(len(c_tokens)),
        }
