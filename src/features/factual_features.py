"""
Factual & Retrieval Consistency Feature Extractor.

Extracts entity overlap (NER), numerical/date precision, evidence rank scores,
and factual grounding metrics.
"""

import re
from typing import Dict, List, Set, Any
from src.utils.logger import get_logger

logger = get_logger("factual_features")


class FactualFeatureExtractor:
    """
    Extracts factual consistency and entity grounding features.
    """

    def _extract_entities(self, text: str) -> Set[str]:
        """Simple rule-based NER: extracts capitalized terms and proper nouns."""
        words = text.split()
        entities = set()
        for w in words:
            clean_w = re.sub(r'^\W+|\W+$', '', w)
            if clean_w and clean_w[0].isupper() and len(clean_w) > 1:
                entities.add(clean_w.lower())
        return entities

    def _extract_numbers(self, text: str) -> Set[str]:
        """Extract numerical digits, years, percentages, and currencies."""
        numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', text))
        return numbers

    def extract_features(self, claim: str, evidence: str, retrieval_score: float = 0.0) -> Dict[str, float]:
        """
        Extract factual consistency metrics between claim and evidence.

        Returns:
            Dict containing:
            - entity_overlap_ratio
            - missing_entity_count
            - number_match_ratio
            - missing_number_count
            - retrieval_score
        """
        if not claim or not evidence:
            return {
                "entity_overlap_ratio": 0.0,
                "missing_entity_count": 0.0,
                "number_match_ratio": 1.0,
                "missing_number_count": 0.0,
                "retrieval_score": float(retrieval_score),
            }

        # Entity overlap
        claim_ents = self._extract_entities(claim)
        evid_ents = self._extract_entities(evidence)

        if claim_ents:
            matched_ents = claim_ents.intersection(evid_ents)
            ent_ratio = len(matched_ents) / len(claim_ents)
            missing_ents = len(claim_ents) - len(matched_ents)
        else:
            ent_ratio = 1.0
            missing_ents = 0.0

        # Numerical overlap
        claim_nums = self._extract_numbers(claim)
        evid_nums = self._extract_numbers(evidence)

        if claim_nums:
            matched_nums = claim_nums.intersection(evid_nums)
            num_ratio = len(matched_nums) / len(claim_nums)
            missing_nums = len(claim_nums) - len(matched_nums)
        else:
            num_ratio = 1.0
            missing_nums = 0.0

        return {
            "entity_overlap_ratio": float(ent_ratio),
            "missing_entity_count": float(missing_ents),
            "number_match_ratio": float(num_ratio),
            "missing_number_count": float(missing_nums),
            "retrieval_score": float(retrieval_score),
        }
