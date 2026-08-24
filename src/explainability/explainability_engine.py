"""
Explainability & Attribution Engine for Hallucination Detection.

Generates human-understandable explanations, feature importance breakdowns,
and evidence attribution mappings for detected claims.
"""

from typing import Dict, List, Any, Optional
from src.utils.logger import get_logger

logger = get_logger("explainability_engine")


class ExplainabilityEngine:
    """
    Computes evidence attribution and multi-dimensional feature explanations.
    """

    def generate_claim_explanation(
        self,
        claim: str,
        evidence: str,
        feature_dict: Dict[str, float],
        hallucination_prob: float,
        verdict: str
    ) -> Dict[str, Any]:
        """
        Generate claim-level explanation payload.

        Args:
            claim: Claim text string.
            evidence: Retrieved evidence passage string.
            feature_dict: Extracted multi-dimensional feature map.
            hallucination_prob: Calibrated hallucination probability [0.0 - 1.0].
            verdict: 'Hallucinated' or 'Factual'.

        Returns:
            Dict containing detailed attribution breakdown and human summary.
        """
        entailment = feature_dict.get("nli_prob_entailment", 0.0)
        contradiction = feature_dict.get("nli_prob_contradiction", 0.0)
        cosine_sim = feature_dict.get("semantic_cosine_sim", 0.0)
        entity_overlap = feature_dict.get("entity_overlap_ratio", 1.0)

        # Primary risk triggers
        triggers = []
        if contradiction > 0.4:
            triggers.append(f"High NLI Contradiction score ({contradiction:.2%}) against evidence.")
        if entity_overlap < 0.5:
            triggers.append(f"Low entity overlap ({entity_overlap:.2%}) - claim introduces ungrounded named entities.")
        if cosine_sim < 0.35:
            triggers.append(f"Low semantic cosine similarity ({cosine_sim:.2f}) with source evidence.")
        if feature_dict.get("hedge_word_count", 0) > 0:
            triggers.append("Presence of linguistic uncertainty cues or hedge words.")

        if not triggers:
            if verdict == "Hallucinated":
                triggers.append("Cumulative multi-feature deviation from grounded context.")
            else:
                triggers.append("Strong semantic, NLI entailment, and entity alignment with source context.")

        human_summary = (
            f"Claim is categorized as **{verdict}** with a hallucination risk score of {hallucination_prob:.2%}. "
            + " ".join(triggers)
        )

        return {
            "claim": claim,
            "evidence": evidence,
            "verdict": verdict,
            "hallucination_probability": float(hallucination_prob),
            "key_features": {
                "nli_entailment": float(entailment),
                "nli_contradiction": float(contradiction),
                "semantic_similarity": float(cosine_sim),
                "entity_overlap_ratio": float(entity_overlap),
                "lexical_jaccard": float(feature_dict.get("lexical_jaccard_similarity", 0.0))
            },
            "risk_triggers": triggers,
            "explanation_summary": human_summary
        }
