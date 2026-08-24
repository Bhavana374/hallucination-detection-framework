"""
Structured Claim Extractor Module.

Segments AI-generated responses into structured claim objects with sentence indices
and metadata mappings.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class StructuredClaim:
    """Dataclass representing an extracted factual claim with position metadata."""
    claim_id: str
    claim_text: str
    sentence_index: int
    response_mapping: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "sentence_index": self.sentence_index,
            "response_mapping": self.response_mapping
        }


class ClaimExtractor:
    """
    Sentence-boundary claim extractor for AI responses.
    """

    def __init__(self, min_length: int = 3):
        self.min_length = min_length

    def extract_claims(self, response_text: str) -> List[StructuredClaim]:
        """
        Segment text into individual claim objects.

        Args:
            response_text: Full response text string.

        Returns:
            List of StructuredClaim instances.
        """
        if not response_text or not response_text.strip():
            return []

        # Sentence boundary split keeping punctuation
        sentences = re.split(r'(?<=[.!?])\s+', response_text.strip())
        claims = []

        for idx, sentence in enumerate(sentences):
            clean_sent = sentence.strip()
            if len(clean_sent) >= self.min_length:
                claim_obj = StructuredClaim(
                    claim_id=f"claim_{idx+1:03d}",
                    claim_text=clean_sent,
                    sentence_index=idx,
                    response_mapping={
                        "character_start": response_text.find(clean_sent),
                        "character_end": response_text.find(clean_sent) + len(clean_sent),
                        "total_sentences": len(sentences)
                    }
                )
                claims.append(claim_obj)

        if not claims:
            claims.append(StructuredClaim(
                claim_id="claim_001",
                claim_text=response_text.strip(),
                sentence_index=0,
                response_mapping={"character_start": 0, "character_end": len(response_text), "total_sentences": 1}
            ))

        return claims
