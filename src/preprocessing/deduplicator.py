"""Near-duplicate claim detection and dataset deduplication utility."""

import re
from typing import List, Set, Tuple, Dict, Any
from src.data.schema import ClaimEvidenceItem
from src.utils.logger import setup_logger

logger = setup_logger("deduplicator")


def get_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from lowercased text."""
    clean_text = re.sub(r"\s+", " ", text.lower().strip())
    if len(clean_text) < n:
        return {clean_text}
    return {clean_text[i : i + n] for i in range(len(clean_text) - n + 1)}


def compute_ngram_jaccard(text_a: str, text_b: str, n: int = 3) -> float:
    """Compute character n-gram Jaccard similarity."""
    ngrams_a = get_ngrams(text_a, n)
    ngrams_b = get_ngrams(text_b, n)
    if not ngrams_a or not ngrams_b:
        return 0.0
    intersection = ngrams_a.intersection(ngrams_b)
    union = ngrams_a.union(ngrams_b)
    return len(intersection) / len(union) if union else 0.0


class ClaimDeduplicator:
    """Deduplication engine to filter exact and near-duplicate claims."""

    def __init__(self, near_duplicate_threshold: float = 0.92, ngram_size: int = 3):
        self.threshold = near_duplicate_threshold
        self.ngram_size = ngram_size

    def deduplicate(self, items: List[ClaimEvidenceItem]) -> Tuple[List[ClaimEvidenceItem], Dict[str, Any]]:
        """Deduplicate items based on exact match and high n-gram similarity.

        Args:
            items: Input list of ClaimEvidenceItems.

        Returns:
            Tuple of (deduplicated_items, audit_summary).
        """
        unique_items: List[ClaimEvidenceItem] = []
        seen_exact_claims: Set[str] = set()
        pruned_exact_count = 0
        pruned_near_count = 0

        for item in items:
            normalized_claim = item.claim_text.strip().lower()

            # Check exact duplicate
            if normalized_claim in seen_exact_claims:
                pruned_exact_count += 1
                continue

            # Check near duplicate against already admitted unique items
            is_near_dup = False
            for admitted in unique_items:
                sim = compute_ngram_jaccard(item.claim_text, admitted.claim_text, n=self.ngram_size)
                if sim >= self.threshold:
                    is_near_dup = True
                    pruned_near_count += 1
                    break

            if not is_near_dup:
                seen_exact_claims.add(normalized_claim)
                unique_items.append(item)

        audit = {
            "initial_items_count": len(items),
            "retained_items_count": len(unique_items),
            "pruned_exact_duplicates": pruned_exact_count,
            "pruned_near_duplicates": pruned_near_count,
        }

        logger.info(
            f"Deduplication complete: Retained {len(unique_items)}/{len(items)} (Pruned {pruned_exact_count} exact, {pruned_near_count} near-duplicates)"
        )
        return unique_items, audit
