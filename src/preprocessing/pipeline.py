"""Leakage-safe end-to-end dataset preprocessing pipeline."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

from src.data.schema import ClaimEvidenceItem, DatasetSplit, validate_item, check_split_leakage
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.deduplicator import ClaimDeduplicator
from src.utils.logger import setup_logger

logger = setup_logger("preprocessing_pipeline")


class DatasetPreprocessor:
    """Preprocess, normalize, clean, and validate claim-evidence dataset splits."""

    def __init__(
        self,
        preserve_case: bool = True,
        strip_html: bool = True,
        min_claim_chars: int = 5,
        max_claim_chars: int = 1000,
        enable_deduplication: bool = False,
        near_duplicate_threshold: float = 0.95,
    ):
        self.cleaner = TextCleaner(
            preserve_case_for_transformers=preserve_case,
            strip_html_tags=strip_html,
        )
        self.min_claim_chars = min_claim_chars
        self.max_claim_chars = max_claim_chars
        self.enable_dedup = enable_deduplication
        self.deduplicator = ClaimDeduplicator(near_duplicate_threshold=near_duplicate_threshold)

    def preprocess_item(self, item: ClaimEvidenceItem) -> Optional[ClaimEvidenceItem]:
        """Clean and validate a single ClaimEvidenceItem.

        Args:
            item: Raw ClaimEvidenceItem.

        Returns:
            Cleaned ClaimEvidenceItem or None if item does not meet length constraints.
        """
        cleaned_claim = self.cleaner.clean(item.claim_text)
        cleaned_evidence = self.cleaner.clean(item.evidence_text)

        if len(cleaned_claim) < self.min_claim_chars or len(cleaned_claim) > self.max_claim_chars:
            return None

        # Return new cleaned item preserving all IDs and labels
        return ClaimEvidenceItem(
            claim_id=item.claim_id,
            response_id=item.response_id,
            claim_text=cleaned_claim,
            evidence_text=cleaned_evidence,
            binary_label=item.binary_label,
            nli_label=item.nli_label,
            evidence_retrieval_score=item.evidence_retrieval_score,
            source_dataset=item.source_dataset,
            metadata=dict(item.metadata),
        )

    def preprocess_items_list(self, items: List[ClaimEvidenceItem]) -> Tuple[List[ClaimEvidenceItem], Dict[str, Any]]:
        """Preprocess a list of items.

        Args:
            items: Input list of ClaimEvidenceItems.

        Returns:
            Tuple of (cleaned_items, summary_dict).
        """
        cleaned_items: List[ClaimEvidenceItem] = []
        dropped_short_count = 0

        for item in items:
            cleaned = self.preprocess_item(item)
            if cleaned is not None:
                validate_item(cleaned)
                cleaned_items.append(cleaned)
            else:
                dropped_short_count += 1

        dedup_audit = {}
        if self.enable_dedup:
            cleaned_items, dedup_audit = self.deduplicator.deduplicate(cleaned_items)

        summary = {
            "initial_count": len(items),
            "retained_count": len(cleaned_items),
            "dropped_length_constraints": dropped_short_count,
            "dedup_audit": dedup_audit,
        }
        return cleaned_items, summary

    def preprocess_split(self, split: DatasetSplit) -> Tuple[DatasetSplit, Dict[str, Any]]:
        """Preprocess train, val, and test partitions strictly in isolation.

        Args:
            split: DatasetSplit container.

        Returns:
            Tuple of (preprocessed_split, audit_report).
        """
        train_clean, train_summary = self.preprocess_items_list(split.train)
        val_clean, val_summary = self.preprocess_items_list(split.val)
        test_clean, test_summary = self.preprocess_items_list(split.test)

        preprocessed_split = DatasetSplit(
            train=train_clean,
            val=val_clean,
            test=test_clean,
            metadata={
                "preprocessing": {
                    "train_summary": train_summary,
                    "val_summary": val_summary,
                    "test_summary": test_summary,
                }
            },
        )

        leakage_audit = check_split_leakage(preprocessed_split)
        report = {
            "leakage_audit": leakage_audit,
            "train_summary": train_summary,
            "val_summary": val_summary,
            "test_summary": test_summary,
            "final_sizes": {
                "train": preprocessed_split.train_size,
                "val": preprocessed_split.val_size,
                "test": preprocessed_split.test_size,
            },
        }

        logger.info(
            f"Preprocessed split complete: Train={preprocessed_split.train_size}, Val={preprocessed_split.val_size}, Test={preprocessed_split.test_size}"
        )
        return preprocessed_split, report
