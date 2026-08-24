"""Data schemas, structured dataclasses, and validation rules for hallucination detection."""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Set
import json


@dataclass
class RawSample:
    """Raw sample ingested from benchmark datasets (e.g., HaluEval, FEVER, FactCC)."""

    sample_id: str
    query: str
    response: str
    grounding_evidence: str
    label: int  # 0: Factual, 1: Hallucinated
    nli_label: Optional[int] = None  # 0: Entailment, 1: Contradiction, 2: Neutral
    source_dataset: str = "halueval"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to dictionary."""
        return asdict(self)


@dataclass
class ClaimEvidenceItem:
    """Segmented factual claim paired with candidate or gold grounding evidence."""

    claim_id: str
    response_id: str
    claim_text: str
    evidence_text: str
    binary_label: int  # 0: Factual / Supported, 1: Hallucinated / Contradicted
    nli_label: int = 0  # 0: Entailment, 1: Contradiction, 2: Neutral
    evidence_retrieval_score: float = 1.0
    source_dataset: str = "halueval"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert item to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClaimEvidenceItem":
        """Instantiate item from dictionary."""
        return cls(
            claim_id=str(data["claim_id"]),
            response_id=str(data.get("response_id", "")),
            claim_text=str(data["claim_text"]),
            evidence_text=str(data.get("evidence_text", "")),
            binary_label=int(data.get("binary_label", 0)),
            nli_label=int(data.get("nli_label", 0)),
            evidence_retrieval_score=float(data.get("evidence_retrieval_score", 1.0)),
            source_dataset=str(data.get("source_dataset", "halueval")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class DatasetSplit:
    """Container for train, validation, and test partitions."""

    train: List[ClaimEvidenceItem] = field(default_factory=list)
    val: List[ClaimEvidenceItem] = field(default_factory=list)
    test: List[ClaimEvidenceItem] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        return iter((self.train, self.val, self.test))

    @property
    def train_size(self) -> int:
        return len(self.train)

    @property
    def val_size(self) -> int:
        return len(self.val)

    @property
    def test_size(self) -> int:
        return len(self.test)

    @property
    def total_size(self) -> int:
        return self.train_size + self.val_size + self.test_size

    def get_class_distribution(self, split_name: str = "train") -> Dict[int, int]:
        """Compute binary class distribution for a specified split."""
        split_data = getattr(self, split_name, [])
        dist: Dict[int, int] = {0: 0, 1: 0}
        for item in split_data:
            dist[item.binary_label] = dist.get(item.binary_label, 0) + 1
        return dist


def validate_item(item: ClaimEvidenceItem) -> bool:
    """Validate that a ClaimEvidenceItem meets minimum quality constraints.

    Args:
        item: ClaimEvidenceItem instance.

    Returns:
        True if valid, raises ValueError otherwise.
    """
    if not item.claim_id or not isinstance(item.claim_id, str):
        raise ValueError("ClaimEvidenceItem must have a non-empty string claim_id.")
    if not item.claim_text or len(item.claim_text.strip()) == 0:
        raise ValueError("ClaimEvidenceItem must have non-empty claim_text.")
    if item.binary_label not in (0, 1):
        raise ValueError(f"Invalid binary_label {item.binary_label}. Must be 0 (Factual) or 1 (Hallucinated).")
    if item.nli_label not in (0, 1, 2):
        raise ValueError(f"Invalid nli_label {item.nli_label}. Must be 0 (Entailment), 1 (Contradiction), or 2 (Neutral).")
    return True


def check_split_leakage(split: DatasetSplit) -> Dict[str, Any]:
    """Verify that there is zero claim text or sample ID overlap across splits.

    Args:
        split: DatasetSplit container.

    Returns:
        Dictionary reporting leakage audit results.
    """
    train_ids: Set[str] = {item.claim_id for item in split.train}
    val_ids: Set[str] = {item.claim_id for item in split.val}
    test_ids: Set[str] = {item.claim_id for item in split.test}

    train_claims: Set[str] = {item.claim_text.strip().lower() for item in split.train}
    val_claims: Set[str] = {item.claim_text.strip().lower() for item in split.val}
    test_claims: Set[str] = {item.claim_text.strip().lower() for item in split.test}

    id_overlap_train_val = train_ids.intersection(val_ids)
    id_overlap_train_test = train_ids.intersection(test_ids)
    id_overlap_val_test = val_ids.intersection(test_ids)

    claim_overlap_train_val = train_claims.intersection(val_claims)
    claim_overlap_train_test = train_claims.intersection(test_claims)
    claim_overlap_val_test = val_claims.intersection(test_claims)

    has_leakage = bool(
        id_overlap_train_val
        or id_overlap_train_test
        or id_overlap_val_test
        or claim_overlap_train_val
        or claim_overlap_train_test
        or claim_overlap_val_test
    )

    return {
        "has_leakage": has_leakage,
        "id_overlap_train_val_count": len(id_overlap_train_val),
        "id_overlap_train_test_count": len(id_overlap_train_test),
        "id_overlap_val_test_count": len(id_overlap_val_test),
        "claim_overlap_train_val_count": len(claim_overlap_train_val),
        "claim_overlap_train_test_count": len(claim_overlap_train_test),
        "claim_overlap_val_test_count": len(claim_overlap_val_test),
    }
