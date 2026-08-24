"""Unit tests for dataset schemas, HaluEval parser, splitting, and leakage detection."""

import json
import pytest
from pathlib import Path
from src.data.schema import ClaimEvidenceItem, DatasetSplit, validate_item, check_split_leakage
from src.data.loader import (
    HaluEvalLoader,
    split_dataset,
    save_split_to_jsonl,
    load_split_from_jsonl,
)


def test_claim_evidence_item_validation():
    """Verify item validation rules."""
    valid_item = ClaimEvidenceItem(
        claim_id="test_001",
        response_id="resp_001",
        claim_text="The Eiffel Tower is in Paris.",
        evidence_text="The Eiffel Tower is located in Paris, France.",
        binary_label=0,
        nli_label=0,
    )
    assert validate_item(valid_item) is True

    # Empty claim text should fail
    with pytest.raises(ValueError):
        invalid_item = ClaimEvidenceItem(
            claim_id="test_002",
            response_id="resp_002",
            claim_text="",
            evidence_text="Some evidence",
            binary_label=0,
        )
        validate_item(invalid_item)

    # Invalid binary label should fail
    with pytest.raises(ValueError):
        invalid_item = ClaimEvidenceItem(
            claim_id="test_003",
            response_id="resp_003",
            claim_text="Some claim",
            evidence_text="Some evidence",
            binary_label=5,
        )
        validate_item(invalid_item)


def test_halueval_parser():
    """Verify parsing of raw HaluEval records."""
    loader = HaluEvalLoader()
    raw_sample = {
        "knowledge": "Albert Einstein won the 1921 Nobel Prize in Physics for his discovery of the law of the photoelectric effect.",
        "question": "What did Albert Einstein win the Nobel Prize for?",
        "right_answer": "Albert Einstein won the Nobel Prize for his discovery of the photoelectric effect.",
        "hallucinated_answer": "Albert Einstein won the Nobel Prize for his development of the theory of general relativity.",
    }

    items = loader.parse_sample(raw_sample, idx=1)
    assert len(items) == 2

    factual_item = [it for it in items if it.binary_label == 0][0]
    assert factual_item.nli_label == 0
    assert "photoelectric effect" in factual_item.claim_text
    assert factual_item.evidence_text == raw_sample["knowledge"]

    hallucinated_item = [it for it in items if it.binary_label == 1][0]
    assert hallucinated_item.nli_label == 1
    assert "general relativity" in hallucinated_item.claim_text


def test_dataset_stratified_split_and_leakage():
    """Verify stratified splitting and leakage auditor."""
    items = []
    for i in range(100):
        # 50 factual, 50 hallucinated
        label = 0 if i < 50 else 1
        items.append(
            ClaimEvidenceItem(
                claim_id=f"claim_{i:03d}",
                response_id=f"resp_{i:03d}",
                claim_text=f"Factual statement number {i}" if label == 0 else f"Hallucinated statement number {i}",
                evidence_text=f"Grounding evidence passage {i}",
                binary_label=label,
                nli_label=0 if label == 0 else 1,
            )
        )

    split = split_dataset(items, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42, stratify=True)

    assert split.train_size == 70
    assert split.val_size == 14
    assert split.test_size == 16

    # Verify no leakage
    leakage_report = check_split_leakage(split)
    assert leakage_report["has_leakage"] is False
    assert leakage_report["id_overlap_train_val_count"] == 0
    assert leakage_report["claim_overlap_train_test_count"] == 0


def test_leakage_detection_trigger():
    """Verify that the leakage auditor catches intentional contamination."""
    shared_item = ClaimEvidenceItem(
        claim_id="leak_001",
        response_id="resp_leak",
        claim_text="The moon is made of green cheese.",
        evidence_text="The moon is rocky.",
        binary_label=1,
    )
    split = DatasetSplit(
        train=[shared_item],
        val=[],
        test=[shared_item],
    )
    leakage_report = check_split_leakage(split)
    assert leakage_report["has_leakage"] is True
    assert leakage_report["id_overlap_train_test_count"] == 1


def test_jsonl_serialization_roundtrip(tmp_path: Path):
    """Verify saving and loading dataset splits to/from JSONL."""
    items = [
        ClaimEvidenceItem(
            claim_id="c1",
            response_id="r1",
            claim_text="Water boils at 100C at 1 atm.",
            evidence_text="At sea level, the boiling point of water is 100 degrees Celsius.",
            binary_label=0,
            nli_label=0,
        ),
        ClaimEvidenceItem(
            claim_id="c2",
            response_id="r2",
            claim_text="Water boils at 500C at 1 atm.",
            evidence_text="At sea level, the boiling point of water is 100 degrees Celsius.",
            binary_label=1,
            nli_label=1,
        ),
    ]

    split = split_dataset(items, train_ratio=0.5, val_ratio=0.0, test_ratio=0.5, seed=42, stratify=False)
    paths = save_split_to_jsonl(split, tmp_path)

    assert Path(paths["train"]).exists()
    assert Path(paths["test"]).exists()

    loaded_split = load_split_from_jsonl(tmp_path)
    assert loaded_split.train_size == 1
    assert loaded_split.test_size == 1
    assert loaded_split.train[0].claim_text == split.train[0].claim_text
