"""Unit tests for dataset exploration, statistical metrics, and validation reporting."""

import json
from pathlib import Path
from src.data.schema import ClaimEvidenceItem, DatasetSplit
from src.data.explorer import (
    compute_statistics,
    compute_jaccard_overlap,
    compute_unigram_recall,
    DatasetExplorer,
    simple_tokenize,
)
from src.data.sample_generator import generate_benchmark_dataset


def test_compute_statistics():
    """Verify numeric descriptive statistics computation."""
    data = [10.0, 20.0, 30.0, 40.0, 50.0]
    stats = compute_statistics(data)

    assert stats["count"] == 5
    assert stats["mean"] == 30.0
    assert stats["median"] == 30.0
    assert stats["min"] == 10.0
    assert stats["max"] == 50.0


def test_lexical_overlap_metrics():
    """Verify Jaccard similarity and unigram recall."""
    claim = "The cat sat on the mat."
    evidence = "A black cat sat comfortably on the blue mat."

    jaccard = compute_jaccard_overlap(claim, evidence)
    recall = compute_unigram_recall(claim, evidence)

    assert 0.0 < jaccard <= 1.0
    # Every token in claim ("the", "cat", "sat", "on", "the", "mat") is present in evidence
    assert recall == 1.0


def test_dataset_explorer_report(tmp_path: Path):
    """Verify DatasetExplorer report generation and persistence."""
    items_train = [
        ClaimEvidenceItem("c1", "r1", "Paris is the capital of France.", "Paris is France capital.", 0, 0),
        ClaimEvidenceItem("c2", "r2", "Berlin is the capital of France.", "Paris is France capital.", 1, 1),
    ]
    items_val = [
        ClaimEvidenceItem("c3", "r3", "Rome is the capital of Italy.", "Rome is Italy capital.", 0, 0),
    ]
    items_test = [
        ClaimEvidenceItem("c4", "r4", "Madrid is the capital of Spain.", "Madrid is Spain capital.", 0, 0),
    ]

    split = DatasetSplit(train=items_train, val=items_val, test=items_test)
    explorer = DatasetExplorer(split)

    full_report = explorer.generate_full_report()
    assert full_report["dataset_summary"]["total_samples"] == 4
    assert full_report["leakage_audit"]["has_leakage"] is False
    assert full_report["overall"]["total_items"] == 4

    table_dir = tmp_path / "tables"
    metrics_dir = tmp_path / "metrics"
    md_file, json_file = explorer.save_validation_reports(table_dir, metrics_dir)

    assert Path(md_file).exists()
    assert Path(json_file).exists()

    md_content = Path(md_file).read_text(encoding="utf-8")
    assert "# Dataset Exploration & Validation Report" in md_content
    assert "Train-Val Claim Overlap" in md_content


def test_generate_benchmark_dataset(tmp_path: Path):
    """Verify automated benchmark sample dataset generation."""
    raw_file = tmp_path / "raw" / "samples.jsonl"
    split_dir = tmp_path / "processed"

    summary = generate_benchmark_dataset(
        output_raw_file=raw_file,
        output_split_dir=split_dir,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    assert Path(summary["raw_file"]).exists()
    assert summary["total_claim_items"] == 40  # 20 samples * 2 (1 factual + 1 hallucinated each)
    assert summary["train_size"] == 28
    assert summary["val_size"] == 6
    assert summary["test_size"] == 6
