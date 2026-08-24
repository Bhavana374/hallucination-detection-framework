"""Data ingestion, schema validation, exploration, and dataset loading module."""

from src.data.schema import (
    RawSample,
    ClaimEvidenceItem,
    DatasetSplit,
    validate_item,
    check_split_leakage,
)
from src.data.loader import (
    BaseDatasetLoader,
    HaluEvalLoader,
    GenericJSONLLoader,
    split_dataset,
    save_split_to_jsonl,
    load_split_from_jsonl,
)
from src.data.sample_generator import (
    generate_benchmark_dataset,
    CURATED_BENCHMARK_SAMPLES,
)
from src.data.explorer import (
    DatasetExplorer,
    compute_statistics,
    compute_jaccard_overlap,
    compute_unigram_recall,
    simple_tokenize,
)
from src.data.visualizer import DatasetVisualizer

__all__ = [
    "RawSample",
    "ClaimEvidenceItem",
    "DatasetSplit",
    "validate_item",
    "check_split_leakage",
    "BaseDatasetLoader",
    "HaluEvalLoader",
    "GenericJSONLLoader",
    "split_dataset",
    "save_split_to_jsonl",
    "load_split_from_jsonl",
    "generate_benchmark_dataset",
    "CURATED_BENCHMARK_SAMPLES",
    "DatasetExplorer",
    "compute_statistics",
    "compute_jaccard_overlap",
    "compute_unigram_recall",
    "simple_tokenize",
    "DatasetVisualizer",
]
