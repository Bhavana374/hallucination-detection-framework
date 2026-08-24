"""Modular dataset loaders with schema validation, format parsing, and serialization."""

import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from src.data.schema import RawSample, ClaimEvidenceItem, DatasetSplit, validate_item
from src.utils.logger import setup_logger

logger = setup_logger("data_loader")


class BaseDatasetLoader:
    """Base abstract interface for dataset ingestion and formatting."""

    def __init__(self, dataset_name: str = "generic"):
        self.dataset_name = dataset_name

    def load_from_file(self, file_path: Union[str, Path]) -> List[ClaimEvidenceItem]:
        """Load items from a file path."""
        raise NotImplementedError("Subclasses must implement load_from_file.")


class HaluEvalLoader(BaseDatasetLoader):
    """Loader for the HaluEval LLM hallucination benchmark format."""

    def __init__(self):
        super().__init__(dataset_name="halueval")

    def parse_sample(self, raw_json: Dict[str, Any], idx: int) -> List[ClaimEvidenceItem]:
        """Parse a single raw HaluEval record into positive and negative claim-evidence items.

        In HaluEval, each record contains:
          - knowledge / context (grounding evidence)
          - question / prompt (query)
          - right_answer (factual)
          - hallucinated_answer (hallucinated)

        Args:
            raw_json: Raw JSON object.
            idx: Global index identifier.

        Returns:
            List containing factual and hallucinated ClaimEvidenceItems.
        """
        items: List[ClaimEvidenceItem] = []
        evidence = raw_json.get("knowledge", "") or raw_json.get("document", "") or raw_json.get("context", "")
        evidence = str(evidence).strip()

        # 1. Factual Answer Instance
        factual_answer = raw_json.get("right_answer", "") or raw_json.get("factual_answer", "") or raw_json.get("ground_truth", "")
        if factual_answer:
            items.append(
                ClaimEvidenceItem(
                    claim_id=f"halueval_{idx:06d}_factual",
                    response_id=f"halueval_{idx:06d}",
                    claim_text=str(factual_answer).strip(),
                    evidence_text=evidence,
                    binary_label=0,  # Factual
                    nli_label=0,     # Entailment
                    source_dataset="halueval",
                    metadata={"query": raw_json.get("question", raw_json.get("query", ""))},
                )
            )

        # 2. Hallucinated Answer Instance
        hallucinated_answer = raw_json.get("hallucinated_answer", "") or raw_json.get("hallucination", "")
        if hallucinated_answer:
            items.append(
                ClaimEvidenceItem(
                    claim_id=f"halueval_{idx:06d}_hallucinated",
                    response_id=f"halueval_{idx:06d}",
                    claim_text=str(hallucinated_answer).strip(),
                    evidence_text=evidence,
                    binary_label=1,  # Hallucinated
                    nli_label=1,     # Contradiction
                    source_dataset="halueval",
                    metadata={"query": raw_json.get("question", raw_json.get("query", ""))},
                )
            )

        return items

    def load_from_file(self, file_path: Union[str, Path]) -> List[ClaimEvidenceItem]:
        """Load and parse HaluEval JSON / JSONL records from file.

        Args:
            file_path: Path to the raw HaluEval file.

        Returns:
            List of validated ClaimEvidenceItem instances.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"HaluEval dataset file not found at: {path.resolve()}")

        items: List[ClaimEvidenceItem] = []
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    parsed_items = self.parse_sample(record, idx)
                    for item in parsed_items:
                        validate_item(item)
                        items.append(item)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed JSON line {idx}: {e}")

        logger.info(f"Loaded {len(items)} claim-evidence items from {path.name}")
        return items


class GenericJSONLLoader(BaseDatasetLoader):
    """Loader for standard structured claim-evidence JSONL files."""

    def __init__(self, dataset_name: str = "generic"):
        super().__init__(dataset_name=dataset_name)

    def load_from_file(self, file_path: Union[str, Path]) -> List[ClaimEvidenceItem]:
        """Load items from structured JSONL containing ClaimEvidenceItem fields."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found at: {path.resolve()}")

        items: List[ClaimEvidenceItem] = []
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    item = ClaimEvidenceItem.from_dict(record)
                    validate_item(item)
                    items.append(item)
                except Exception as e:
                    logger.warning(f"Error parsing record line {idx}: {e}")

        logger.info(f"Loaded {len(items)} items from {path.name}")
        return items


def split_dataset(
    items: List[ClaimEvidenceItem],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify: bool = True,
) -> DatasetSplit:
    """Split items into train, validation, and test partitions with stratification and zero leakage.

    Args:
        items: Full list of ClaimEvidenceItem instances.
        train_ratio: Proportion for training (default: 0.70).
        val_ratio: Proportion for validation (default: 0.15).
        test_ratio: Proportion for testing (default: 0.15).
        seed: Random seed for deterministic reproducibility.
        stratify: Whether to preserve class label distributions across splits.

    Returns:
        DatasetSplit container.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-5:
        raise ValueError(f"Split ratios must sum to 1.0, got: {total}")

    rng = random.Random(seed)

    if stratify:
        # Group by binary label
        def get_label(item):
            if hasattr(item, "binary_label"):
                return item.binary_label
            if isinstance(item, dict):
                return item.get("binary_label", item.get("hallucination_label", 0))
            return 0

        factual_items = [it for it in items if get_label(it) == 0]
        hallucinated_items = [it for it in items if get_label(it) == 1]

        rng.shuffle(factual_items)
        rng.shuffle(hallucinated_items)

        def partition_group(group: List[ClaimEvidenceItem]):
            n = len(group)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            train_part = group[:n_train]
            val_part = group[n_train : n_train + n_val]
            test_part = group[n_train + n_val :]
            return train_part, val_part, test_part

        f_tr, f_va, f_te = partition_group(factual_items)
        h_tr, h_va, h_te = partition_group(hallucinated_items)

        train = f_tr + h_tr
        val = f_va + h_va
        test = f_te + h_te

        rng.shuffle(train)
        rng.shuffle(val)
        rng.shuffle(test)
    else:
        shuffled = list(items)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train = shuffled[:n_train]
        val = shuffled[n_train : n_train + n_val]
        test = shuffled[n_train + n_val :]

    split = DatasetSplit(
        train=train,
        val=val,
        test=test,
        metadata={
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "seed": seed,
            "stratified": stratify,
            "total_samples": len(items),
        },
    )

    logger.info(
        f"Split complete -> Train: {split.train_size}, Val: {split.val_size}, Test: {split.test_size} (Total: {split.total_size})"
    )
    return split


def save_split_to_jsonl(split: DatasetSplit, output_dir: Union[str, Path]) -> Dict[str, str]:
    """Save DatasetSplit partitions to structured JSONL files.

    Args:
        split: DatasetSplit container.
        output_dir: Directory where train.jsonl, val.jsonl, and test.jsonl will be saved.

    Returns:
        Dictionary mapping partition names to written file paths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, str] = {}
    for name in ["train", "val", "test"]:
        items: List[ClaimEvidenceItem] = getattr(split, name)
        file_path = out_path / f"{name}.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
        paths[name] = str(file_path)

    # Save split metadata summary
    meta_path = out_path / "split_summary.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_size": split.train_size,
                "val_size": split.val_size,
                "test_size": split.test_size,
                "train_distribution": split.get_class_distribution("train"),
                "val_distribution": split.get_class_distribution("val"),
                "test_distribution": split.get_class_distribution("test"),
                "metadata": split.metadata,
            },
            f,
            indent=2,
        )

    logger.info(f"Dataset split serialized successfully to: {out_path}")
    return paths


def load_split_from_jsonl(input_dir: Union[str, Path]) -> DatasetSplit:
    """Load DatasetSplit from directory containing train.jsonl, val.jsonl, and test.jsonl.

    Args:
        input_dir: Directory containing partition JSONL files.

    Returns:
        DatasetSplit container.
    """
    dir_path = Path(input_dir)
    loader = GenericJSONLLoader()

    train = loader.load_from_file(dir_path / "train.jsonl")
    val = loader.load_from_file(dir_path / "val.jsonl")
    test = loader.load_from_file(dir_path / "test.jsonl")

    meta_path = dir_path / "split_summary.json"
    meta = {}
    if meta_path.is_file():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    return DatasetSplit(train=train, val=val, test=test, metadata=meta)
