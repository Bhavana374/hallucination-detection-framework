"""
Data Leakage Check: Train/Test split cross-contamination audit.

Checks if identical or near-identical questions, knowledge, or answers 
cross train/test split boundaries.
"""

import json
import sys
import re
from pathlib import Path
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger

logger = get_logger("split_leakage_check")


def tokenize(text: str) -> set:
    return set(re.findall(r'\w+', text.lower()))


def jaccard_similarity(s1: set, s2: set) -> float:
    union = s1.union(s2)
    if not union:
        return 0.0
    return len(s1.intersection(s2)) / len(union)


def main():
    # Load raw data
    raw_path = PROJECT_ROOT / "data" / "raw" / "halueval" / "qa_samples.jsonl"
    records = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Exclude duplicates (mirroring benchmark script)
    valid_labels = {"yes", "no"}
    seen_hashes = set()
    valid_records = []
    for rec in records:
        row_hash = json.dumps([rec.get("answer", ""), rec.get("knowledge", "")])
        if row_hash not in seen_hashes and rec.get("hallucination") in valid_labels:
            seen_hashes.add(row_hash)
            valid_records.append(rec)

    # Recreate splits (seed 42)
    rng = np.random.RandomState(42)
    pos = [r for r in valid_records if r["hallucination"] == "yes"]
    neg = [r for r in valid_records if r["hallucination"] == "no"]

    rng.shuffle(pos)
    rng.shuffle(neg)

    test_ratio = 0.20
    n_test_pos = int(len(pos) * test_ratio)
    n_test_neg = int(len(neg) * test_ratio)

    test_pos = pos[:n_test_pos]
    train_pos = pos[n_test_pos:]
    test_neg = neg[:n_test_neg]
    train_neg = neg[n_test_neg:]

    train_data = train_pos + train_neg
    test_data = test_pos + test_neg

    logger.info(f"Loaded Train set ({len(train_data)} samples) and Test set ({len(test_data)} samples)")

    logger.info("Tokenizing questions and knowledge fields for leakage checking...")
    train_q_tokens = [tokenize(r["question"]) for r in train_data]
    train_k_tokens = [tokenize(r["knowledge"]) for r in train_data]

    leakage_records = []

    # Run check for a subset of test samples to be computationally practical
    logger.info("Scanning for train-test cross-contamination...")
    for i, test_rec in enumerate(test_data):
        t_q = tokenize(test_rec["question"])
        t_k = tokenize(test_rec["knowledge"])
        t_a = tokenize(test_rec["answer"])

        for j, train_rec in enumerate(train_data):
            # Check exact duplicates crossing boundaries
            q_sim = jaccard_similarity(t_q, train_q_tokens[j])
            k_sim = jaccard_similarity(t_k, train_k_tokens[j])

            # If both context and question match heavily (>0.95)
            if q_sim > 0.95 and k_sim > 0.95:
                leakage_records.append({
                    "test_index": i,
                    "train_index": j,
                    "test_question": test_rec["question"],
                    "train_question": train_rec["question"],
                    "question_similarity": q_sim,
                    "knowledge_similarity": k_sim,
                    "test_answer": test_rec["answer"],
                    "train_answer": train_rec["answer"],
                    "test_label": test_rec["hallucination"],
                    "train_label": train_rec["hallucination"]
                })
                break

    out_path = PROJECT_ROOT / "results" / "halueval" / "train_test_leakage_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(leakage_records, f, indent=2)

    logger.info(f"Split leakage check complete! Saved report to: {out_path}")
    logger.info(f"Found {len(leakage_records)} near-duplicates crossing split boundaries.")
    if len(leakage_records) > 0:
        logger.warning(f"WARNING: Cross-split leakage found in {len(leakage_records)} cases!")


if __name__ == "__main__":
    main()
