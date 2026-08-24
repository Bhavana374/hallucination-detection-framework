"""
Generate detailed error analysis payload for the HaluEval hybrid model predictions.
"""

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_vectorizer import FeatureVectorizer
from src.utils.logger import get_logger

logger = get_logger("generate_error_analysis")


def main():
    logger.info("Initializing FeatureVectorizer...")
    vectorizer = FeatureVectorizer()

    # Load HaluEval dataset raw to align with indices
    raw_path = PROJECT_ROOT / "data" / "raw" / "halueval" / "qa_samples.jsonl"
    logger.info(f"Loading raw data from {raw_path}...")
    records = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Exclude duplicates in the same order as benchmark script validation
    valid_labels = {"yes", "no"}
    seen_hashes = set()
    valid_records = []
    for rec in records:
        row_hash = json.dumps([rec.get("answer", ""), rec.get("knowledge", "")])
        if row_hash not in seen_hashes and rec.get("hallucination") in valid_labels:
            seen_hashes.add(row_hash)
            valid_records.append(rec)

    # Recreate the train/test splits in the exact same order as split meta (using seed 42)
    import numpy as np
    rng = np.random.RandomState(42)

    pos = [r for r in valid_records if r["hallucination"] == "yes"]
    neg = [r for r in valid_records if r["hallucination"] == "no"]

    rng.shuffle(pos)
    rng.shuffle(neg)

    test_ratio = 0.20
    n_test_pos = int(len(pos) * test_ratio)
    n_test_neg = int(len(neg) * test_ratio)

    test_pos = pos[:n_test_pos]
    test_neg = neg[:n_test_neg]
    test_data = test_pos + test_neg
    rng.shuffle(test_data)

    # Load exp07 predictions
    pred_path = PROJECT_ROOT / "results" / "halueval" / "full" / "exp07_hybrid_rf" / "predictions.csv"
    logger.info(f"Loading predictions from {pred_path}...")

    predictions = []
    with open(pred_path, "r", encoding="utf-8") as f:
        headers = f.readline().strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 6:
                # Handle quoted answers if present
                idx = int(parts[0])
                is_correct = int(parts[-1])
                prob = float(parts[-2])
                yp = int(parts[-3])
                yt = int(parts[-4])
                predictions.append({
                    "index": idx,
                    "true_label": yt,
                    "predicted_label": yp,
                    "probability": prob,
                    "is_correct": is_correct
                })

    logger.info(f"Analyzing {len(predictions)} predictions against {len(test_data)} test items...")

    # Categories
    false_positives = []
    false_negatives = []
    correct_positives = []
    correct_negatives = []

    for i, pred in enumerate(predictions):
        item = test_data[i]
        yt = pred["true_label"]
        yp = pred["predicted_label"]
        prob = pred["probability"]

        # Run vectorizer to extract detailed feature features
        f_dict = vectorizer.extract_dict(item["answer"], item["knowledge"], 0.90)

        info = {
            "index": i,
            "original_prompt": item["question"],
            "generated_response": item["answer"],
            "claim": item["answer"],
            "retrieved_evidence": item["knowledge"],
            "predicted_label": yp,
            "actual_label": yt,
            "prediction_probability": prob,
            "retrieval_score": 0.90,
            "nli_probabilities": {
                "entailment": float(f_dict.get("nli_prob_entailment", 0.0)),
                "contradiction": float(f_dict.get("nli_prob_contradiction", 0.0)),
                "neutral": float(f_dict.get("nli_prob_neutral", 0.0))
            },
            "features": {
                "semantic_similarity": float(f_dict.get("semantic_cosine_sim", 0.0)),
                "entity_overlap_ratio": float(f_dict.get("entity_overlap_ratio", 0.0)),
                "lexical_jaccard": float(f_dict.get("lexical_jaccard_similarity", 0.0)),
                "hedge_word_count": float(f_dict.get("hedge_word_count", 0.0))
            }
        }

        if yt == 1 and yp == 0:
            false_negatives.append(info)
        elif yt == 0 and yp == 1:
            false_positives.append(info)
        elif yt == 1 and yp == 1:
            if len(correct_positives) < 50: # cap correct categories for space
                correct_positives.append(info)
        elif yt == 0 and yp == 0:
            if len(correct_negatives) < 50:
                correct_negatives.append(info)

    error_analysis_data = {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "correct_positives": correct_positives,
        "correct_negatives": correct_negatives,
        "summary": {
            "total_errors": len(false_positives) + len(false_negatives),
            "false_positives_count": len(false_positives),
            "false_negatives_count": len(false_negatives)
        }
    }

    out_path = PROJECT_ROOT / "results" / "halueval" / "error_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(error_analysis_data, f, indent=2)

    logger.info(f"Successfully generated error analysis dataset at: {out_path}")
    logger.info(f"Total Errors found: {len(false_positives) + len(false_negatives)} "
                f"(FP: {len(false_positives)}, FN: {len(false_negatives)})")


if __name__ == "__main__":
    main()
