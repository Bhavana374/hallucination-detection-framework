"""Exploratory Data Analysis (EDA) and data validation reporting engine."""

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

from src.data.schema import ClaimEvidenceItem, DatasetSplit, check_split_leakage
from src.utils.logger import setup_logger

logger = setup_logger("data_explorer")


def simple_tokenize(text: str) -> List[str]:
    """Lightweight whitespace and punctuation tokenizer."""
    return re.findall(r"\b\w+\b", text.lower())


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute comprehensive descriptive statistics for a numeric sequence."""
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "p95": 0.0, "p99": 0.0}

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mean_val = sum(sorted_vals) / n

    variance = sum((x - mean_val) ** 2 for x in sorted_vals) / n if n > 1 else 0.0
    std_val = math.sqrt(variance)

    median_val = sorted_vals[n // 2] if n % 2 != 0 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
    p95_idx = min(int(0.95 * n), n - 1)
    p99_idx = min(int(0.99 * n), n - 1)

    return {
        "count": n,
        "mean": round(mean_val, 2),
        "median": round(median_val, 2),
        "std": round(std_val, 2),
        "min": round(sorted_vals[0], 2),
        "max": round(sorted_vals[-1], 2),
        "p95": round(sorted_vals[p95_idx], 2),
        "p99": round(sorted_vals[p99_idx], 2),
    }


def compute_jaccard_overlap(text_a: str, text_b: str) -> float:
    """Compute token-level Jaccard similarity between two texts."""
    tokens_a = set(simple_tokenize(text_a))
    tokens_b = set(simple_tokenize(text_b))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a.intersection(tokens_b)
    union = tokens_a.union(tokens_b)
    return len(intersection) / len(union) if union else 0.0


def compute_unigram_recall(claim: str, evidence: str) -> float:
    """Compute fraction of claim tokens present in the grounding evidence."""
    claim_tokens = simple_tokenize(claim)
    if not claim_tokens:
        return 0.0
    evidence_tokens_set = set(simple_tokenize(evidence))
    matched = sum(1 for tok in claim_tokens if tok in evidence_tokens_set)
    return matched / len(claim_tokens)


class DatasetExplorer:
    """Comprehensive statistical analyzer and validation reporter for hallucination datasets."""

    def __init__(self, split: DatasetSplit):
        self.split = split

    def analyze_partition(self, items: List[ClaimEvidenceItem], partition_name: str) -> Dict[str, Any]:
        """Perform statistical analysis on a single dataset partition."""
        if not items:
            return {"partition_name": partition_name, "sample_count": 0}

        claim_token_lengths = [len(simple_tokenize(item.claim_text)) for item in items]
        evidence_token_lengths = [len(simple_tokenize(item.evidence_text)) for item in items]
        claim_char_lengths = [len(item.claim_text) for item in items]
        evidence_char_lengths = [len(item.evidence_text) for item in items]

        jaccard_scores = [compute_jaccard_overlap(item.claim_text, item.evidence_text) for item in items]
        unigram_recalls = [compute_unigram_recall(item.claim_text, item.evidence_text) for item in items]

        # Vocabulary computation
        all_tokens: List[str] = []
        for item in items:
            all_tokens.extend(simple_tokenize(item.claim_text))
            all_tokens.extend(simple_tokenize(item.evidence_text))

        vocab_counter = Counter(all_tokens)
        unique_tokens = len(vocab_counter)
        total_tokens = len(all_tokens)
        type_token_ratio = round(unique_tokens / total_tokens, 4) if total_tokens > 0 else 0.0

        # Class counts
        binary_counts = Counter([item.binary_label for item in items])
        nli_counts = Counter([item.nli_label for item in items])

        return {
            "partition_name": partition_name,
            "total_items": len(items),
            "class_distribution": {
                "factual_count (0)": binary_counts.get(0, 0),
                "hallucinated_count (1)": binary_counts.get(1, 0),
                "hallucinated_percentage": round(binary_counts.get(1, 0) / len(items) * 100, 2),
            },
            "nli_distribution": {
                "entailment_count (0)": nli_counts.get(0, 0),
                "contradiction_count (1)": nli_counts.get(1, 0),
                "neutral_count (2)": nli_counts.get(2, 0),
            },
            "claim_token_length": compute_statistics(claim_token_lengths),
            "evidence_token_length": compute_statistics(evidence_token_lengths),
            "claim_char_length": compute_statistics(claim_char_lengths),
            "evidence_char_length": compute_statistics(evidence_char_lengths),
            "lexical_jaccard_overlap": compute_statistics(jaccard_scores),
            "claim_unigram_recall_in_evidence": compute_statistics(unigram_recalls),
            "vocabulary_metrics": {
                "total_tokens": total_tokens,
                "unique_vocabulary": unique_tokens,
                "type_token_ratio": type_token_ratio,
            },
        }

    def generate_full_report(self) -> Dict[str, Any]:
        """Generate comprehensive multi-split statistical validation report."""
        leakage_report = check_split_leakage(self.split)

        train_analysis = self.analyze_partition(self.split.train, "train")
        val_analysis = self.analyze_partition(self.split.val, "val")
        test_analysis = self.analyze_partition(self.split.test, "test")

        all_items = self.split.train + self.split.val + self.split.test
        overall_analysis = self.analyze_partition(all_items, "overall")

        return {
            "dataset_summary": {
                "total_samples": self.split.total_size,
                "train_samples": self.split.train_size,
                "val_samples": self.split.val_size,
                "test_samples": self.split.test_size,
            },
            "leakage_audit": leakage_report,
            "overall": overall_analysis,
            "train": train_analysis,
            "val": val_analysis,
            "test": test_analysis,
        }

    def save_validation_reports(
        self,
        output_table_dir: Union[str, Path] = "results/tables",
        output_metrics_dir: Union[str, Path] = "results/metrics",
    ) -> Tuple[str, str]:
        """Export validation analysis to Markdown table report and JSON metric files.

        Returns:
            Tuple of (markdown_file_path, json_file_path).
        """
        table_path = Path(output_table_dir)
        metrics_path = Path(output_metrics_dir)
        table_path.mkdir(parents=True, exist_ok=True)
        metrics_path.mkdir(parents=True, exist_ok=True)

        report = self.generate_full_report()

        # Save JSON
        json_file = metrics_path / "data_validation_report.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        # Generate Markdown Report
        md_file = table_path / "data_validation_report.md"
        md_content = self._render_markdown_report(report)
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(f"Saved validation report to: {md_file} and {json_file}")
        return str(md_file), str(json_file)

    def _render_markdown_report(self, report: Dict[str, Any]) -> str:
        """Render formatted Markdown table report."""
        leakage = report["leakage_audit"]
        overall = report["overall"]
        train = report["train"]
        val = report["val"]
        test = report["test"]

        lines = [
            "# Dataset Exploration & Validation Report",
            "",
            "**Framework:** Evidence-Grounded Hybrid Transformer Hallucination Detector  ",
            f"**Total Samples:** {report['dataset_summary']['total_samples']}  ",
            f"**Split Proportions:** Train={report['dataset_summary']['train_samples']}, Val={report['dataset_summary']['val_samples']}, Test={report['dataset_summary']['test_samples']}  ",
            "",
            "---",
            "",
            "## 1. Data Leakage & Integrity Audit",
            "",
            f"- **Leakage Detected:** `{'NO (CLEAN)' if not leakage['has_leakage'] else 'YES (WARNING)'}`",
            f"- **Train-Val Claim Overlap:** {leakage['claim_overlap_train_val_count']}",
            f"- **Train-Test Claim Overlap:** {leakage['claim_overlap_train_test_count']}",
            f"- **Val-Test Claim Overlap:** {leakage['claim_overlap_val_test_count']}",
            "",
            "---",
            "",
            "## 2. Partition Class Balance",
            "",
            "| Partition | Total Items | Factual (0) | Hallucinated (1) | Hallucinated % |",
            "| :--- | :--- | :--- | :--- | :--- |",
            f"| **Overall** | {overall.get('total_items', 0)} | {overall.get('class_distribution', {}).get('factual_count (0)', 0)} | {overall.get('class_distribution', {}).get('hallucinated_count (1)', 0)} | {overall.get('class_distribution', {}).get('hallucinated_percentage', 0)}% |",
            f"| **Train** | {train.get('total_items', 0)} | {train.get('class_distribution', {}).get('factual_count (0)', 0)} | {train.get('class_distribution', {}).get('hallucinated_count (1)', 0)} | {train.get('class_distribution', {}).get('hallucinated_percentage', 0)}% |",
            f"| **Val** | {val.get('total_items', 0)} | {val.get('class_distribution', {}).get('factual_count (0)', 0)} | {val.get('class_distribution', {}).get('hallucinated_count (1)', 0)} | {val.get('class_distribution', {}).get('hallucinated_percentage', 0)}% |",
            f"| **Test** | {test.get('total_items', 0)} | {test.get('class_distribution', {}).get('factual_count (0)', 0)} | {test.get('class_distribution', {}).get('hallucinated_count (1)', 0)} | {test.get('class_distribution', {}).get('hallucinated_percentage', 0)}% |",
            "",
            "---",
            "",
            "## 3. Sequence Length & Lexical Statistics (Overall)",
            "",
            "| Metric | Mean ± Std | Median | Min - Max | 95th Percentile |",
            "| :--- | :--- | :--- | :--- | :--- |",
            f"| **Claim Token Length** | {overall['claim_token_length']['mean']} ± {overall['claim_token_length']['std']} | {overall['claim_token_length']['median']} | {overall['claim_token_length']['min']} - {overall['claim_token_length']['max']} | {overall['claim_token_length']['p95']} |",
            f"| **Evidence Token Length** | {overall['evidence_token_length']['mean']} ± {overall['evidence_token_length']['std']} | {overall['evidence_token_length']['median']} | {overall['evidence_token_length']['min']} - {overall['evidence_token_length']['max']} | {overall['evidence_token_length']['p95']} |",
            f"| **Claim-Evidence Jaccard Overlap** | {overall['lexical_jaccard_overlap']['mean']} ± {overall['lexical_jaccard_overlap']['std']} | {overall['lexical_jaccard_overlap']['median']} | {overall['lexical_jaccard_overlap']['min']} - {overall['lexical_jaccard_overlap']['max']} | {overall['lexical_jaccard_overlap']['p95']} |",
            f"| **Claim Unigram Recall in Evidence** | {overall['claim_unigram_recall_in_evidence']['mean']} ± {overall['claim_unigram_recall_in_evidence']['std']} | {overall['claim_unigram_recall_in_evidence']['median']} | {overall['claim_unigram_recall_in_evidence']['min']} - {overall['claim_unigram_recall_in_evidence']['max']} | {overall['claim_unigram_recall_in_evidence']['p95']} |",
            "",
            "---",
            "",
            "## 4. Vocabulary Density",
            "",
            f"- **Total Ingested Tokens:** {overall['vocabulary_metrics']['total_tokens']}",
            f"- **Unique Vocabulary Size:** {overall['vocabulary_metrics']['unique_vocabulary']}",
            f"- **Type-Token Ratio (TTR):** {overall['vocabulary_metrics']['type_token_ratio']}",
            "",
        ]
        return "\n".join(lines)
