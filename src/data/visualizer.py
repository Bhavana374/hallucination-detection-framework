"""Visualization engine for dataset distributions, sequence lengths, and class balance."""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from src.data.schema import DatasetSplit, ClaimEvidenceItem
from src.data.explorer import simple_tokenize, compute_jaccard_overlap, compute_unigram_recall
from src.utils.logger import setup_logger

logger = setup_logger("data_visualizer")

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class DatasetVisualizer:
    """Generate publication-ready visualizations for dataset distributions."""

    def __init__(self, split: DatasetSplit, output_dir: Union[str, Path] = "results/figures"):
        self.split = split
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_class_distribution(self, save_name: str = "class_distribution.png") -> Optional[str]:
        """Plot class balance across train, val, and test splits."""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib is not installed. Skipping graphical chart generation.")
            return None

        out_path = self.output_dir / save_name
        fig, ax = plt.subplots(figsize=(8, 5))

        splits = ["Train", "Validation", "Test"]
        factual_counts = [
            sum(1 for it in self.split.train if it.binary_label == 0),
            sum(1 for it in self.split.val if it.binary_label == 0),
            sum(1 for it in self.split.test if it.binary_label == 0),
        ]
        hallucinated_counts = [
            sum(1 for it in self.split.train if it.binary_label == 1),
            sum(1 for it in self.split.val if it.binary_label == 1),
            sum(1 for it in self.split.test if it.binary_label == 1),
        ]

        x = range(len(splits))
        width = 0.35

        ax.bar([i - width/2 for i in x], factual_counts, width, label="Factual (0)", color="#2b8a3e", edgecolor="black")
        ax.bar([i + width/2 for i in x], hallucinated_counts, width, label="Hallucinated (1)", color="#c92a2a", edgecolor="black")

        ax.set_ylabel("Number of Samples", fontsize=12)
        ax.set_title("Class Distribution Across Dataset Partitions", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(splits, fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close(fig)

        logger.info(f"Saved class distribution figure to: {out_path}")
        return str(out_path)

    def plot_token_length_distributions(self, save_name: str = "token_length_distribution.png") -> Optional[str]:
        """Plot histogram of token lengths for claims and evidence."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        all_items = self.split.train + self.split.val + self.split.test
        claim_lens = [len(simple_tokenize(it.claim_text)) for it in all_items]
        evidence_lens = [len(simple_tokenize(it.evidence_text)) for it in all_items]

        out_path = self.output_dir / save_name
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.hist(claim_lens, bins=20, color="#1971c2", edgecolor="black", alpha=0.75)
        ax1.set_title("Claim Token Length Distribution", fontsize=13, fontweight="bold")
        ax1.set_xlabel("Token Count", fontsize=11)
        ax1.set_ylabel("Frequency", fontsize=11)
        ax1.grid(axis="y", linestyle="--", alpha=0.7)

        ax2.hist(evidence_lens, bins=20, color="#d9480f", edgecolor="black", alpha=0.75)
        ax2.set_title("Evidence Token Length Distribution", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Token Count", fontsize=11)
        ax2.set_ylabel("Frequency", fontsize=11)
        ax2.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close(fig)

        logger.info(f"Saved token length distribution figure to: {out_path}")
        return str(out_path)

    def plot_lexical_overlap(self, save_name: str = "lexical_overlap_distribution.png") -> Optional[str]:
        """Plot distribution of lexical overlap metrics between claims and evidence."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        all_items = self.split.train + self.split.val + self.split.test
        jaccard = [compute_jaccard_overlap(it.claim_text, it.evidence_text) for it in all_items]
        recall = [compute_unigram_recall(it.claim_text, it.evidence_text) for it in all_items]

        out_path = self.output_dir / save_name
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.hist(jaccard, bins=15, alpha=0.6, label="Jaccard Overlap", color="#4263eb", edgecolor="black")
        ax.hist(recall, bins=15, alpha=0.6, label="Claim Unigram Recall in Evidence", color="#0ca678", edgecolor="black")

        ax.set_title("Lexical Grounding & Overlap Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Overlap Ratio [0.0 - 1.0]", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close(fig)

        logger.info(f"Saved lexical overlap figure to: {out_path}")
        return str(out_path)

    def generate_all_plots(self) -> List[str]:
        """Generate and save all exploratory dataset figures."""
        paths: List[str] = []
        p1 = self.plot_class_distribution()
        p2 = self.plot_token_length_distributions()
        p3 = self.plot_lexical_overlap()
        for p in (p1, p2, p3):
            if p:
                paths.append(p)
        return paths
