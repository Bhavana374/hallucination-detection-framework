"""Executable script to generate benchmark sample dataset, run EDA, and export validation reports."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.sample_generator import generate_benchmark_dataset
from src.data.loader import load_split_from_jsonl
from src.data.explorer import DatasetExplorer
from src.data.visualizer import DatasetVisualizer
from src.utils.logger import setup_logger

logger = setup_logger("explore_dataset_script")


def main():
    logger.info("Starting Phase 3: Dataset Exploration & Validation...")

    # Step 1: Generate or load benchmark data
    logger.info("Generating / verifying benchmark sample dataset...")
    gen_summary = generate_benchmark_dataset(
        output_raw_file="data/raw/halueval_sample.jsonl",
        output_split_dir="data/processed",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )
    logger.info(f"Generated dataset summary: {gen_summary}")

    # Step 2: Load processed split
    split = load_split_from_jsonl("data/processed")
    logger.info(f"Loaded split -> Train: {split.train_size}, Val: {split.val_size}, Test: {split.test_size}")

    # Step 3: Run Explorer & generate validation reports
    explorer = DatasetExplorer(split)
    md_path, json_path = explorer.save_validation_reports(
        output_table_dir="results/tables",
        output_metrics_dir="results/metrics",
    )
    logger.info(f"Validation reports generated successfully:\n - Markdown: {md_path}\n - JSON: {json_path}")

    # Step 4: Generate exploratory figures
    visualizer = DatasetVisualizer(split, output_dir="results/figures")
    fig_paths = visualizer.generate_all_plots()
    if fig_paths:
        logger.info(f"Generated {len(fig_paths)} visualization figures in results/figures/")
    else:
        logger.info("Visualizer completed (Graphical plotting skipped if matplotlib is uninstalled).")

    print("\n" + "=" * 60)
    print("PHASE 3: DATASET EXPLORATION & VALIDATION COMPLETED")
    print("=" * 60)
    print(f"Total Ingested Items: {split.total_size}")
    print(f"Train Partitions: {split.train_size} | Val: {split.val_size} | Test: {split.test_size}")
    print(f"Validation Report: {md_path}")
    print(f"Metrics JSON:      {json_path}")


if __name__ == "__main__":
    main()
