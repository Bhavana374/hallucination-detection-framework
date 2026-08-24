"""Unit tests for text cleaning, deduplication, and preprocessing pipeline."""

from src.data.schema import ClaimEvidenceItem, DatasetSplit
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.deduplicator import ClaimDeduplicator, compute_ngram_jaccard
from src.preprocessing.pipeline import DatasetPreprocessor


def test_text_cleaner_html_and_whitespace():
    """Verify HTML stripping, entity unescaping, and whitespace collapsing."""
    cleaner = TextCleaner(preserve_case_for_transformers=True, strip_html_tags=True)
    raw_text = "  <p>The <b>Eiffel Tower</b> is in &quot;Paris&quot;, France.&nbsp;&nbsp;</p>  "
    cleaned = cleaner.clean(raw_text)

    assert "<p>" not in cleaned
    assert "<b>" not in cleaned
    assert '"Paris"' in cleaned
    assert "The Eiffel Tower is in \"Paris\", France." == cleaned


def test_text_cleaner_case_preservation():
    """Verify casing is preserved for Transformer NER and semantics."""
    cleaner = TextCleaner(preserve_case_for_transformers=True)
    raw_text = "Albert Einstein won the Nobel Prize in Physics."
    cleaned = cleaner.clean(raw_text)

    assert "Einstein" in cleaned
    assert "Nobel Prize" in cleaned


def test_deduplicator_exact_and_near_duplicates():
    """Verify deduplicator prunes exact and near-duplicate items."""
    dedup = ClaimDeduplicator(near_duplicate_threshold=0.90, ngram_size=3)
    items = [
        ClaimEvidenceItem("c1", "r1", "The Eiffel Tower is located in Paris, France.", "Ev 1", 0, 0),
        ClaimEvidenceItem("c2", "r2", "The Eiffel Tower is located in Paris, France.", "Ev 2", 0, 0),  # Exact duplicate
        ClaimEvidenceItem("c3", "r3", "The Eiffel Tower is located in Paris France", "Ev 3", 0, 0),    # Near duplicate
        ClaimEvidenceItem("c4", "r4", "Albert Einstein was born in Ulm, Germany.", "Ev 4", 0, 0),      # Unique
    ]

    unique_items, audit = dedup.deduplicate(items)
    assert len(unique_items) == 2
    assert audit["pruned_exact_duplicates"] == 1
    assert audit["pruned_near_duplicates"] == 1
    assert unique_items[0].claim_id == "c1"
    assert unique_items[1].claim_id == "c4"


def test_preprocessing_pipeline():
    """Verify end-to-end dataset preprocessing on a split."""
    preprocessor = DatasetPreprocessor(preserve_case=True, min_claim_chars=5, max_claim_chars=500)

    train_items = [
        ClaimEvidenceItem("c1", "r1", "<p>Water boils at 100C.</p>", "Ev 1", 0, 0),
        ClaimEvidenceItem("c2", "r2", "Bad", "Ev 2", 1, 1),  # Too short (len < 5) -> should be dropped
    ]
    val_items = [
        ClaimEvidenceItem("c3", "r3", "Mars is the red planet.", "Ev 3", 0, 0),
    ]
    test_items = [
        ClaimEvidenceItem("c4", "r4", "Jupiter is a gas giant.", "Ev 4", 0, 0),
    ]

    split = DatasetSplit(train=train_items, val=val_items, test=test_items)
    clean_split, report = preprocessor.preprocess_split(split)

    assert clean_split.train_size == 1
    assert clean_split.val_size == 1
    assert clean_split.test_size == 1
    assert "<p>" not in clean_split.train[0].claim_text
    assert report["leakage_audit"]["has_leakage"] is False
