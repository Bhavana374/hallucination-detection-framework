"""Modular text cleaning and normalization routines preserving Transformer semantics."""

import html
import re
import unicodedata
from typing import Optional


class TextCleaner:
    """Configurable text cleaning utility.

    Preserves grammatical punctuation, casing, and semantic indicators required
    by contextual Transformer models while eliminating HTML noise and erratic whitespace.
    """

    def __init__(
        self,
        preserve_case_for_transformers: bool = True,
        strip_html_tags: bool = True,
        normalize_unicode: bool = True,
        normalize_whitespace: bool = True,
    ):
        self.preserve_case = preserve_case_for_transformers
        self.strip_html = strip_html_tags
        self.normalize_unicode = normalize_unicode
        self.normalize_ws = normalize_whitespace

        self._html_pattern = re.compile(r"<[^>]+>")
        self._ws_pattern = re.compile(r"\s+")

    def clean(self, text: Optional[str]) -> str:
        """Clean and normalize a text string.

        Args:
            text: Raw input text.

        Returns:
            Normalized clean string.
        """
        if text is None:
            return ""

        cleaned = str(text)

        # 1. Strip HTML tags and decode HTML entities
        if self.strip_html:
            cleaned = html.unescape(cleaned)
            cleaned = self._html_pattern.sub(" ", cleaned)

        # 2. Unicode normalization (NFKC decomposes special formatting, preserving standard glyphs)
        if self.normalize_unicode:
            cleaned = unicodedata.normalize("NFKC", cleaned)

        # 3. Standardize quotes and hyphens without stripping semantic punctuation
        cleaned = cleaned.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
        cleaned = cleaned.replace("—", " - ").replace("–", " - ")

        # 4. Collapse erratic whitespace
        if self.normalize_ws:
            cleaned = self._ws_pattern.sub(" ", cleaned).strip()

        # 5. Casing adjustment
        if not self.preserve_case:
            cleaned = cleaned.lower()

        return cleaned
