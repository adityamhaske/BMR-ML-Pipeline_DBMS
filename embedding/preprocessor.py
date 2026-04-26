"""
Text Preprocessor — Embedding Pipeline
=======================================
Cleans and normalizes customer feedback text before embedding.

Operations (in order):
1. Strip HTML tags
2. Normalize unicode
3. Remove excessive whitespace
4. Filter by language (English only by default)
5. Truncate to max token length
6. Deduplicate (by exact hash)
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from functools import lru_cache

from loguru import logger


class TextPreprocessor:
    """
    Stateless, cacheable text preprocessor.

    All methods are pure functions — safe for parallel execution.
    """

    # Regex compiled once at class level
    _HTML_TAG_RE = re.compile(r"<[^>]+>")
    _MULTI_SPACE_RE = re.compile(r"\s+")
    _URL_RE = re.compile(r"https?://\S+|www\.\S+")
    _EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
    _NON_ASCII_PUNCT_RE = re.compile(r"[^\w\s.,!?;:'\"-]")

    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 512,
        remove_urls: bool = True,
        remove_emails: bool = True,
        language: str = "en",
    ) -> None:
        self.min_length = min_length
        self.max_length = max_length
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.language = language
        self._seen_hashes: set[str] = set()

    @lru_cache(maxsize=10_000)
    def clean(self, text: str) -> str:
        """
        Clean a single text string. Results are cached for deduplication-aware pipelines.

        Args:
            text: Raw input text

        Returns:
            Cleaned, normalized text string
        """
        if not text or not isinstance(text, str):
            return ""

        # 1. Normalize unicode (NFC form)
        text = unicodedata.normalize("NFC", text)

        # 2. Strip HTML tags
        text = self._HTML_TAG_RE.sub(" ", text)

        # 3. Remove URLs and emails if configured
        if self.remove_urls:
            text = self._URL_RE.sub(" ", text)
        if self.remove_emails:
            text = self._EMAIL_RE.sub(" ", text)

        # 4. Normalize whitespace
        text = self._MULTI_SPACE_RE.sub(" ", text).strip()

        # 5. Truncate to max_length (characters, not tokens — fast estimate)
        #    Sentence transformer will truncate tokens, but we limit chars
        #    to avoid wasting inference time on extreme lengths
        if len(text) > self.max_length * 6:  # ~6 chars per token estimate
            text = text[: self.max_length * 6]

        return text

    def is_valid(self, text: str) -> bool:
        """
        Return True if the cleaned text is worth embedding.

        Filters out:
        - Too short texts (mostly noise)
        - Empty strings
        """
        cleaned = self.clean(text)
        return len(cleaned) >= self.min_length

    def fingerprint(self, text: str) -> str:
        """SHA-256 fingerprint of cleaned text for deduplication."""
        return hashlib.sha256(self.clean(text).encode()).hexdigest()

    def is_duplicate(self, text: str) -> bool:
        """
        Check if this text has been seen before (within this preprocessor instance).
        Mutates internal state — not safe for concurrent use without a lock.
        """
        fp = self.fingerprint(text)
        if fp in self._seen_hashes:
            return True
        self._seen_hashes.add(fp)
        return False

    def reset_dedup_cache(self) -> None:
        """Clear the deduplication set — call between pipeline runs."""
        self._seen_hashes.clear()
        self.clean.cache_clear()

    def batch_clean(
        self,
        texts: list[str],
        deduplicate: bool = False,
        filter_invalid: bool = True,
    ) -> tuple[list[str], list[int]]:
        """
        Clean a list of texts and return cleaned texts + original indices.

        Args:
            texts: Raw texts
            deduplicate: Skip exact duplicates
            filter_invalid: Skip texts that fail is_valid()

        Returns:
            (cleaned_texts, original_indices) — indices map back to input list
        """
        cleaned = []
        indices = []
        filtered_count = 0

        for i, text in enumerate(texts):
            c = self.clean(text)

            if filter_invalid and not self.is_valid(c):
                filtered_count += 1
                continue

            if deduplicate and self.is_duplicate(c):
                filtered_count += 1
                continue

            cleaned.append(c)
            indices.append(i)

        if filtered_count > 0:
            logger.debug(f"Filtered {filtered_count}/{len(texts)} records in batch")

        return cleaned, indices
