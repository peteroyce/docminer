"""Text utility helpers — sentence splitting, paragraph detection, statistics."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Sequence


def split_sentences(text: str) -> list[str]:
    """Split *text* into sentences using punctuation heuristics.

    Handles common abbreviations (Mr., Dr., etc.) to avoid false splits.
    """
    # Protect common abbreviations
    abbreviations = [
        "Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr", "Rev",
        "e.g", "i.e", "etc", "vs", "fig", "no", "vol", "pp",
        "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    protected = text
    for abbr in abbreviations:
        protected = protected.replace(f"{abbr}.", f"{abbr}<DOT>")

    # Split on sentence-ending punctuation followed by whitespace and capital
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\'])", protected)

    # Restore abbreviation dots
    sentences = [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]
    return sentences


def split_paragraphs(text: str) -> list[str]:
    """Split *text* into paragraphs separated by one or more blank lines."""
    blocks = re.split(r"\n\s*\n", text)
    return [b.strip() for b in blocks if b.strip()]


def detect_language(text: str) -> str:
    """Very simple heuristic language detection.

    Returns ``"en"`` for English-looking text or ``"unknown"`` otherwise.
    Only designed for basic English detection.
    """
    text_lower = text.lower()
    # Common English function words
    en_markers = {"the", "and", "is", "in", "of", "to", "a", "that", "for", "it"}
    words = set(re.findall(r"\b[a-z]{2,}\b", text_lower))
    overlap = len(words & en_markers)
    if overlap >= 3:
        return "en"
    return "unknown"


def word_frequency(text: str, top_n: int = 20) -> list[tuple[str, int]]:
    """Return the most common words in *text*.

    Stopwords are not filtered.
    """
    words = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
    counter = Counter(words)
    return counter.most_common(top_n)


def flesch_reading_ease(text: str) -> float:
    """Compute the Flesch Reading Ease score for *text*.

    Higher scores indicate easier readability (0–100 scale).
    """
    sentences = split_sentences(text)
    words = text.split()
    num_sentences = max(len(sentences), 1)
    num_words = max(len(words), 1)
    num_syllables = sum(_count_syllables(w) for w in words)

    score = (
        206.835
        - 1.015 * (num_words / num_sentences)
        - 84.6 * (num_syllables / num_words)
    )
    return round(max(0.0, min(100.0, score)), 2)


def _count_syllables(word: str) -> int:
    """Estimate syllable count for an English word."""
    word = word.lower().strip(".,!?;:")
    if not word:
        return 0
    # Count vowel groups
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    # Silent 'e' at end
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def text_statistics(text: str) -> dict:
    """Return a dictionary of statistics for *text*.

    Keys: ``char_count``, ``word_count``, ``sentence_count``,
    ``paragraph_count``, ``avg_word_length``, ``avg_sentence_length``,
    ``reading_ease``.
    """
    sentences = split_sentences(text)
    paragraphs = split_paragraphs(text)
    words = text.split()

    avg_word_len = (
        sum(len(re.sub(r"[^\w]", "", w)) for w in words) / len(words)
        if words else 0.0
    )
    avg_sent_len = len(words) / max(len(sentences), 1)

    return {
        "char_count": len(text),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "paragraph_count": len(paragraphs),
        "avg_word_length": round(avg_word_len, 2),
        "avg_sentence_length": round(avg_sent_len, 2),
        "reading_ease": flesch_reading_ease(text),
    }


def truncate_text(text: str, max_words: int = 200, ellipsis: str = "...") -> str:
    """Truncate *text* to at most *max_words* words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + ellipsis


def normalise_whitespace(text: str) -> str:
    """Collapse all whitespace sequences to a single space."""
    return re.sub(r"\s+", " ", text).strip()


def remove_headers_footers(
    pages_text: Sequence[str],
    min_occurrences: int = 3,
) -> list[str]:
    """Remove repeated header/footer lines from a list of page texts.

    A line is considered a header or footer if it appears in the same
    relative position (first/last N lines) on at least *min_occurrences* pages.
    """
    if len(pages_text) < min_occurrences:
        return list(pages_text)

    def top_lines(page_text: str, n: int = 2) -> list[str]:
        return [line.strip() for line in page_text.splitlines()[:n] if line.strip()]

    def bottom_lines(page_text: str, n: int = 2) -> list[str]:
        return [line.strip() for line in page_text.splitlines()[-n:] if line.strip()]

    # Collect candidate header/footer lines
    header_counter: Counter[str] = Counter()
    footer_counter: Counter[str] = Counter()
    for pt in pages_text:
        for line in top_lines(pt):
            header_counter[line] += 1
        for line in bottom_lines(pt):
            footer_counter[line] += 1

    headers = {line for line, count in header_counter.items() if count >= min_occurrences}
    footers = {line for line, count in footer_counter.items() if count >= min_occurrences}
    remove_lines = headers | footers

    cleaned: list[str] = []
    for pt in pages_text:
        lines = [
            line for line in pt.splitlines()
            if line.strip() not in remove_lines
        ]
        cleaned.append("\n".join(lines))
    return cleaned
