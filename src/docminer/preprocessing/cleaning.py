"""Text cleaning and normalisation utilities."""

from __future__ import annotations

import re
import unicodedata


class TextCleaner:
    """Clean and normalise raw text extracted from documents.

    Operations applied (in order):
    1. Unicode NFKD normalisation + ASCII transliteration
    2. Fix common OCR substitution errors
    3. Collapse whitespace and fix line breaks
    4. Remove null bytes and control characters
    5. Strip leading/trailing whitespace per line

    Usage::

        cleaner = TextCleaner()
        clean = cleaner.clean("  Invoice\\n\\n\\rDate:  01/01/2024  ")
    """

    # Common OCR mis-reads: (wrong, correct)
    _OCR_FIXES: list[tuple[str, str]] = [
        (r"\brn\b", "m"),       # 'rn' misread as 'm'
        (r"\bl\b(?=\d)", "1"),  # lowercase l before digit = 1
        (r"\bO\b(?=\d)", "0"),  # uppercase O before digit = 0
        (r"\|", "I"),           # pipe misread as capital I
    ]

    # Ligatures and common Unicode replacements
    _LIGATURE_MAP: dict[str, str] = {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\u2013": "-",   # en dash
        "\u2014": "--",  # em dash
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2022": "*",   # bullet
        "\u00a0": " ",   # non-breaking space
        "\u00ad": "",    # soft hyphen
    }

    def __init__(self, fix_ocr: bool = True, normalize_unicode: bool = True) -> None:
        self.fix_ocr = fix_ocr
        self.normalize_unicode = normalize_unicode
        self._ocr_patterns = [
            (re.compile(pattern), replacement)
            for pattern, replacement in self._OCR_FIXES
        ]

    def clean(self, text: str) -> str:
        """Return a cleaned version of *text*."""
        if not text:
            return text

        # Step 1: Replace ligatures / special chars
        text = self._replace_ligatures(text)

        # Step 2: Unicode normalisation
        if self.normalize_unicode:
            text = unicodedata.normalize("NFKD", text)
            text = text.encode("ascii", "ignore").decode("ascii")

        # Step 3: Remove null bytes and non-printable control characters
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # Step 4: Fix OCR errors
        if self.fix_ocr:
            for pat, repl in self._ocr_patterns:
                text = pat.sub(repl, text)

        # Step 5: Normalise whitespace within lines
        lines = text.splitlines()
        cleaned_lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]

        # Step 6: Collapse excessive blank lines (max 2 consecutive)
        result_lines: list[str] = []
        blank_count = 0
        for line in cleaned_lines:
            if not line:
                blank_count += 1
                if blank_count <= 2:
                    result_lines.append("")
            else:
                blank_count = 0
                result_lines.append(line)

        return "\n".join(result_lines).strip()

    def clean_block(self, text: str) -> str:
        """Clean a single text block (minimal — no aggressive OCR fixing)."""
        if not text:
            return text
        text = self._replace_ligatures(text)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def _replace_ligatures(self, text: str) -> str:
        for orig, repl in self._LIGATURE_MAP.items():
            text = text.replace(orig, repl)
        return text

    @staticmethod
    def extract_sentences(text: str) -> list[str]:
        """Split *text* into sentences using simple punctuation rules."""
        text = re.sub(r"\s+", " ", text.strip())
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def word_count(text: str) -> int:
        """Return the number of words in *text*."""
        return len(text.split())

    @staticmethod
    def char_count(text: str, exclude_whitespace: bool = True) -> int:
        """Return character count, optionally excluding whitespace."""
        if exclude_whitespace:
            return len(re.sub(r"\s", "", text))
        return len(text)

    @staticmethod
    def average_word_length(text: str) -> float:
        """Return the average word length in *text*."""
        words = text.split()
        if not words:
            return 0.0
        return sum(len(w) for w in words) / len(words)
