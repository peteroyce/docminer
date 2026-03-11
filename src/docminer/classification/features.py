"""Feature extraction for document classification."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docminer.core.types import Document


class FeatureExtractor:
    """Extract numeric and keyword features from a :class:`Document` for classification."""

    # ------------------------------------------------------------------
    # Top-level feature vector
    # ------------------------------------------------------------------

    def extract(self, document: "Document") -> dict:
        """Return a flat dictionary of named features.

        Features are grouped into:
        - ``keyword_*`` — per-type keyword hit counts
        - ``struct_*``  — structural indicators
        - ``meta_*``    — metadata-derived features
        """
        text = document.text.lower()
        features: dict = {}
        features.update(self._keyword_features(text))
        features.update(self._structural_features(document))
        features.update(self._metadata_features(document))
        return features

    # ------------------------------------------------------------------
    # Keyword density features
    # ------------------------------------------------------------------

    @staticmethod
    def _keyword_features(text_lower: str) -> dict[str, float]:
        """Count keyword hits per document type category."""
        from docminer.classification.labels import KEYWORD_SIGNATURES

        features: dict[str, float] = {}
        word_count = max(len(text_lower.split()), 1)

        for doc_type, keywords in KEYWORD_SIGNATURES.items():
            hits = sum(1 for kw in keywords if kw in text_lower)
            features[f"keyword_{doc_type}_hits"] = float(hits)
            features[f"keyword_{doc_type}_density"] = hits / word_count

        return features

    # ------------------------------------------------------------------
    # Structural features
    # ------------------------------------------------------------------

    @staticmethod
    def _structural_features(document: "Document") -> dict[str, float]:
        """Extract structural signals from page layout."""
        page_count = document.page_count
        all_blocks = document.all_blocks
        all_tables = document.all_tables
        text = document.text

        # Block type counts
        headers = [b for b in all_blocks if b.block_type in ("header", "title")]
        list_items = [b for b in all_blocks if b.block_type == "list_item"]
        paragraphs = [b for b in all_blocks if b.block_type == "paragraph"]

        # Numeric presence
        amount_re = re.compile(r"\$[\d,]+\.?\d*|\d+\.\d{2}")
        date_re = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
        blank_re = re.compile(r"_____+")
        checkbox_re = re.compile(r"\[\s*\]|\(\s*\)")

        return {
            "struct_page_count": float(page_count),
            "struct_multi_page": float(page_count > 1),
            "struct_single_page": float(page_count == 1),
            "struct_has_tables": float(len(all_tables) > 0),
            "struct_table_count": float(len(all_tables)),
            "struct_has_headers": float(len(headers) > 0),
            "struct_header_count": float(len(headers)),
            "struct_has_bullet_lists": float(len(list_items) > 0),
            "struct_list_item_count": float(len(list_items)),
            "struct_paragraph_count": float(len(paragraphs)),
            "struct_has_numbers": float(bool(amount_re.search(text))),
            "struct_amount_count": float(len(amount_re.findall(text))),
            "struct_has_dates": float(bool(date_re.search(text))),
            "struct_date_count": float(len(date_re.findall(text))),
            "struct_has_blanks": float(bool(blank_re.search(text))),
            "struct_has_checkboxes": float(bool(checkbox_re.search(text))),
            "struct_avg_block_words": (
                sum(b.word_count() for b in all_blocks) / len(all_blocks)
                if all_blocks
                else 0.0
            ),
            "struct_total_words": float(len(text.split())),
        }

    # ------------------------------------------------------------------
    # Metadata features
    # ------------------------------------------------------------------

    @staticmethod
    def _metadata_features(document: "Document") -> dict[str, float]:
        """Extract classification signals from document metadata."""
        meta = document.metadata
        title = (meta.get("title") or "").lower()
        keywords_field = (meta.get("keywords") or "").lower()

        type_keywords = {
            "invoice": ["invoice", "bill", "receipt"],
            "contract": ["contract", "agreement", "legal"],
            "resume": ["resume", "cv", "curriculum"],
            "report": ["report", "analysis", "study"],
            "letter": ["letter", "correspondence"],
            "form": ["form", "application"],
        }

        features: dict[str, float] = {}
        for doc_type, kws in type_keywords.items():
            hit = any(kw in title or kw in keywords_field for kw in kws)
            features[f"meta_{doc_type}_title"] = float(hit)

        return features
