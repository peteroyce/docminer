"""Layout analyzer — classifies TextBlocks by semantic role."""

from __future__ import annotations

import logging
import re
from typing import Sequence

from docminer.core.types import Page, TextBlock
from docminer.layout.regions import RegionDetector

logger = logging.getLogger(__name__)

# Patterns for detecting list items
_BULLET_RE = re.compile(r"^[\u2022\u2023\u25e6\u2043\u2219\-\*\+]\s+")
_NUMBERED_RE = re.compile(r"^(\d+[\.\)]\s+|[a-zA-Z][\.\)]\s+)")

# Footer patterns (page numbers, copyright lines, etc.)
_FOOTER_RE = re.compile(
    r"(page\s+\d+|\bpg\.\s*\d+|\d+\s*/\s*\d+|copyright|©|\bconfidential\b)",
    re.IGNORECASE,
)

# Typical font-size thresholds (points)
_TITLE_FONT_THRESHOLD = 16.0
_HEADER_FONT_THRESHOLD = 12.0
_BODY_FONT_SIZE_MIN = 8.0

# Maximum word count for a line to be considered a heading
_HEADER_MAX_WORDS = 15


class LayoutAnalyzer:
    """Classify the semantic role of each :class:`TextBlock` on a page.

    Block types assigned: ``"header"``, ``"paragraph"``, ``"list_item"``,
    ``"caption"``, ``"footer"``, ``"title"``.
    """

    def __init__(self) -> None:
        self.region_detector = RegionDetector()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_page(self, page: Page) -> None:
        """Classify all blocks on *page* in place and apply reading order."""
        if not page.blocks:
            return

        # First-pass: per-block classification using visual features
        dominant_font = self._dominant_font_size(page.blocks)
        zones = self.region_detector.classify_zones(page.blocks, page.height)

        for block in page.blocks:
            block.block_type = self._classify_block(
                block=block,
                dominant_font=dominant_font,
                is_header_zone=block in zones["header"],
                is_footer_zone=block in zones["footer"],
            )

        # Re-order blocks into reading order
        num_cols = self.region_detector.detect_columns(page.blocks, page.width)
        page.blocks = self.region_detector.reading_order(page.blocks, page.width, num_cols)
        logger.debug(
            "Page %d: %d blocks, %d columns detected", page.number, len(page.blocks), num_cols
        )

    def classify_block(self, block: TextBlock, dominant_font: float = 11.0) -> str:
        """Public single-block classification (without zone context)."""
        return self._classify_block(block, dominant_font)

    # ------------------------------------------------------------------
    # Internal classification
    # ------------------------------------------------------------------

    def _classify_block(
        self,
        block: TextBlock,
        dominant_font: float = 11.0,
        is_header_zone: bool = False,
        is_footer_zone: bool = False,
    ) -> str:
        text = block.text.strip()
        if not text:
            return block.block_type

        word_count = len(text.split())

        # Footer detection (zone + pattern)
        if is_footer_zone or _FOOTER_RE.search(text):
            return "footer"

        # List item detection
        if _BULLET_RE.match(text) or _NUMBERED_RE.match(text):
            return "list_item"

        # Caption detection (short block below an image/figure keyword)
        if re.match(r"^(figure|fig\.|table|chart|diagram|image)\s*[\d:.]", text, re.IGNORECASE):
            return "caption"

        # Header / title detection via font size
        if block.font_size >= _TITLE_FONT_THRESHOLD and word_count <= _HEADER_MAX_WORDS:
            return "title" if block.font_size > dominant_font * 1.5 else "header"

        if block.font_size >= _HEADER_FONT_THRESHOLD and word_count <= _HEADER_MAX_WORDS:
            return "header"

        # Header zone with short text
        if is_header_zone and word_count <= _HEADER_MAX_WORDS:
            return "header"

        # Short single-line all-caps or bold-looking text
        if word_count <= _HEADER_MAX_WORDS and text.isupper() and word_count <= 8:
            return "header"

        return "paragraph"

    # ------------------------------------------------------------------
    # Font analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _dominant_font_size(blocks: Sequence[TextBlock]) -> float:
        """Return the most common font size across all blocks (mode by character count)."""
        size_counts: dict[float, int] = {}
        for block in blocks:
            if block.font_size > 0:
                rounded = round(block.font_size, 1)
                size_counts[rounded] = size_counts.get(rounded, 0) + len(block.text)
        if not size_counts:
            return 11.0
        return max(size_counts, key=size_counts.get)  # type: ignore[arg-type]
