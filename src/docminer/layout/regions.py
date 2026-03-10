"""Region detection — reading order, column detection, margin analysis."""

from __future__ import annotations

import logging
from typing import Sequence

from docminer.core.types import BoundingBox, TextBlock

logger = logging.getLogger(__name__)

# Fraction of page width below which a block is considered marginal
_MARGIN_FRACTION = 0.1


class RegionDetector:
    """Analyse spatial structure of page blocks to determine layout regions.

    Detects:
    - Number of text columns
    - Reading order
    - Header / footer zones
    - Margin annotations
    """

    def __init__(
        self,
        column_gap_threshold: float = 0.05,
        header_zone_fraction: float = 0.1,
        footer_zone_fraction: float = 0.1,
    ) -> None:
        self.column_gap_threshold = column_gap_threshold
        self.header_zone_fraction = header_zone_fraction
        self.footer_zone_fraction = footer_zone_fraction

    def detect_columns(
        self,
        blocks: Sequence[TextBlock],
        page_width: float,
    ) -> int:
        """Estimate the number of text columns on the page.

        Uses k-means style clustering of block x-centre coordinates.

        Returns
        -------
        int
            Detected number of columns (1, 2, or 3).
        """
        if not blocks:
            return 1

        cx_values = [b.bbox.center[0] for b in blocks if b.bbox is not None]
        if not cx_values:
            return 1

        # Detect clear gaps in the x-centre distribution
        cx_sorted = sorted(cx_values)
        gaps: list[float] = []
        for i in range(1, len(cx_sorted)):
            gaps.append(cx_sorted[i] - cx_sorted[i - 1])

        if not gaps:
            return 1

        mean_gap = sum(gaps) / len(gaps)
        # A column break is a gap significantly larger than the mean
        large_gaps = [g for g in gaps if g > mean_gap * 3 and g > page_width * 0.05]

        return min(len(large_gaps) + 1, 3)

    def reading_order(
        self,
        blocks: Sequence[TextBlock],
        page_width: float,
        num_columns: int | None = None,
    ) -> list[TextBlock]:
        """Return blocks sorted in natural reading order.

        For multi-column layouts, sorts left column top-to-bottom first,
        then right column.
        """
        if not blocks:
            return []

        if num_columns is None:
            num_columns = self.detect_columns(blocks, page_width)

        if num_columns <= 1:
            return sorted(blocks, key=lambda b: (b.bbox.y0 if b.bbox else 0, 0))

        col_width = page_width / num_columns
        columns: list[list[TextBlock]] = [[] for _ in range(num_columns)]
        for block in blocks:
            if block.bbox is None:
                columns[0].append(block)
                continue
            col_idx = min(int(block.bbox.center[0] / col_width), num_columns - 1)
            columns[col_idx].append(block)

        ordered: list[TextBlock] = []
        for col in columns:
            ordered.extend(sorted(col, key=lambda b: b.bbox.y0 if b.bbox else 0))
        return ordered

    def classify_zones(
        self,
        blocks: Sequence[TextBlock],
        page_height: float,
    ) -> dict[str, list[TextBlock]]:
        """Classify blocks into header, footer, and body zones.

        Returns
        -------
        dict
            Keys: ``"header"``, ``"footer"``, ``"body"``.
        """
        header_limit = page_height * self.header_zone_fraction
        footer_start = page_height * (1.0 - self.footer_zone_fraction)

        zones: dict[str, list[TextBlock]] = {"header": [], "footer": [], "body": []}
        for block in blocks:
            if block.bbox is None:
                zones["body"].append(block)
                continue
            cy = block.bbox.center[1]
            if cy <= header_limit:
                zones["header"].append(block)
            elif cy >= footer_start:
                zones["footer"].append(block)
            else:
                zones["body"].append(block)
        return zones

    def detect_margins(
        self,
        blocks: Sequence[TextBlock],
        page_width: float,
        page_height: float,
    ) -> dict[str, float]:
        """Estimate page margins from the outermost block coordinates.

        Returns
        -------
        dict
            Keys: ``"left"``, ``"right"``, ``"top"``, ``"bottom"``
            (all as fractions of page dimension).
        """
        boxed = [b for b in blocks if b.bbox is not None]
        if not boxed:
            return {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0}

        min_x = min(b.bbox.x0 for b in boxed) / page_width  # type: ignore[union-attr]
        max_x = max(b.bbox.x1 for b in boxed) / page_width  # type: ignore[union-attr]
        min_y = min(b.bbox.y0 for b in boxed) / page_height  # type: ignore[union-attr]
        max_y = max(b.bbox.y1 for b in boxed) / page_height  # type: ignore[union-attr]

        return {
            "left": min_x,
            "right": 1.0 - max_x,
            "top": min_y,
            "bottom": 1.0 - max_y,
        }
