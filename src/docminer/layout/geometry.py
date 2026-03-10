"""BoundingBox geometry utilities."""

from __future__ import annotations

import math
from typing import Sequence

from docminer.core.types import BoundingBox


class BoundingBoxOps:
    """Static utility methods for collections of BoundingBox objects."""

    @staticmethod
    def merge_all(boxes: Sequence[BoundingBox]) -> BoundingBox:
        """Return the minimum bounding box containing all *boxes*."""
        if not boxes:
            raise ValueError("Cannot merge empty sequence of bounding boxes")
        x0 = min(b.x0 for b in boxes)
        y0 = min(b.y0 for b in boxes)
        x1 = max(b.x1 for b in boxes)
        y1 = max(b.y1 for b in boxes)
        return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)

    @staticmethod
    def cluster_by_proximity(
        boxes: Sequence[BoundingBox],
        threshold: float,
    ) -> list[list[BoundingBox]]:
        """Group boxes into clusters where adjacent members are within *threshold* distance."""
        if not boxes:
            return []
        clusters: list[list[BoundingBox]] = []
        remaining = list(boxes)
        while remaining:
            seed = remaining.pop(0)
            cluster = [seed]
            i = 0
            while i < len(remaining):
                if any(seed.distance_to(remaining[i]) <= threshold for seed in cluster):
                    cluster.append(remaining.pop(i))
                else:
                    i += 1
            clusters.append(cluster)
        return clusters

    @staticmethod
    def non_maximum_suppression(
        boxes: Sequence[BoundingBox],
        iou_threshold: float = 0.5,
    ) -> list[BoundingBox]:
        """Remove highly overlapping boxes, keeping the larger one."""
        sorted_boxes = sorted(boxes, key=lambda b: b.area, reverse=True)
        kept: list[BoundingBox] = []
        for box in sorted_boxes:
            if all(box.iou(k) < iou_threshold for k in kept):
                kept.append(box)
        return kept

    @staticmethod
    def reading_order(boxes: Sequence[BoundingBox]) -> list[BoundingBox]:
        """Sort boxes in approximate reading order (top-to-bottom, left-to-right)."""
        return sorted(boxes, key=lambda b: (round(b.y0 / 20) * 20, b.x0))

    @staticmethod
    def horizontal_overlap(a: BoundingBox, b: BoundingBox) -> float:
        """Fraction of overlap in the X dimension."""
        overlap_x = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
        min_width = min(a.width, b.width)
        return overlap_x / min_width if min_width > 0 else 0.0

    @staticmethod
    def vertical_gap(a: BoundingBox, b: BoundingBox) -> float:
        """Vertical gap between two boxes (positive if a is above b)."""
        return b.y0 - a.y1

    @staticmethod
    def point_in_box(x: float, y: float, box: BoundingBox) -> bool:
        """Check whether a point lies within *box*."""
        return box.x0 <= x <= box.x1 and box.y0 <= y <= box.y1

    @staticmethod
    def relative_position(a: BoundingBox, b: BoundingBox) -> str:
        """Describe the position of *a* relative to *b*.

        Returns one of: ``"above"``, ``"below"``, ``"left"``, ``"right"``,
        ``"overlapping"``.
        """
        if a.iou(b) > 0:
            return "overlapping"
        cx_a, cy_a = a.center
        cx_b, cy_b = b.center
        dx = cx_a - cx_b
        dy = cy_a - cy_b
        if abs(dy) >= abs(dx):
            return "above" if dy < 0 else "below"
        return "left" if dx < 0 else "right"

    @staticmethod
    def aspect_ratio(box: BoundingBox) -> float:
        """Width-to-height ratio; returns 0 if height is zero."""
        return box.width / box.height if box.height > 0 else 0.0

    @staticmethod
    def grid_position(
        box: BoundingBox,
        page_width: float,
        page_height: float,
    ) -> tuple[str, str]:
        """Classify box into a 3x3 grid cell on the page.

        Returns (col, row) where col in {left, center, right},
        row in {top, middle, bottom}.
        """
        cx, cy = box.center
        col_idx = int(cx / page_width * 3)
        row_idx = int(cy / page_height * 3)
        col = ["left", "center", "right"][min(col_idx, 2)]
        row = ["top", "middle", "bottom"][min(row_idx, 2)]
        return col, row
