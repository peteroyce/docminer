"""Table extractor — detects and extracts tables from PDF pages."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from docminer.core.types import BoundingBox, Document, Table
from docminer.extraction.base import BaseExtractor

logger = logging.getLogger(__name__)

# Minimum number of cells to consider a region a table
_MIN_CELLS = 4
# Snap tolerance for aligning grid lines (points)
_SNAP_TOL = 3.0


class TableExtractor(BaseExtractor):
    """Detect and extract tables from PDF documents.

    Strategy:
    1. Detect horizontal and vertical ruling lines (vector graphics).
    2. Build a grid from line intersections.
    3. Extract cell text by position.
    4. Fall back to whitespace-based column detection for borderless tables.
    """

    def extract_document(self, path: str | Path) -> Document:
        """Extract tables from a PDF and return a Document with tables populated."""
        import fitz

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        doc_id = self.make_document_id(path)
        pdf = fitz.open(str(path))
        from docminer.core.types import Page

        pages = []
        try:
            for i in range(len(pdf)):
                fitz_page = pdf[i]
                rect = fitz_page.rect
                tables = self.extract_page_tables(fitz_page)
                page = Page(
                    number=i + 1,
                    width=rect.width,
                    height=rect.height,
                    tables=tables,
                )
                pages.append(page)
        finally:
            pdf.close()

        return Document(
            id=doc_id,
            source_path=str(path),
            file_type="pdf",
            pages=pages,
        )

    # ------------------------------------------------------------------
    # Core table detection
    # ------------------------------------------------------------------

    def extract_page_tables(self, fitz_page) -> list[Table]:
        """Extract all tables from a single PyMuPDF page object."""
        tables: list[Table] = []
        page_num = fitz_page.number + 1

        # Try ruling-line based detection
        ruled = self._extract_ruled_tables(fitz_page, page_num)
        tables.extend(ruled)

        if not ruled:
            # Fallback: whitespace column detection
            text_tables = self._extract_whitespace_tables(fitz_page, page_num)
            tables.extend(text_tables)

        return tables

    # ------------------------------------------------------------------
    # Ruled-line table extraction
    # ------------------------------------------------------------------

    def _extract_ruled_tables(self, fitz_page, page_num: int) -> list[Table]:
        """Detect tables defined by horizontal/vertical line segments."""
        h_lines, v_lines = self._get_ruling_lines(fitz_page)
        if len(h_lines) < 2 or len(v_lines) < 2:
            return []

        # Snap lines to a grid
        h_sorted = sorted(set(self._snap(y, _SNAP_TOL) for y in h_lines))
        v_sorted = sorted(set(self._snap(x, _SNAP_TOL) for x in v_lines))

        if len(h_sorted) < 2 or len(v_sorted) < 2:
            return []

        tables: list[Table] = []
        # Group into rectangular grid regions
        rows: list[list[str]] = []
        for row_idx in range(len(h_sorted) - 1):
            y0 = h_sorted[row_idx]
            y1 = h_sorted[row_idx + 1]
            row: list[str] = []
            for col_idx in range(len(v_sorted) - 1):
                x0 = v_sorted[col_idx]
                x1 = v_sorted[col_idx + 1]
                cell_text = self._get_text_in_rect(fitz_page, x0, y0, x1, y1)
                row.append(cell_text)
            rows.append(row)

        if rows and len(rows) * len(rows[0]) >= _MIN_CELLS:
            bbox = BoundingBox(
                x0=v_sorted[0], y0=h_sorted[0], x1=v_sorted[-1], y1=h_sorted[-1]
            )
            tables.append(
                Table(
                    rows=rows[1:] if len(rows) > 1 else rows,
                    headers=rows[0] if len(rows) > 1 else None,
                    bbox=bbox,
                    page_num=page_num,
                    metadata={"detection_method": "ruled_lines"},
                )
            )
        return tables

    @staticmethod
    def _get_ruling_lines(fitz_page) -> tuple[list[float], list[float]]:
        """Extract unique horizontal and vertical ruling line coordinates."""
        h_lines: list[float] = []
        v_lines: list[float] = []
        try:
            paths = fitz_page.get_drawings()
        except Exception:
            return h_lines, v_lines

        for path in paths:
            for item in path.get("items", []):
                if item[0] == "l":  # line segment
                    p1, p2 = item[1], item[2]
                    dy = abs(p2.y - p1.y)
                    dx = abs(p2.x - p1.x)
                    if dy < _SNAP_TOL and dx > 10:  # horizontal
                        h_lines.append((p1.y + p2.y) / 2)
                    elif dx < _SNAP_TOL and dy > 10:  # vertical
                        v_lines.append((p1.x + p2.x) / 2)
        return h_lines, v_lines

    # ------------------------------------------------------------------
    # Whitespace-based borderless table detection
    # ------------------------------------------------------------------

    def _extract_whitespace_tables(self, fitz_page, page_num: int) -> list[Table]:
        """Detect tabular structures via consistent column alignment in text."""
        words = fitz_page.get_text("words")  # (x0,y0,x1,y1,word,block,line,word_num)
        if not words:
            return []

        # Group words by approximate y-coordinate (rows)
        row_map: dict[int, list[tuple]] = {}
        for w in words:
            y_key = round(w[1] / 5) * 5  # 5-point buckets
            row_map.setdefault(y_key, []).append(w)

        sorted_rows = sorted(row_map.items())
        if len(sorted_rows) < 3:
            return []

        # Detect consistent column x-positions across multiple rows
        col_x_sets: list[list[float]] = []
        for _, row_words in sorted_rows:
            col_x_sets.append(sorted(w[0] for w in row_words))

        # Find rows with consistent multi-column layout (2+ columns with similar spacing)
        table_rows: list[list[str]] = []
        for y_key, row_words in sorted_rows:
            sorted_row_words = sorted(row_words, key=lambda w: w[0])
            if len(sorted_row_words) >= 2:
                table_rows.append([w[4] for w in sorted_row_words])

        if len(table_rows) < 3:
            return []

        # Verify column count is consistent
        col_counts = [len(r) for r in table_rows]
        most_common = max(set(col_counts), key=col_counts.count)
        consistent = [r for r in table_rows if len(r) == most_common]

        if len(consistent) < 3 or most_common < 2:
            return []

        return [
            Table(
                rows=consistent[1:],
                headers=consistent[0],
                page_num=page_num,
                metadata={"detection_method": "whitespace_columns"},
            )
        ]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _get_text_in_rect(fitz_page, x0: float, y0: float, x1: float, y1: float) -> str:
        """Extract and clean text within a rectangular region."""
        import fitz

        rect = fitz.Rect(x0, y0, x1, y1)
        text = fitz_page.get_text("text", clip=rect).strip()
        return " ".join(text.split())  # collapse whitespace

    @staticmethod
    def _snap(value: float, tolerance: float) -> float:
        """Snap a coordinate to a grid determined by tolerance."""
        return round(value / tolerance) * tolerance
