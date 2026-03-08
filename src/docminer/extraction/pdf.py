"""PDF extractor using PyMuPDF (fitz)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from docminer.core.types import BoundingBox, Document, Page, TextBlock
from docminer.extraction.base import BaseExtractor

logger = logging.getLogger(__name__)


class PDFExtractor(BaseExtractor):
    """Extract text, layout blocks, and metadata from PDF files using PyMuPDF.

    Handles:
    - Native text PDFs (selectable text)
    - Multi-column layouts
    - Metadata extraction (author, title, creation date, etc.)
    - Per-block font size and name
    """

    def __init__(self, config=None) -> None:
        super().__init__(config=config)
        self._word_flags = 0  # fitz word extraction flags

    def extract_document(self, path: str | Path) -> Document:
        """Extract all pages from a PDF file."""
        import fitz  # PyMuPDF

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        doc_id = self.make_document_id(path)
        logger.info("Extracting PDF: %s (id=%s)", path.name, doc_id)

        pdf = fitz.open(str(path))
        try:
            metadata = self._extract_metadata(pdf)
            pages = [self._extract_page(pdf, i) for i in range(len(pdf))]
        finally:
            pdf.close()

        full_text = "\n\n".join(p.text for p in pages)

        return Document(
            id=doc_id,
            source_path=str(path),
            file_type="pdf",
            pages=pages,
            metadata=metadata,
            text=full_text,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_metadata(pdf) -> dict:
        """Pull standard PDF metadata fields."""
        raw = pdf.metadata or {}
        return {
            "title": raw.get("title", ""),
            "author": raw.get("author", ""),
            "subject": raw.get("subject", ""),
            "keywords": raw.get("keywords", ""),
            "creator": raw.get("creator", ""),
            "producer": raw.get("producer", ""),
            "creation_date": raw.get("creationDate", ""),
            "modification_date": raw.get("modDate", ""),
            "page_count": pdf.page_count,
            "is_encrypted": pdf.is_encrypted,
        }

    def _extract_page(self, pdf, page_idx: int) -> Page:
        """Extract a single PDF page into a :class:`Page` object."""
        import fitz

        fitz_page = pdf[page_idx]
        rect = fitz_page.rect
        page = Page(
            number=page_idx + 1,
            width=rect.width,
            height=rect.height,
        )

        # Get structured text with dict mode (blocks, lines, spans)
        text_dict = fitz_page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

        for block in text_dict.get("blocks", []):
            if block.get("type") == 0:  # text block
                tb = self._process_text_block(block, page_idx)
                if tb and not tb.is_empty():
                    page.blocks.append(tb)
            elif block.get("type") == 1:  # image block
                page.images.append(
                    {
                        "bbox": {
                            "x0": block["bbox"][0],
                            "y0": block["bbox"][1],
                            "x1": block["bbox"][2],
                            "y1": block["bbox"][3],
                        },
                        "page_num": page_idx + 1,
                    }
                )

        return page

    @staticmethod
    def _process_text_block(block: dict, page_idx: int) -> Optional[TextBlock]:
        """Convert a fitz text block dict to a :class:`TextBlock`."""
        bbox_raw = block.get("bbox", (0, 0, 0, 0))
        bbox = BoundingBox(x0=bbox_raw[0], y0=bbox_raw[1], x1=bbox_raw[2], y1=bbox_raw[3])

        lines_text: list[str] = []
        dominant_font_size = 0.0
        dominant_font_name = ""
        max_span_len = 0

        for line in block.get("lines", []):
            line_parts: list[str] = []
            for span in line.get("spans", []):
                span_text = span.get("text", "")
                line_parts.append(span_text)
                # Track dominant font by longest span
                if len(span_text) > max_span_len:
                    max_span_len = len(span_text)
                    dominant_font_size = span.get("size", 0.0)
                    dominant_font_name = span.get("font", "")
            lines_text.append("".join(line_parts))

        text = "\n".join(lines_text).strip()
        if not text:
            return None

        return TextBlock(
            text=text,
            bbox=bbox,
            block_type="paragraph",
            confidence=1.0,
            page_num=page_idx + 1,
            font_size=dominant_font_size,
            font_name=dominant_font_name,
        )
