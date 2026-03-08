"""OCR extractor using Tesseract via pytesseract."""

from __future__ import annotations

import logging
from pathlib import Path

from docminer.core.types import BoundingBox, Document, Page, TextBlock
from docminer.extraction.base import BaseExtractor

logger = logging.getLogger(__name__)

# Tesseract output data column indices
_TSV_LEVEL = 0
_TSV_PAGE_NUM = 1
_TSV_BLOCK_NUM = 2
_TSV_PAR_NUM = 3
_TSV_LINE_NUM = 4
_TSV_WORD_NUM = 5
_TSV_LEFT = 6
_TSV_TOP = 7
_TSV_WIDTH = 8
_TSV_HEIGHT = 9
_TSV_CONF = 10
_TSV_TEXT = 11


class OCRExtractor(BaseExtractor):
    """Extract text from scanned documents or rasterised pages using Tesseract.

    Parameters
    ----------
    config:
        Optional :class:`~docminer.config.schema.OCRConfig`.
    """

    def __init__(self, config=None) -> None:
        super().__init__(config=config)
        self._lang = "eng"
        self._dpi = 300
        self._psm = 3  # Tesseract PSM: fully automatic page segmentation
        if config is not None:
            self._lang = getattr(config, "language", "eng")
            self._dpi = getattr(config, "dpi", 300)
            self._psm = getattr(config, "psm", 3)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_document(self, path: str | Path) -> Document:
        """Extract text from an image or PDF file via OCR."""
        from PIL import Image

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        doc_id = self.make_document_id(path)
        logger.info("OCR extracting: %s (id=%s)", path.name, doc_id)

        pages: list[Page] = []
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            pages = self._extract_pdf_via_ocr(path)
        else:
            img = Image.open(path)
            page = self._ocr_image(img, page_num=1)
            pages = [page]

        full_text = "\n\n".join(p.text for p in pages)
        return Document(
            id=doc_id,
            source_path=str(path),
            file_type="scan",
            pages=pages,
            metadata={"ocr_language": self._lang, "dpi": self._dpi},
            text=full_text,
        )

    def ocr_image(self, image, page_num: int = 1) -> Page:
        """Public wrapper for external callers."""
        return self._ocr_image(image, page_num=page_num)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_pdf_via_ocr(self, path: Path) -> list[Page]:
        """Render each PDF page and run OCR."""
        import fitz
        from PIL import Image

        pdf = fitz.open(str(path))
        pages: list[Page] = []
        try:
            for i in range(len(pdf)):
                fitz_page = pdf[i]
                # Render at configured DPI
                zoom = self._dpi / 72
                mat = fitz.Matrix(zoom, zoom)
                pix = fitz_page.get_pixmap(matrix=mat, alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page = self._ocr_image(img, page_num=i + 1)
                pages.append(page)
        finally:
            pdf.close()
        return pages

    def _ocr_image(self, image, page_num: int = 1) -> Page:
        """Run Tesseract on a PIL image and return a :class:`Page`."""
        import pytesseract
        from PIL import Image

        width, height = image.size

        cfg = f"--psm {self._psm}"
        try:
            tsv_data = pytesseract.image_to_data(
                image,
                lang=self._lang,
                config=cfg,
                output_type=pytesseract.Output.DICT,
            )
        except Exception as exc:
            logger.warning("Tesseract failed on page %d: %s", page_num, exc)
            return Page(number=page_num, width=float(width), height=float(height))

        blocks = self._parse_tsv(tsv_data, page_num)
        return Page(
            number=page_num,
            width=float(width),
            height=float(height),
            blocks=blocks,
        )

    @staticmethod
    def _parse_tsv(data: dict, page_num: int) -> list[TextBlock]:
        """Group Tesseract word-level output into paragraph blocks."""
        n = len(data.get("text", []))
        # Group by block_num + par_num
        groups: dict[tuple[int, int], list[dict]] = {}
        for i in range(n):
            conf = int(data["conf"][i])
            if conf < 0:
                continue  # non-word row
            text = data["text"][i].strip()
            if not text:
                continue
            block_num = data["block_num"][i]
            par_num = data["par_num"][i]
            key = (block_num, par_num)
            groups.setdefault(key, []).append(
                {
                    "text": text,
                    "left": data["left"][i],
                    "top": data["top"][i],
                    "width": data["width"][i],
                    "height": data["height"][i],
                    "conf": conf,
                }
            )

        blocks: list[TextBlock] = []
        for words in groups.values():
            if not words:
                continue
            combined_text = " ".join(w["text"] for w in words)
            avg_conf = sum(w["conf"] for w in words) / len(words)
            x0 = min(w["left"] for w in words)
            y0 = min(w["top"] for w in words)
            x1 = max(w["left"] + w["width"] for w in words)
            y1 = max(w["top"] + w["height"] for w in words)
            blocks.append(
                TextBlock(
                    text=combined_text,
                    bbox=BoundingBox(x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1)),
                    block_type="paragraph",
                    confidence=avg_conf / 100.0,
                    page_num=page_num,
                )
            )
        return blocks
