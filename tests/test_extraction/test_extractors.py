"""Tests for PDF, OCR, and image extractors."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from docminer.core.types import BoundingBox, Document, Page, TextBlock


# ---------------------------------------------------------------------------
# PDFExtractor tests
# ---------------------------------------------------------------------------


class TestPDFExtractor:
    def test_extract_returns_document(self, sample_pdf_path: Path) -> None:
        """PDFExtractor should return a Document with at least one page."""
        pytest.importorskip("fitz", reason="PyMuPDF not installed")
        from docminer.extraction.pdf import PDFExtractor

        extractor = PDFExtractor()
        doc = extractor.extract_document(sample_pdf_path)

        assert isinstance(doc, Document)
        assert doc.file_type == "pdf"
        assert doc.page_count >= 1
        assert doc.source_path == str(sample_pdf_path)
        assert len(doc.id) > 0

    def test_extract_fills_text(self, sample_pdf_path: Path) -> None:
        """Full text should be populated after extraction."""
        pytest.importorskip("fitz", reason="PyMuPDF not installed")
        from docminer.extraction.pdf import PDFExtractor

        extractor = PDFExtractor()
        doc = extractor.extract_document(sample_pdf_path)
        assert isinstance(doc.text, str)

    def test_extract_metadata(self, sample_pdf_path: Path) -> None:
        """Metadata dict should be present."""
        pytest.importorskip("fitz", reason="PyMuPDF not installed")
        from docminer.extraction.pdf import PDFExtractor

        extractor = PDFExtractor()
        doc = extractor.extract_document(sample_pdf_path)
        assert isinstance(doc.metadata, dict)
        assert "page_count" in doc.metadata

    def test_file_not_found(self, tmp_path: Path) -> None:
        pytest.importorskip("fitz", reason="PyMuPDF not installed")
        from docminer.extraction.pdf import PDFExtractor

        extractor = PDFExtractor()
        with pytest.raises(FileNotFoundError):
            extractor.extract_document(tmp_path / "nonexistent.pdf")

    def test_document_id_is_deterministic(self, sample_pdf_path: Path) -> None:
        pytest.importorskip("fitz", reason="PyMuPDF not installed")
        from docminer.extraction.pdf import PDFExtractor

        extractor = PDFExtractor()
        doc1 = extractor.extract_document(sample_pdf_path)
        doc2 = extractor.extract_document(sample_pdf_path)
        assert doc1.id == doc2.id

    def test_invoice_pdf_has_text(self, invoice_pdf_path: Path) -> None:
        pytest.importorskip("fitz", reason="PyMuPDF not installed")
        from docminer.extraction.pdf import PDFExtractor

        extractor = PDFExtractor()
        doc = extractor.extract_document(invoice_pdf_path)
        # Invoice text should contain numeric amounts
        assert doc.text is not None


# ---------------------------------------------------------------------------
# OCRExtractor tests (mocked Tesseract)
# ---------------------------------------------------------------------------


class TestOCRExtractor:
    def _make_mock_tsv(self) -> dict:
        """Return a pytesseract DICT output mock."""
        return {
            "level": [5, 5, 5],
            "page_num": [1, 1, 1],
            "block_num": [1, 1, 1],
            "par_num": [1, 1, 1],
            "line_num": [1, 1, 1],
            "word_num": [1, 2, 3],
            "left": [10, 50, 100],
            "top": [20, 20, 20],
            "width": [30, 40, 50],
            "height": [15, 15, 15],
            "conf": [90, 85, 88],
            "text": ["Hello", "World", "Test"],
        }

    def test_ocr_image_returns_page(self, tmp_path: Path) -> None:
        pytest.importorskip("PIL", reason="Pillow not installed")
        from PIL import Image

        from docminer.extraction.ocr import OCRExtractor

        img = Image.new("RGB", (200, 100), color=(255, 255, 255))
        extractor = OCRExtractor()

        mock_tsv = self._make_mock_tsv()
        with patch("pytesseract.image_to_data", return_value=mock_tsv):
            page = extractor.ocr_image(img, page_num=1)

        assert isinstance(page, Page)
        assert page.number == 1
        assert len(page.blocks) > 0
        text = " ".join(b.text for b in page.blocks)
        assert "Hello" in text or "World" in text

    def test_ocr_file_not_found(self, tmp_path: Path) -> None:
        from docminer.extraction.ocr import OCRExtractor

        extractor = OCRExtractor()
        with pytest.raises(FileNotFoundError):
            extractor.extract_document(tmp_path / "no_such_file.png")

    def test_parse_tsv_groups_by_block(self) -> None:
        from docminer.extraction.ocr import OCRExtractor

        tsv = {
            "level": [5, 5, 5, 5],
            "page_num": [1, 1, 1, 1],
            "block_num": [1, 1, 2, 2],
            "par_num": [1, 1, 1, 1],
            "line_num": [1, 1, 1, 1],
            "word_num": [1, 2, 1, 2],
            "left": [10, 60, 200, 250],
            "top": [10, 10, 10, 10],
            "width": [40, 40, 40, 40],
            "height": [15, 15, 15, 15],
            "conf": [90, 90, 80, 80],
            "text": ["Block", "One", "Block", "Two"],
        }
        blocks = OCRExtractor._parse_tsv(tsv, page_num=1)
        assert len(blocks) == 2

    def test_parse_tsv_filters_low_confidence(self) -> None:
        from docminer.extraction.ocr import OCRExtractor

        tsv = {
            "level": [5],
            "page_num": [1],
            "block_num": [1],
            "par_num": [1],
            "line_num": [1],
            "word_num": [1],
            "left": [10],
            "top": [10],
            "width": [40],
            "height": [15],
            "conf": [-1],  # -1 = non-word
            "text": [""],
        }
        blocks = OCRExtractor._parse_tsv(tsv, page_num=1)
        assert len(blocks) == 0


# ---------------------------------------------------------------------------
# ImageExtractor tests (mocked OCR)
# ---------------------------------------------------------------------------


class TestImageExtractor:
    def test_extract_image(self, sample_image_path: Path) -> None:
        from docminer.extraction.image import ImageExtractor

        extractor = ImageExtractor()
        mock_tsv = {
            "level": [5],
            "page_num": [1],
            "block_num": [1],
            "par_num": [1],
            "line_num": [1],
            "word_num": [1],
            "left": [10],
            "top": [10],
            "width": [100],
            "height": [20],
            "conf": [90],
            "text": ["SampleText"],
        }
        with patch("pytesseract.image_to_data", return_value=mock_tsv):
            doc = extractor.extract_document(sample_image_path)

        assert isinstance(doc, Document)
        assert doc.file_type == "image"
        assert doc.page_count == 1

    def test_extract_image_file_not_found(self, tmp_path: Path) -> None:
        from docminer.extraction.image import ImageExtractor

        extractor = ImageExtractor()
        with pytest.raises(FileNotFoundError):
            extractor.extract_document(tmp_path / "missing.png")


# ---------------------------------------------------------------------------
# create_extractor factory
# ---------------------------------------------------------------------------


class TestCreateExtractor:
    def test_create_pdf_extractor(self) -> None:
        from docminer.extraction import create_extractor
        from docminer.extraction.pdf import PDFExtractor

        extractor = create_extractor("pdf")
        assert isinstance(extractor, PDFExtractor)

    def test_create_image_extractor(self) -> None:
        from docminer.extraction import create_extractor
        from docminer.extraction.image import ImageExtractor

        extractor = create_extractor("image")
        assert isinstance(extractor, ImageExtractor)

    def test_create_scan_extractor(self) -> None:
        from docminer.extraction import create_extractor
        from docminer.extraction.ocr import OCRExtractor

        extractor = create_extractor("scan")
        assert isinstance(extractor, OCRExtractor)

    def test_create_unknown_falls_back_to_pdf(self) -> None:
        from docminer.extraction import create_extractor
        from docminer.extraction.pdf import PDFExtractor

        extractor = create_extractor("xyz_unknown")
        assert isinstance(extractor, PDFExtractor)
