"""Image extractor — preprocesses images then runs OCR."""

from __future__ import annotations

import logging
from pathlib import Path

from docminer.core.types import Document, Page
from docminer.extraction.base import BaseExtractor

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".gif"}


class ImageExtractor(BaseExtractor):
    """Extract text from image files.

    Pipeline:
      1. Load image with PIL
      2. Apply preprocessing (deskew, denoise, binarize)
      3. Run OCR via :class:`~docminer.extraction.ocr.OCRExtractor`

    Parameters
    ----------
    config:
        Optional :class:`~docminer.config.schema.ExtractionConfig`.
    """

    def __init__(self, config=None) -> None:
        super().__init__(config=config)
        self._preprocess = True
        if config is not None:
            self._preprocess = getattr(config, "preprocess_images", True)

    def extract_document(self, path: str | Path) -> Document:
        """Extract text from an image file."""
        from PIL import Image

        from docminer.extraction.ocr import OCRExtractor

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        suffix = path.suffix.lower()
        if suffix not in _SUPPORTED_EXTENSIONS:
            logger.warning("Possibly unsupported image extension: %s", suffix)

        doc_id = self.make_document_id(path)
        logger.info("Extracting image: %s (id=%s)", path.name, doc_id)

        img = Image.open(path).convert("RGB")

        if self._preprocess:
            img = self._apply_preprocessing(img)

        ocr_extractor = OCRExtractor(config=self.config)
        page: Page = ocr_extractor.ocr_image(img, page_num=1)

        full_text = page.text
        return Document(
            id=doc_id,
            source_path=str(path),
            file_type="image",
            pages=[page],
            metadata={
                "original_size": list(img.size),
                "mode": img.mode,
            },
            text=full_text,
        )

    # ------------------------------------------------------------------
    # Preprocessing delegation
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_preprocessing(image):
        """Apply image preprocessing pipeline before OCR."""
        try:
            from docminer.preprocessing.image_prep import (
                binarize,
                denoise,
                deskew,
                enhance_contrast,
            )

            image = deskew(image)
            image = denoise(image)
            image = enhance_contrast(image)
            image = binarize(image)
            return image
        except Exception as exc:
            logger.warning("Image preprocessing failed, using raw image: %s", exc)
            return image
