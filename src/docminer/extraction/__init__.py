"""Extraction sub-package — factory and base class."""

from docminer.extraction.base import BaseExtractor

__all__ = ["BaseExtractor", "create_extractor"]


def create_extractor(file_type: str, config=None) -> BaseExtractor:
    """Factory that returns the appropriate extractor for *file_type*.

    Parameters
    ----------
    file_type:
        One of ``"pdf"``, ``"image"``, ``"scan"``.
    config:
        Optional :class:`~docminer.config.schema.ExtractionConfig`.
    """
    ft = file_type.lower()
    if ft == "pdf":
        from docminer.extraction.pdf import PDFExtractor

        return PDFExtractor(config=config)
    if ft in ("image", "png", "jpg", "jpeg", "tiff", "tif", "bmp", "webp"):
        from docminer.extraction.image import ImageExtractor

        return ImageExtractor(config=config)
    if ft == "scan":
        from docminer.extraction.ocr import OCRExtractor

        return OCRExtractor(config=config)
    # Fallback: try PDF first, then image
    from docminer.extraction.pdf import PDFExtractor

    return PDFExtractor(config=config)
