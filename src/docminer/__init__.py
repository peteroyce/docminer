"""DocMiner — document intelligence pipeline."""

__version__ = "0.2.0"
__author__ = "DocMiner Contributors"

from docminer.core.types import (
    BoundingBox,
    ClassificationResult,
    Document,
    Entity,
    ExtractionResult,
    Page,
    Table,
    TextBlock,
)

__all__ = [
    "__version__",
    "BoundingBox",
    "ClassificationResult",
    "Document",
    "Entity",
    "ExtractionResult",
    "Page",
    "Table",
    "TextBlock",
]
