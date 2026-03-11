"""Document classification sub-package."""

from docminer.classification.classifier import DocumentClassifier
from docminer.classification.labels import DOCUMENT_TYPES

__all__ = ["DocumentClassifier", "DOCUMENT_TYPES"]
