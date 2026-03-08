"""Abstract base extractor."""

from __future__ import annotations

import abc
import uuid
from pathlib import Path

from docminer.core.types import Document


class BaseExtractor(abc.ABC):
    """Abstract base class for all document extractors.

    Subclasses must implement :meth:`extract_document`.
    """

    def __init__(self, config=None) -> None:
        self.config = config

    @abc.abstractmethod
    def extract_document(self, path: str | Path) -> Document:
        """Extract content from *path* and return a :class:`Document`.

        Parameters
        ----------
        path:
            Filesystem path to the source document.

        Returns
        -------
        Document
            Populated document object with pages and text.
        """

    @staticmethod
    def make_document_id(path: str | Path) -> str:
        """Return a deterministic document ID based on the file path."""
        import hashlib

        return hashlib.sha1(str(path).encode()).hexdigest()[:12]

    @staticmethod
    def new_document_id() -> str:
        """Return a random document ID."""
        return uuid.uuid4().hex[:12]
