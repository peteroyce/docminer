"""Storage sub-package."""

from docminer.storage.backend import SQLiteBackend, StorageBackend

__all__ = ["StorageBackend", "SQLiteBackend"]
