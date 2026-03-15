"""Storage backends — abstract base and SQLite implementation."""

from __future__ import annotations

import abc
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class StorageBackend(abc.ABC):
    """Abstract storage backend for processed documents."""

    @abc.abstractmethod
    def save(self, result) -> int:
        """Persist an :class:`~docminer.core.types.ExtractionResult` and return its row id."""

    @abc.abstractmethod
    def get(self, document_id: str) -> Optional[dict]:
        """Retrieve a processed document by its *document_id*."""

    @abc.abstractmethod
    def list_documents(self, limit: int = 100, offset: int = 0) -> list[dict]:
        """Return a paginated list of stored document summaries."""

    @abc.abstractmethod
    def delete(self, document_id: str) -> bool:
        """Delete the document with *document_id*.  Returns True if found."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release resources held by the backend."""


class SQLiteBackend(StorageBackend):
    """SQLite-backed storage using SQLAlchemy.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Use ``":memory:"`` for an
        in-memory database (useful for tests).
    """

    def __init__(self, db_path: str | Path = "docminer.db") -> None:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from docminer.storage.models import Base

        url = (
            "sqlite:///:memory:"
            if str(db_path) == ":memory:"
            else f"sqlite:///{Path(db_path).resolve()}"
        )
        self._engine = create_engine(url, connect_args={"check_same_thread": False})
        Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)
        logger.info("SQLiteBackend initialised at %s", db_path)

    # ------------------------------------------------------------------
    # StorageBackend interface
    # ------------------------------------------------------------------

    def save(self, result) -> int:
        """Persist *result* and return the primary-key row id."""
        from docminer.storage.models import DocumentChunk, ExtractedEntity, ProcessedDocument

        session = self._Session()
        try:
            doc = result.document
            # Upsert: delete existing record if present
            existing = (
                session.query(ProcessedDocument)
                .filter_by(document_id=doc.id)
                .first()
            )
            if existing:
                session.delete(existing)
                session.flush()

            pd = ProcessedDocument(
                document_id=doc.id,
                source_path=doc.source_path,
                file_type=doc.file_type,
                page_count=doc.page_count,
                document_type=(
                    result.classification.document_type if result.classification else None
                ),
                classification_confidence=(
                    result.classification.confidence if result.classification else None
                ),
                summary=result.summary,
                processing_time_ms=result.processing_time_ms,
                full_text=doc.text[:100_000],  # cap to 100 KB
            )
            pd.keywords = result.keywords
            pd.doc_metadata = doc.metadata
            session.add(pd)
            session.flush()  # get pd.id

            # Entities
            for ent in result.entities:
                ee = ExtractedEntity(
                    document_id=pd.id,
                    entity_text=ent.text,
                    entity_type=ent.entity_type,
                    normalized=ent.normalized,
                    confidence=ent.confidence,
                    start_offset=ent.start,
                    end_offset=ent.end,
                    role=ent.metadata.get("role"),
                )
                session.add(ee)

            # Chunks (one per TextBlock)
            for page in doc.pages:
                for i, block in enumerate(page.blocks):
                    dc = DocumentChunk(
                        document_id=pd.id,
                        chunk_index=i,
                        page_num=page.number,
                        block_type=block.block_type,
                        text=block.text,
                        bbox_json=json.dumps(block.bbox.to_dict()) if block.bbox else None,
                    )
                    session.add(dc)

            session.commit()
            row_id: int = pd.id
            logger.info("Saved document %s (row_id=%d)", doc.id, row_id)
            return row_id
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get(self, document_id: str) -> Optional[dict]:
        """Retrieve a document by its string *document_id*."""
        from docminer.storage.models import ProcessedDocument

        session = self._Session()
        try:
            pd = (
                session.query(ProcessedDocument)
                .filter_by(document_id=document_id)
                .first()
            )
            if pd is None:
                return None
            return self._pd_to_dict(pd)
        finally:
            session.close()

    def list_documents(self, limit: int = 100, offset: int = 0) -> list[dict]:
        """List documents with pagination."""
        from docminer.storage.models import ProcessedDocument

        session = self._Session()
        try:
            rows = (
                session.query(ProcessedDocument)
                .order_by(ProcessedDocument.created_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
            return [self._pd_to_dict(r) for r in rows]
        finally:
            session.close()

    def delete(self, document_id: str) -> bool:
        """Delete document with the given *document_id*."""
        from docminer.storage.models import ProcessedDocument

        session = self._Session()
        try:
            pd = (
                session.query(ProcessedDocument)
                .filter_by(document_id=document_id)
                .first()
            )
            if pd is None:
                return False
            session.delete(pd)
            session.commit()
            return True
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def close(self) -> None:
        """Dispose the SQLAlchemy engine."""
        self._engine.dispose()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pd_to_dict(pd) -> dict:
        return {
            "document_id": pd.document_id,
            "source_path": pd.source_path,
            "file_type": pd.file_type,
            "page_count": pd.page_count,
            "document_type": pd.document_type,
            "classification_confidence": pd.classification_confidence,
            "summary": pd.summary,
            "keywords": pd.keywords,
            "processing_time_ms": pd.processing_time_ms,
            "created_at": pd.created_at.isoformat() if pd.created_at else None,
            "metadata": pd.doc_metadata,
        }
