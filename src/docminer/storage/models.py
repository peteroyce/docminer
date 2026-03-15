"""SQLAlchemy ORM models for persisting pipeline results."""

from __future__ import annotations

import datetime
import json

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class ProcessedDocument(Base):
    """Stores top-level metadata for each processed document."""

    __tablename__ = "processed_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String(64), unique=True, nullable=False, index=True)
    source_path = Column(Text, nullable=False)
    file_type = Column(String(20), nullable=False)
    page_count = Column(Integer, default=0)
    document_type = Column(String(50), nullable=True)
    classification_confidence = Column(Float, nullable=True)
    summary = Column(Text, nullable=True)
    keywords_json = Column(Text, nullable=True)  # JSON array
    processing_time_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    metadata_json = Column(Text, nullable=True)  # JSON object
    full_text = Column(Text, nullable=True)

    entities = relationship(
        "ExtractedEntity", back_populates="document", cascade="all, delete-orphan"
    )
    chunks = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )

    @property
    def keywords(self) -> list[str]:
        if self.keywords_json:
            return json.loads(self.keywords_json)
        return []

    @keywords.setter
    def keywords(self, value: list[str]) -> None:
        self.keywords_json = json.dumps(value)

    @property
    def doc_metadata(self) -> dict:
        if self.metadata_json:
            return json.loads(self.metadata_json)
        return {}

    @doc_metadata.setter
    def doc_metadata(self, value: dict) -> None:
        self.metadata_json = json.dumps(value)

    def __repr__(self) -> str:
        return (
            f"<ProcessedDocument id={self.document_id!r} "
            f"type={self.document_type!r} path={self.source_path!r}>"
        )


class ExtractedEntity(Base):
    """Stores individual named entities extracted from a document."""

    __tablename__ = "extracted_entities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(
        Integer, ForeignKey("processed_documents.id", ondelete="CASCADE"), nullable=False
    )
    entity_text = Column(Text, nullable=False)
    entity_type = Column(String(50), nullable=False, index=True)
    normalized = Column(Text, nullable=True)
    confidence = Column(Float, default=1.0)
    start_offset = Column(Integer, nullable=True)
    end_offset = Column(Integer, nullable=True)
    role = Column(String(100), nullable=True)

    document = relationship("ProcessedDocument", back_populates="entities")

    def __repr__(self) -> str:
        return (
            f"<ExtractedEntity type={self.entity_type!r} "
            f"text={self.entity_text!r} role={self.role!r}>"
        )


class DocumentChunk(Base):
    """Stores text chunks (paragraphs / sections) for retrieval use-cases."""

    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(
        Integer, ForeignKey("processed_documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index = Column(Integer, nullable=False)
    page_num = Column(Integer, nullable=True)
    block_type = Column(String(50), nullable=True)
    text = Column(Text, nullable=False)
    bbox_json = Column(Text, nullable=True)

    document = relationship("ProcessedDocument", back_populates="chunks")

    def __repr__(self) -> str:
        return (
            f"<DocumentChunk doc_id={self.document_id} "
            f"index={self.chunk_index} page={self.page_num}>"
        )
