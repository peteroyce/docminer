"""Pydantic configuration schemas."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class OCRConfig(BaseModel):
    """Configuration for the Tesseract OCR engine."""

    language: str = Field(default="eng", description="Tesseract language code(s)")
    dpi: int = Field(default=300, ge=72, le=1200, description="DPI for PDF-to-image rendering")
    psm: int = Field(
        default=3,
        ge=0,
        le=13,
        description="Tesseract page segmentation mode",
    )
    confidence_threshold: float = Field(
        default=30.0,
        ge=0.0,
        le=100.0,
        description="Minimum word confidence to include (0–100)",
    )


class ExtractionConfig(BaseModel):
    """Configuration for document extraction."""

    preprocess_images: bool = Field(
        default=True,
        description="Apply image preprocessing before OCR",
    )
    extract_tables: bool = Field(
        default=True,
        description="Enable table extraction from PDFs",
    )
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    max_pages: Optional[int] = Field(
        default=None,
        description="Maximum number of pages to process (None = all)",
    )


class ClassificationConfig(BaseModel):
    """Configuration for document classification."""

    confidence_threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Global minimum confidence for classification",
    )
    use_ml_model: bool = Field(
        default=True,
        description="Use TF-IDF + LR model (falls back to rule-based if unavailable)",
    )


class AnalysisConfig(BaseModel):
    """Configuration for document analysis."""

    summary_sentences: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of sentences in the extractive summary",
    )
    top_keywords: int = Field(
        default=15,
        ge=1,
        le=100,
        description="Number of keywords to extract",
    )


class PipelineConfig(BaseModel):
    """Toggle individual pipeline steps."""

    enable_layout: bool = Field(default=True)
    enable_classification: bool = Field(default=True)
    enable_entities: bool = Field(default=True)
    enable_entity_linking: bool = Field(default=True)
    enable_analysis: bool = Field(default=True)


class StorageConfig(BaseModel):
    """Configuration for the storage backend."""

    backend: str = Field(
        default="sqlite",
        description="Storage backend type ('sqlite' or 'none')",
    )
    db_path: str = Field(
        default="docminer.db",
        description="Path to the SQLite database file",
    )


class ServerConfig(BaseModel):
    """FastAPI server settings."""

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    reload: bool = Field(default=False)
    workers: int = Field(default=1, ge=1)


class DocMinerConfig(BaseModel):
    """Root configuration for the DocMiner pipeline."""

    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    log_level: str = Field(default="INFO", description="Python logging level")

    class Config:
        extra = "ignore"
