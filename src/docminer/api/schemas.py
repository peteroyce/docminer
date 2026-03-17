"""Pydantic request/response models for the DocMiner API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Common nested models
# ---------------------------------------------------------------------------


class BoundingBoxResponse(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class EntityResponse(BaseModel):
    text: str
    entity_type: str
    start: int
    end: int
    confidence: float
    normalized: Optional[str] = None
    role: Optional[str] = None


class TableResponse(BaseModel):
    headers: Optional[list[str]] = None
    rows: list[list[str]]
    num_rows: int
    num_cols: int
    page_num: int
    caption: str = ""


class ClassificationResponse(BaseModel):
    document_type: str
    confidence: float
    all_scores: dict[str, float]


class TextBlockResponse(BaseModel):
    text: str
    block_type: str
    confidence: float
    page_num: int
    font_size: float = 0.0
    bbox: Optional[BoundingBoxResponse] = None


class PageResponse(BaseModel):
    number: int
    width: float
    height: float
    blocks: list[TextBlockResponse]
    tables: list[TableResponse]


class DocumentResponse(BaseModel):
    id: str
    source_path: str
    file_type: str
    page_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Endpoint response models
# ---------------------------------------------------------------------------


class ExtractionResponse(BaseModel):
    """Full extraction result returned by POST /extract and POST /analyze."""

    document: DocumentResponse
    entities: list[EntityResponse]
    tables: list[TableResponse]
    classification: Optional[ClassificationResponse] = None
    summary: Optional[str] = None
    keywords: list[str]
    processing_time_ms: float
    errors: list[str] = Field(default_factory=list)


class ClassifyResponse(BaseModel):
    """Response from POST /classify."""

    document_id: str
    classification: ClassificationResponse
    processing_time_ms: float


class DocumentListItem(BaseModel):
    """Single item in the GET /documents listing."""

    document_id: str
    source_path: str
    file_type: str
    page_count: int
    document_type: Optional[str] = None
    classification_confidence: Optional[float] = None
    created_at: Optional[str] = None


class DocumentListResponse(BaseModel):
    """Paginated list from GET /documents."""

    documents: list[DocumentListItem]
    total: int
    limit: int
    offset: int


class DocumentDetailResponse(BaseModel):
    """Full document detail from GET /documents/{id}."""

    document_id: str
    source_path: str
    file_type: str
    page_count: int
    document_type: Optional[str] = None
    classification_confidence: Optional[float] = None
    summary: Optional[str] = None
    keywords: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None


class HealthResponse(BaseModel):
    """GET /health response."""

    status: str
    version: str
    components: dict[str, str]


class ErrorResponse(BaseModel):
    """Generic error response."""

    detail: str
    error_type: str = "error"
