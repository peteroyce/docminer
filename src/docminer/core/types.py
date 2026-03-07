"""Core data types for the DocMiner pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BoundingBox:
    """Axis-aligned bounding box in page coordinate space (points or pixels)."""

    x0: float
    y0: float
    x1: float
    y1: float

    # ------------------------------------------------------------------
    # Basic geometry
    # ------------------------------------------------------------------

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    # ------------------------------------------------------------------
    # Spatial relationships
    # ------------------------------------------------------------------

    def overlap(self, other: BoundingBox) -> float:
        """Return the overlapping area between this box and *other*."""
        ix0 = max(self.x0, other.x0)
        iy0 = max(self.y0, other.y0)
        ix1 = min(self.x1, other.x1)
        iy1 = min(self.y1, other.y1)
        if ix0 >= ix1 or iy0 >= iy1:
            return 0.0
        return (ix1 - ix0) * (iy1 - iy0)

    def iou(self, other: BoundingBox) -> float:
        """Intersection-over-Union (Jaccard index)."""
        inter = self.overlap(other)
        if inter == 0.0:
            return 0.0
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def contains(self, other: BoundingBox) -> bool:
        """Return True if this box fully contains *other*."""
        return (
            self.x0 <= other.x0
            and self.y0 <= other.y0
            and self.x1 >= other.x1
            and self.y1 >= other.y1
        )

    def merge(self, other: BoundingBox) -> BoundingBox:
        """Return the smallest box that contains both boxes."""
        return BoundingBox(
            x0=min(self.x0, other.x0),
            y0=min(self.y0, other.y0),
            x1=max(self.x1, other.x1),
            y1=max(self.y1, other.y1),
        )

    def distance_to(self, other: BoundingBox) -> float:
        """Euclidean distance between the centers of the two boxes."""
        cx1, cy1 = self.center
        cx2, cy2 = other.center
        return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

    def to_dict(self) -> dict[str, float]:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> BoundingBox:
        return cls(x0=data["x0"], y0=data["y0"], x1=data["x1"], y1=data["y1"])

    def __repr__(self) -> str:
        return (
            f"BoundingBox(x0={self.x0:.1f}, y0={self.y0:.1f}, "
            f"x1={self.x1:.1f}, y1={self.y1:.1f})"
        )


@dataclass
class TextBlock:
    """A contiguous region of text extracted from a page."""

    text: str
    bbox: Optional[BoundingBox] = None
    # paragraph | header | list_item | caption | footer | table_cell
    block_type: str = "paragraph"
    confidence: float = 1.0
    page_num: int = 0
    font_size: float = 0.0
    font_name: str = ""
    metadata: dict = field(default_factory=dict)

    def word_count(self) -> int:
        return len(self.text.split())

    def is_empty(self) -> bool:
        return not self.text.strip()

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "block_type": self.block_type,
            "confidence": self.confidence,
            "page_num": self.page_num,
            "font_size": self.font_size,
            "font_name": self.font_name,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "metadata": self.metadata,
        }


@dataclass
class Table:
    """A table extracted from a document page."""

    rows: list[list[str]]
    bbox: Optional[BoundingBox] = None
    headers: Optional[list[str]] = None
    page_num: int = 0
    caption: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def num_rows(self) -> int:
        return len(self.rows)

    @property
    def num_cols(self) -> int:
        return max((len(r) for r in self.rows), default=0)

    def to_dict(self) -> dict:
        return {
            "headers": self.headers,
            "rows": self.rows,
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "page_num": self.page_num,
            "caption": self.caption,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "metadata": self.metadata,
        }

    def to_csv_rows(self) -> list[list[str]]:
        """Return all rows optionally preceded by header row."""
        if self.headers:
            return [self.headers] + self.rows
        return self.rows


@dataclass
class Entity:
    """A named entity extracted from document text."""

    text: str
    # date | amount | email | phone | address | person | organization |
    # reference_number | url
    entity_type: str
    start: int
    end: int
    confidence: float = 1.0
    normalized: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "entity_type": self.entity_type,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "normalized": self.normalized,
            "metadata": self.metadata,
        }


@dataclass
class Page:
    """A single page of a document."""

    number: int
    width: float
    height: float
    blocks: list[TextBlock] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.blocks if not b.is_empty())

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "width": self.width,
            "height": self.height,
            "blocks": [b.to_dict() for b in self.blocks],
            "tables": [t.to_dict() for t in self.tables],
            "images": self.images,
            "metadata": self.metadata,
        }


@dataclass
class Document:
    """The top-level representation of a processed document."""

    id: str
    source_path: str
    # pdf | image | scan
    file_type: str
    pages: list[Page] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    text: str = ""

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def all_blocks(self) -> list[TextBlock]:
        return [block for page in self.pages for block in page.blocks]

    @property
    def all_tables(self) -> list[Table]:
        return [table for page in self.pages for table in page.tables]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_path": self.source_path,
            "file_type": self.file_type,
            "page_count": self.page_count,
            "pages": [p.to_dict() for p in self.pages],
            "metadata": self.metadata,
            "text": self.text,
        }


@dataclass
class ClassificationResult:
    """Result of document type classification."""

    document_type: str
    confidence: float
    all_scores: dict[str, float]
    features_used: list[str]

    def to_dict(self) -> dict:
        return {
            "document_type": self.document_type,
            "confidence": self.confidence,
            "all_scores": self.all_scores,
            "features_used": self.features_used,
        }


@dataclass
class ExtractionResult:
    """Fully processed result for a single document."""

    document: Document
    entities: list[Entity] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    classification: Optional[ClassificationResult] = None
    summary: Optional[str] = None
    keywords: list[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "document": self.document.to_dict(),
            "entities": [e.to_dict() for e in self.entities],
            "tables": [t.to_dict() for t in self.tables],
            "classification": self.classification.to_dict() if self.classification else None,
            "summary": self.summary,
            "keywords": self.keywords,
            "processing_time_ms": self.processing_time_ms,
            "errors": self.errors,
        }
