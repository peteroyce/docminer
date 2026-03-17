"""FastAPI route definitions."""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse

from docminer.api.schemas import (
    ClassificationResponse,
    ClassifyResponse,
    DocumentDetailResponse,
    DocumentListItem,
    DocumentListResponse,
    EntityResponse,
    ExtractionResponse,
    HealthResponse,
    TableResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Dependency: shared Pipeline and Storage instances
# ---------------------------------------------------------------------------


def get_pipeline():
    """Dependency that returns the application-level Pipeline instance."""
    from docminer.api.app import _pipeline

    return _pipeline


def get_storage():
    """Dependency that returns the application-level Storage instance."""
    from docminer.api.app import _storage

    return _storage


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """Return the service health status."""
    import docminer

    components: dict[str, str] = {}

    # Check tesseract
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        components["tesseract"] = "ok"
    except Exception:
        components["tesseract"] = "unavailable"

    # Check fitz
    try:
        import fitz  # noqa

        components["pymupdf"] = "ok"
    except ImportError:
        components["pymupdf"] = "unavailable"

    # Check sklearn
    try:
        import sklearn  # noqa

        components["scikit_learn"] = "ok"
    except ImportError:
        components["scikit_learn"] = "unavailable"

    return HealthResponse(
        status="ok",
        version=docminer.__version__,
        components=components,
    )


# ---------------------------------------------------------------------------
# POST /extract — upload file and extract text
# ---------------------------------------------------------------------------


@router.post(
    "/extract",
    response_model=ExtractionResponse,
    status_code=status.HTTP_200_OK,
    tags=["extraction"],
    summary="Extract text and structure from a document",
)
async def extract_document(
    file: UploadFile = File(..., description="Document file (PDF or image)"),
    pipeline=Depends(get_pipeline),
    storage=Depends(get_storage),
):
    """Upload a document and receive structured extraction results."""
    suffix = Path(file.filename or "upload").suffix.lower() or ".pdf"
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        result = pipeline.process_file(tmp_path)

        if storage is not None:
            try:
                storage.save(result)
            except Exception as exc:
                logger.warning("Failed to persist result: %s", exc)

        return _build_extraction_response(result)
    except Exception as exc:
        logger.exception("Extraction failed")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# POST /classify — classify document type
# ---------------------------------------------------------------------------


@router.post(
    "/classify",
    response_model=ClassifyResponse,
    status_code=status.HTTP_200_OK,
    tags=["classification"],
    summary="Classify the document type",
)
async def classify_document(
    file: UploadFile = File(..., description="Document file"),
    pipeline=Depends(get_pipeline),
):
    """Upload a document and get its document type classification."""
    suffix = Path(file.filename or "upload").suffix.lower() or ".pdf"
    tmp_path = ""
    try:
        start = time.perf_counter()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        from docminer.utils.file_utils import detect_file_type

        file_type = detect_file_type(Path(tmp_path))
        from docminer.extraction import create_extractor

        extractor = create_extractor(file_type)
        document = extractor.extract_document(tmp_path)
        classification = pipeline.classifier.classify(document)
        elapsed = (time.perf_counter() - start) * 1000

        return ClassifyResponse(
            document_id=document.id,
            classification=ClassificationResponse(
                document_type=classification.document_type,
                confidence=classification.confidence,
                all_scores=classification.all_scores,
            ),
            processing_time_ms=elapsed,
        )
    except Exception as exc:
        logger.exception("Classification failed")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# POST /analyze — full pipeline
# ---------------------------------------------------------------------------


@router.post(
    "/analyze",
    response_model=ExtractionResponse,
    status_code=status.HTTP_200_OK,
    tags=["analysis"],
    summary="Run the full pipeline on a document",
)
async def analyze_document(
    file: UploadFile = File(..., description="Document file"),
    pipeline=Depends(get_pipeline),
    storage=Depends(get_storage),
):
    """Run the complete DocMiner pipeline (extraction + classification + NER + analysis)."""
    # Reuse extract endpoint logic
    return await extract_document(file=file, pipeline=pipeline, storage=storage)


# ---------------------------------------------------------------------------
# GET /documents — list processed documents
# ---------------------------------------------------------------------------


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    tags=["documents"],
    summary="List processed documents",
)
async def list_documents(
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    storage=Depends(get_storage),
):
    """Return a paginated list of previously processed documents."""
    if storage is None:
        return DocumentListResponse(documents=[], total=0, limit=limit, offset=offset)

    docs = storage.list_documents(limit=limit, offset=offset)
    items = [
        DocumentListItem(
            document_id=d["document_id"],
            source_path=d["source_path"],
            file_type=d["file_type"],
            page_count=d["page_count"],
            document_type=d.get("document_type"),
            classification_confidence=d.get("classification_confidence"),
            created_at=d.get("created_at"),
        )
        for d in docs
    ]
    return DocumentListResponse(
        documents=items, total=len(items), limit=limit, offset=offset
    )


# ---------------------------------------------------------------------------
# GET /documents/{id} — document detail
# ---------------------------------------------------------------------------


@router.get(
    "/documents/{document_id}",
    response_model=DocumentDetailResponse,
    tags=["documents"],
    summary="Get details for a processed document",
)
async def get_document(
    document_id: str,
    storage=Depends(get_storage),
):
    """Retrieve the stored metadata for a document by its ID."""
    if storage is None:
        raise HTTPException(status_code=404, detail="Storage not configured")

    doc = storage.get(document_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found",
        )

    return DocumentDetailResponse(
        document_id=doc["document_id"],
        source_path=doc["source_path"],
        file_type=doc["file_type"],
        page_count=doc["page_count"],
        document_type=doc.get("document_type"),
        classification_confidence=doc.get("classification_confidence"),
        summary=doc.get("summary"),
        keywords=doc.get("keywords", []),
        metadata=doc.get("metadata", {}),
        created_at=doc.get("created_at"),
    )


# ---------------------------------------------------------------------------
# Response builder helper
# ---------------------------------------------------------------------------


def _build_extraction_response(result) -> ExtractionResponse:
    """Convert an ExtractionResult to an ExtractionResponse schema."""
    from docminer.api.schemas import (
        BoundingBoxResponse,
        ClassificationResponse,
        DocumentResponse,
        EntityResponse,
        ExtractionResponse,
        PageResponse,
        TableResponse,
        TextBlockResponse,
    )

    doc = result.document

    entities = [
        EntityResponse(
            text=e.text,
            entity_type=e.entity_type,
            start=e.start,
            end=e.end,
            confidence=e.confidence,
            normalized=e.normalized,
            role=e.metadata.get("role"),
        )
        for e in result.entities
    ]

    tables = [
        TableResponse(
            headers=t.headers,
            rows=t.rows,
            num_rows=t.num_rows,
            num_cols=t.num_cols,
            page_num=t.page_num,
            caption=t.caption,
        )
        for t in result.tables
    ]

    classification = None
    if result.classification:
        classification = ClassificationResponse(
            document_type=result.classification.document_type,
            confidence=result.classification.confidence,
            all_scores=result.classification.all_scores,
        )

    return ExtractionResponse(
        document=DocumentResponse(
            id=doc.id,
            source_path=doc.source_path,
            file_type=doc.file_type,
            page_count=doc.page_count,
            metadata=doc.metadata,
        ),
        entities=entities,
        tables=tables,
        classification=classification,
        summary=result.summary,
        keywords=result.keywords,
        processing_time_ms=result.processing_time_ms,
        errors=result.errors,
    )
