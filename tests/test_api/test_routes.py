"""API route tests using httpx and TestClient."""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_pipeline():
    """Return a mock Pipeline that always returns a minimal ExtractionResult."""
    from docminer.core.types import (
        ClassificationResult,
        Document,
        ExtractionResult,
        Page,
        TextBlock,
    )

    block = TextBlock(text="Test document content", page_num=1)
    page = Page(number=1, width=612, height=792, blocks=[block])
    doc = Document(
        id="abc123",
        source_path="/tmp/test.pdf",
        file_type="pdf",
        pages=[page],
        text="Test document content",
    )
    classification = ClassificationResult(
        document_type="invoice",
        confidence=0.85,
        all_scores={"invoice": 0.85, "unknown": 0.15},
        features_used=["tfidf"],
    )
    result = ExtractionResult(
        document=doc,
        classification=classification,
        summary="Test summary.",
        keywords=["test", "document"],
        processing_time_ms=50.0,
    )

    pipeline = MagicMock()
    pipeline.process_file.return_value = result
    pipeline.classifier.classify.return_value = classification
    return pipeline


@pytest.fixture
def mock_storage():
    """Return a minimal mock storage backend."""
    storage = MagicMock()
    storage.save.return_value = 1
    storage.list_documents.return_value = [
        {
            "document_id": "abc123",
            "source_path": "/tmp/test.pdf",
            "file_type": "pdf",
            "page_count": 1,
            "document_type": "invoice",
            "classification_confidence": 0.85,
            "summary": "Test summary.",
            "keywords": ["test"],
            "processing_time_ms": 50.0,
            "created_at": "2024-01-15T12:00:00",
            "metadata": {},
        }
    ]
    storage.get.return_value = {
        "document_id": "abc123",
        "source_path": "/tmp/test.pdf",
        "file_type": "pdf",
        "page_count": 1,
        "document_type": "invoice",
        "classification_confidence": 0.85,
        "summary": "Test summary.",
        "keywords": ["test"],
        "processing_time_ms": 50.0,
        "created_at": "2024-01-15T12:00:00",
        "metadata": {},
    }
    return storage


@pytest.fixture
def app(mock_pipeline, mock_storage):
    """Create a test FastAPI app with mocked dependencies."""
    pytest.importorskip("fastapi", reason="FastAPI not installed")

    from docminer.api import create_app

    app = create_app()

    # Override the module-level singletons
    import docminer.api.app as app_module

    app_module._pipeline = mock_pipeline
    app_module._storage = mock_storage

    return app


@pytest.fixture
def client(app):
    """Return a synchronous TestClient for the app."""
    pytest.importorskip("httpx", reason="httpx not installed")
    from fastapi.testclient import TestClient

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


class TestHealthRoute:
    def test_health_returns_ok(self, client) -> None:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert "version" in body
        assert "components" in body


class TestExtractRoute:
    def test_extract_pdf(self, client, sample_pdf_bytes: bytes) -> None:
        response = client.post(
            "/api/v1/extract",
            files={"file": ("test.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")},
        )
        assert response.status_code == 200
        body = response.json()
        assert "document" in body
        assert body["document"]["id"] == "abc123"

    def test_extract_includes_classification(self, client, sample_pdf_bytes: bytes) -> None:
        response = client.post(
            "/api/v1/extract",
            files={"file": ("test.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")},
        )
        body = response.json()
        assert body["classification"]["document_type"] == "invoice"

    def test_extract_includes_keywords(self, client, sample_pdf_bytes: bytes) -> None:
        response = client.post(
            "/api/v1/extract",
            files={"file": ("test.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")},
        )
        body = response.json()
        assert "keywords" in body
        assert isinstance(body["keywords"], list)


class TestClassifyRoute:
    def test_classify_returns_classification(self, client, sample_pdf_bytes: bytes) -> None:
        from docminer.core.types import Document, Page, TextBlock

        with patch("docminer.utils.file_utils.detect_file_type", return_value="pdf"):
            mock_doc = Document(
                id="abc123",
                source_path="test.pdf",
                file_type="pdf",
                pages=[Page(number=1, width=612, height=792)],
                text="invoice total amount",
            )
            with patch("docminer.extraction.pdf.PDFExtractor.extract_document", return_value=mock_doc):
                response = client.post(
                    "/api/v1/classify",
                    files={"file": ("test.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")},
                )
        assert response.status_code == 200
        body = response.json()
        assert "classification" in body
        assert "document_type" in body["classification"]


class TestDocumentsRoute:
    def test_list_documents(self, client) -> None:
        response = client.get("/api/v1/documents")
        assert response.status_code == 200
        body = response.json()
        assert "documents" in body
        assert isinstance(body["documents"], list)
        assert len(body["documents"]) == 1
        assert body["documents"][0]["document_id"] == "abc123"

    def test_list_documents_pagination_params(self, client) -> None:
        response = client.get("/api/v1/documents?limit=5&offset=0")
        assert response.status_code == 200

    def test_get_document_by_id(self, client) -> None:
        response = client.get("/api/v1/documents/abc123")
        assert response.status_code == 200
        body = response.json()
        assert body["document_id"] == "abc123"
        assert body["document_type"] == "invoice"

    def test_get_nonexistent_document(self, client, mock_storage) -> None:
        mock_storage.get.return_value = None
        response = client.get("/api/v1/documents/nonexistent")
        assert response.status_code == 404


class TestRootRoute:
    def test_root_returns_name(self, client) -> None:
        response = client.get("/")
        assert response.status_code == 200
        body = response.json()
        assert "DocMiner" in body.get("name", "")
