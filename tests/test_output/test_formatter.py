"""Tests for output formatting."""

from __future__ import annotations

import csv
import io
import json

import pytest

from docminer.core.types import (
    BoundingBox,
    ClassificationResult,
    Document,
    Entity,
    ExtractionResult,
    Page,
    Table,
    TextBlock,
)
from docminer.output.formatter import OutputFormatter
from docminer.output.schema import get_output_schema


def make_extraction_result(
    text: str = "Sample document text",
    doc_type: str = "invoice",
) -> ExtractionResult:
    block = TextBlock(text=text, page_num=1)
    page = Page(number=1, width=612, height=792, blocks=[block])
    doc = Document(
        id="test123",
        source_path="/tmp/test.pdf",
        file_type="pdf",
        pages=[page],
        text=text,
    )
    classification = ClassificationResult(
        document_type=doc_type,
        confidence=0.92,
        all_scores={"invoice": 0.92, "contract": 0.05, "unknown": 0.03},
        features_used=["tfidf"],
    )
    entities = [
        Entity(
            text="test@example.com",
            entity_type="email",
            start=0,
            end=16,
            confidence=0.99,
            normalized="test@example.com",
            metadata={"role": "contact_email"},
        ),
        Entity(
            text="$247.50",
            entity_type="amount",
            start=20,
            end=27,
            confidence=0.88,
            normalized="247.50",
            metadata={"role": "invoice_total"},
        ),
        Entity(
            text="2024-01-15",
            entity_type="date",
            start=30,
            end=40,
            confidence=0.90,
            normalized="2024-01-15",
            metadata={"role": "invoice_date"},
        ),
    ]
    tables = [
        Table(
            rows=[["Widget A", "10", "$10.00"], ["Widget B", "5", "$25.00"]],
            headers=["Item", "Qty", "Price"],
            page_num=1,
        )
    ]
    return ExtractionResult(
        document=doc,
        entities=entities,
        tables=tables,
        classification=classification,
        summary="This is the summary of the document.",
        keywords=["invoice", "payment", "total"],
        processing_time_ms=123.45,
    )


class TestOutputFormatter:
    def test_json_output_is_valid(self) -> None:
        formatter = OutputFormatter()
        result = make_extraction_result()
        output = formatter.format(result, fmt="json")
        parsed = json.loads(output)
        assert isinstance(parsed, dict)
        assert parsed["document_id"] == "test123"

    def test_json_has_all_top_level_keys(self) -> None:
        formatter = OutputFormatter()
        result = make_extraction_result()
        parsed = json.loads(formatter.to_json(result))
        for key in ("document_id", "classification", "entities", "tables", "summary", "keywords"):
            assert key in parsed

    def test_json_entities_present(self) -> None:
        formatter = OutputFormatter()
        result = make_extraction_result()
        parsed = json.loads(formatter.to_json(result))
        assert len(parsed["entities"]) == 3
        types = {e["entity_type"] for e in parsed["entities"]}
        assert "email" in types
        assert "amount" in types

    def test_json_structured_fields_invoice(self) -> None:
        formatter = OutputFormatter()
        result = make_extraction_result(doc_type="invoice")
        parsed = json.loads(formatter.to_json(result))
        fields = parsed.get("structured_fields", {})
        assert "total" in fields or "contact_email" in fields

    def test_csv_output_is_parseable(self) -> None:
        formatter = OutputFormatter()
        result = make_extraction_result()
        output = formatter.to_csv(result)
        reader = csv.reader(io.StringIO(output))
        rows = list(reader)
        assert len(rows) > 0

    def test_csv_entities_section_present(self) -> None:
        formatter = OutputFormatter()
        result = make_extraction_result()
        output = formatter.to_csv(result)
        assert "ENTITIES" in output
        assert "email" in output

    def test_csv_table_section_present(self) -> None:
        formatter = OutputFormatter()
        result = make_extraction_result()
        output = formatter.to_csv(result)
        assert "TABLE" in output
        assert "Widget A" in output

    def test_markdown_output_has_headers(self) -> None:
        formatter = OutputFormatter()
        result = make_extraction_result()
        output = formatter.to_markdown(result)
        assert "# Document" in output
        assert "## Classification" in output
        assert "## Entities" in output
        assert "## Summary" in output
        assert "## Keywords" in output
        assert "## Tables" in output

    def test_markdown_contains_entity_data(self) -> None:
        formatter = OutputFormatter()
        result = make_extraction_result()
        output = formatter.to_markdown(result)
        assert "test@example.com" in output
        assert "$247.50" in output

    def test_markdown_contains_table_data(self) -> None:
        formatter = OutputFormatter()
        result = make_extraction_result()
        output = formatter.to_markdown(result)
        assert "Widget A" in output
        assert "Widget B" in output

    def test_invalid_format_raises(self) -> None:
        formatter = OutputFormatter()
        result = make_extraction_result()
        with pytest.raises(ValueError, match="Unsupported format"):
            formatter.format(result, fmt="xml")  # type: ignore[arg-type]

    def test_no_classification_result(self) -> None:
        formatter = OutputFormatter()
        block = TextBlock(text="Some text", page_num=1)
        page = Page(number=1, width=612, height=792, blocks=[block])
        doc = Document(id="abc", source_path="x.pdf", file_type="pdf", pages=[page], text="Some text")
        result = ExtractionResult(document=doc)
        output = formatter.to_json(result)
        parsed = json.loads(output)
        assert parsed["classification"] is None


class TestOutputSchema:
    def test_invoice_schema(self) -> None:
        schema = get_output_schema("invoice")
        assert schema["document_type"] == "invoice"
        assert "total" in schema["fields"]
        assert "invoice_number" in schema["fields"]

    def test_contract_schema(self) -> None:
        schema = get_output_schema("contract")
        assert schema["document_type"] == "contract"
        assert "effective_date" in schema["fields"]

    def test_resume_schema(self) -> None:
        schema = get_output_schema("resume")
        assert "email" in schema["fields"]

    def test_unknown_schema(self) -> None:
        schema = get_output_schema("unknown")
        assert schema["document_type"] == "unknown"

    def test_missing_type_returns_generic(self) -> None:
        schema = get_output_schema("nonexistent_type")
        assert schema["document_type"] == "unknown"
