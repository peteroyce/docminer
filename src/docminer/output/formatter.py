"""Output formatter — convert ExtractionResult to JSON, CSV, or Markdown."""

from __future__ import annotations

import csv
import io
import json
import logging
from typing import Literal

from docminer.core.types import Entity, ExtractionResult
from docminer.output.schema import get_output_schema

logger = logging.getLogger(__name__)

OutputFormat = Literal["json", "csv", "markdown"]


class OutputFormatter:
    """Serialise an :class:`ExtractionResult` to various output formats.

    Supported formats: ``"json"``, ``"csv"``, ``"markdown"``.
    """

    def format(
        self,
        result: ExtractionResult,
        fmt: OutputFormat = "json",
        indent: int = 2,
    ) -> str:
        """Format *result* as the requested output type.

        Parameters
        ----------
        result:
            The extraction result to format.
        fmt:
            Output format: ``"json"``, ``"csv"``, or ``"markdown"``.
        indent:
            JSON indentation level (only used for JSON output).

        Returns
        -------
        str
            Serialised output string.
        """
        if fmt == "json":
            return self.to_json(result, indent=indent)
        if fmt == "csv":
            return self.to_csv(result)
        if fmt == "markdown":
            return self.to_markdown(result)
        raise ValueError(f"Unsupported format: {fmt!r}. Use 'json', 'csv', or 'markdown'.")

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def to_json(self, result: ExtractionResult, indent: int = 2) -> str:
        """Serialise to a structured JSON string."""
        doc_type = (
            result.classification.document_type
            if result.classification
            else "unknown"
        )
        schema = get_output_schema(doc_type)
        structured_fields = self._extract_schema_fields(result, schema)

        payload = {
            "document_id": result.document.id,
            "source_path": result.document.source_path,
            "file_type": result.document.file_type,
            "page_count": result.document.page_count,
            "classification": result.classification.to_dict() if result.classification else None,
            "structured_fields": structured_fields,
            "entities": [e.to_dict() for e in result.entities],
            "tables": [t.to_dict() for t in result.tables],
            "summary": result.summary,
            "keywords": result.keywords,
            "processing_time_ms": result.processing_time_ms,
            "errors": result.errors,
        }
        return json.dumps(payload, indent=indent, ensure_ascii=False)

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    def to_csv(self, result: ExtractionResult) -> str:
        """Serialise entities and tables to CSV.

        Returns a multi-section CSV with an entities section followed
        by a tables section for each extracted table.
        """
        buf = io.StringIO()
        writer = csv.writer(buf)

        # Entities section
        writer.writerow(["=== ENTITIES ==="])
        writer.writerow(["type", "text", "normalized", "confidence", "start", "end", "role"])
        for ent in result.entities:
            writer.writerow(
                [
                    ent.entity_type,
                    ent.text,
                    ent.normalized or "",
                    f"{ent.confidence:.2f}",
                    ent.start,
                    ent.end,
                    ent.metadata.get("role", ""),
                ]
            )

        # Tables section
        for i, table in enumerate(result.tables):
            writer.writerow([])
            writer.writerow([f"=== TABLE {i + 1} (page {table.page_num}) ==="])
            if table.headers:
                writer.writerow(table.headers)
            for row in table.rows:
                writer.writerow(row)

        return buf.getvalue()

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------

    def to_markdown(self, result: ExtractionResult) -> str:
        """Serialise to a human-readable Markdown document."""
        doc = result.document
        lines: list[str] = []

        lines.append(f"# Document: {doc.source_path}")
        lines.append("")
        lines.append(f"**Document ID:** `{doc.id}`  ")
        lines.append(f"**File type:** {doc.file_type}  ")
        lines.append(f"**Pages:** {doc.page_count}  ")
        lines.append(f"**Processing time:** {result.processing_time_ms:.1f} ms")
        lines.append("")

        if result.classification:
            cls = result.classification
            lines.append("## Classification")
            lines.append("")
            lines.append(f"**Type:** {cls.document_type}  ")
            lines.append(f"**Confidence:** {cls.confidence:.1%}")
            lines.append("")
            lines.append("| Document Type | Score |")
            lines.append("|---|---|")
            for dt, score in sorted(cls.all_scores.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"| {dt} | {score:.1%} |")
            lines.append("")

        if result.summary:
            lines.append("## Summary")
            lines.append("")
            lines.append(result.summary)
            lines.append("")

        if result.keywords:
            lines.append("## Keywords")
            lines.append("")
            lines.append(", ".join(f"`{kw}`" for kw in result.keywords))
            lines.append("")

        if result.entities:
            lines.append("## Entities")
            lines.append("")
            lines.append("| Type | Text | Normalized | Role | Confidence |")
            lines.append("|---|---|---|---|---|")
            for ent in result.entities:
                role = ent.metadata.get("role", "")
                lines.append(
                    f"| {ent.entity_type} | {ent.text} | {ent.normalized or ''} "
                    f"| {role} | {ent.confidence:.0%} |"
                )
            lines.append("")

        if result.tables:
            lines.append("## Tables")
            for i, table in enumerate(result.tables):
                lines.append(f"### Table {i + 1} (page {table.page_num})")
                lines.append("")
                if table.headers:
                    lines.append("| " + " | ".join(table.headers) + " |")
                    lines.append("|" + "---|" * len(table.headers))
                for row in table.rows:
                    lines.append("| " + " | ".join(str(c) for c in row) + " |")
                lines.append("")

        if doc.metadata:
            lines.append("## Metadata")
            lines.append("")
            for k, v in doc.metadata.items():
                if v:
                    lines.append(f"- **{k}:** {v}")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Schema field extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_schema_fields(result: ExtractionResult, schema: dict) -> dict:
        """Populate schema fields from linked entities."""
        fields: dict[str, list | str | None] = {}
        field_defs = schema.get("fields", {})

        for field_name, field_def in field_defs.items():
            roles = field_def.get("roles", [])
            entity_types = field_def.get("entity_types", [])

            # Find entities matching the roles or entity types
            matches: list[Entity] = []
            for ent in result.entities:
                role = ent.metadata.get("role", "")
                if role in roles and ent.entity_type in entity_types:
                    matches.append(ent)
            # Fall back to entity type matching
            if not matches:
                for ent in result.entities:
                    if ent.entity_type in entity_types:
                        matches.append(ent)

            if not matches:
                fields[field_name] = None
            elif len(matches) == 1:
                fields[field_name] = matches[0].normalized or matches[0].text
            else:
                fields[field_name] = [e.normalized or e.text for e in matches]

        return fields
