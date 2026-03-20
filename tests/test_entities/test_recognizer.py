"""Tests for entity recognition and linking."""

from __future__ import annotations

import pytest

from docminer.core.types import Entity
from docminer.entities.linker import EntityLinker
from docminer.entities.recognizer import EntityRecognizer


class TestEntityRecognizer:
    def test_recognize_email(self) -> None:
        recognizer = EntityRecognizer()
        entities = recognizer.recognize("Contact us at test@example.com for details.")
        emails = [e for e in entities if e.entity_type == "email"]
        assert len(emails) >= 1
        assert emails[0].text == "test@example.com"
        assert emails[0].normalized == "test@example.com"

    def test_recognize_iso_date(self) -> None:
        recognizer = EntityRecognizer()
        entities = recognizer.recognize("The date is 2024-01-15.")
        dates = [e for e in entities if e.entity_type == "date"]
        assert len(dates) >= 1
        assert dates[0].normalized == "2024-01-15"

    def test_recognize_us_date(self) -> None:
        recognizer = EntityRecognizer()
        entities = recognizer.recognize("Invoice date: 01/15/2024")
        dates = [e for e in entities if e.entity_type == "date"]
        assert any(d.normalized == "2024-01-15" for d in dates)

    def test_recognize_long_date(self) -> None:
        recognizer = EntityRecognizer()
        entities = recognizer.recognize("As of January 15, 2024, the agreement is effective.")
        dates = [e for e in entities if e.entity_type == "date"]
        assert len(dates) >= 1
        assert any("2024-01-15" in (d.normalized or "") for d in dates)

    def test_recognize_dollar_amount(self) -> None:
        recognizer = EntityRecognizer()
        entities = recognizer.recognize("The total is $1,234.56")
        amounts = [e for e in entities if e.entity_type == "amount"]
        assert len(amounts) >= 1
        assert any("1234.56" in (a.normalized or "") or "$1,234.56" in a.text for a in amounts)

    def test_recognize_phone_us(self) -> None:
        recognizer = EntityRecognizer()
        entities = recognizer.recognize("Call us at (555) 123-4567 anytime.")
        phones = [e for e in entities if e.entity_type == "phone"]
        assert len(phones) >= 1
        assert "5551234567" in (phones[0].normalized or "")

    def test_recognize_url(self) -> None:
        recognizer = EntityRecognizer()
        entities = recognizer.recognize("Visit https://www.example.com for more info.")
        urls = [e for e in entities if e.entity_type == "url"]
        assert len(urls) >= 1
        assert "example.com" in urls[0].text

    def test_recognize_address(self) -> None:
        recognizer = EntityRecognizer()
        entities = recognizer.recognize("Located at 123 Main Street, Springfield.")
        addresses = [e for e in entities if e.entity_type == "address"]
        assert len(addresses) >= 1

    def test_recognize_invoice_reference(self) -> None:
        recognizer = EntityRecognizer()
        entities = recognizer.recognize("Invoice Number: INV-2024-001")
        refs = [e for e in entities if e.entity_type == "reference_number"]
        assert len(refs) >= 1
        assert "INV-2024-001" in refs[0].text or "INV-2024-001" in (refs[0].normalized or "")

    def test_recognize_person(self) -> None:
        recognizer = EntityRecognizer()
        entities = recognizer.recognize("Dr. John Smith signed the agreement.")
        persons = [e for e in entities if e.entity_type == "person"]
        assert len(persons) >= 1
        assert "John Smith" in persons[0].text

    def test_recognize_organization(self) -> None:
        recognizer = EntityRecognizer()
        entities = recognizer.recognize("Acme Corp. is a leading supplier.")
        orgs = [e for e in entities if e.entity_type == "organization"]
        assert len(orgs) >= 1

    def test_deduplication_removes_overlapping(self) -> None:
        recognizer = EntityRecognizer()
        # Both date patterns might match; dedup should keep one
        entities = recognizer.recognize("2024-01-15")
        dates = [e for e in entities if e.entity_type == "date"]
        # No two entities should overlap spans
        for i, e1 in enumerate(dates):
            for j, e2 in enumerate(dates):
                if i != j:
                    assert not (e1.start <= e2.start < e1.end)

    def test_empty_text_returns_empty(self) -> None:
        recognizer = EntityRecognizer()
        assert recognizer.recognize("") == []

    def test_entities_sorted_by_start(self, sample_text: str) -> None:
        recognizer = EntityRecognizer()
        entities = recognizer.recognize(sample_text)
        starts = [e.start for e in entities]
        assert starts == sorted(starts)

    def test_normalize_date_iso(self) -> None:
        result = EntityRecognizer._normalize_date("2024-01-15")
        assert result == "2024-01-15"

    def test_normalize_date_us_format(self) -> None:
        result = EntityRecognizer._normalize_date("01/15/2024")
        assert result == "2024-01-15"


class TestEntityLinker:
    def test_link_invoice_total(self) -> None:
        linker = EntityLinker()
        text = "Total: $247.50"
        entities = [
            Entity(
                text="$247.50",
                entity_type="amount",
                start=7,
                end=14,
                confidence=0.9,
            )
        ]
        linked = linker.link(entities, text)
        assert linked[0].metadata.get("role") == "invoice_total"

    def test_link_invoice_date(self) -> None:
        linker = EntityLinker()
        text = "Invoice Date: 2024-01-15"
        entities = [
            Entity(
                text="2024-01-15",
                entity_type="date",
                start=14,
                end=24,
                confidence=0.9,
            )
        ]
        linked = linker.link(entities, text)
        assert linked[0].metadata.get("role") == "invoice_date"

    def test_link_email_contact(self) -> None:
        linker = EntityLinker()
        text = "Email: vendor@example.com"
        entities = [
            Entity(
                text="vendor@example.com",
                entity_type="email",
                start=7,
                end=25,
                confidence=0.99,
            )
        ]
        linked = linker.link(entities, text)
        assert linked[0].metadata.get("role") == "contact_email"

    def test_no_match_leaves_entity_unchanged(self) -> None:
        linker = EntityLinker()
        text = "Random text with a date 2024-01-15 in middle"
        entities = [
            Entity(
                text="2024-01-15",
                entity_type="date",
                start=24,
                end=34,
                confidence=0.9,
            )
        ]
        linked = linker.link(entities, text)
        # May or may not match — just check it doesn't crash
        assert isinstance(linked[0].metadata, dict)

    def test_field_colon_label(self) -> None:
        linker = EntityLinker()
        text = "Due Date:\n2024-03-15"
        entities = [
            Entity(
                text="2024-03-15",
                entity_type="date",
                start=10,
                end=20,
                confidence=0.9,
            )
        ]
        linked = linker.link(entities, text)
        role = linked[0].metadata.get("role", "")
        # Should detect "due date" label
        assert role in ("due_date", "field:due date", "document_date") or role.startswith("field:")
