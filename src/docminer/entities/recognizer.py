"""Entity recognizer — pattern-based NER."""

from __future__ import annotations

import logging
import re
from typing import Optional

from docminer.core.types import Entity
from docminer.entities.patterns import (
    ADDRESS_PATTERN,
    AMOUNT_PATTERNS,
    DATE_PATTERNS,
    EMAIL_PATTERN,
    ORGANIZATION_PATTERN,
    PERSON_PATTERN,
    PHONE_PATTERNS,
    REFERENCE_PATTERNS,
    URL_PATTERN,
)

logger = logging.getLogger(__name__)


class EntityRecognizer:
    """Extract named entities from plain text using compiled regex patterns.

    Entity types extracted:
    - ``date``
    - ``amount``
    - ``email``
    - ``phone``
    - ``address``
    - ``person``
    - ``organization``
    - ``reference_number``
    - ``url``
    """

    def recognize(self, text: str) -> list[Entity]:
        """Extract all entities from *text*.

        Parameters
        ----------
        text:
            The full document text (or any substring).

        Returns
        -------
        list[Entity]
            Deduplicated, sorted list of extracted entities.
        """
        if not text:
            return []

        entities: list[Entity] = []
        entities.extend(self._find_emails(text))
        entities.extend(self._find_urls(text))
        entities.extend(self._find_dates(text))
        entities.extend(self._find_amounts(text))
        entities.extend(self._find_phones(text))
        entities.extend(self._find_addresses(text))
        entities.extend(self._find_references(text))
        entities.extend(self._find_persons(text))
        entities.extend(self._find_organizations(text))

        # Deduplicate by span overlap
        return self._deduplicate(entities)

    # ------------------------------------------------------------------
    # Per-type extractors
    # ------------------------------------------------------------------

    @staticmethod
    def _find_emails(text: str) -> list[Entity]:
        entities: list[Entity] = []
        for m in EMAIL_PATTERN.finditer(text):
            entities.append(
                Entity(
                    text=m.group(1),
                    entity_type="email",
                    start=m.start(),
                    end=m.end(),
                    confidence=0.99,
                    normalized=m.group(1).lower(),
                )
            )
        return entities

    @staticmethod
    def _find_urls(text: str) -> list[Entity]:
        entities: list[Entity] = []
        for m in URL_PATTERN.finditer(text):
            entities.append(
                Entity(
                    text=m.group(),
                    entity_type="url",
                    start=m.start(),
                    end=m.end(),
                    confidence=0.95,
                )
            )
        return entities

    @staticmethod
    def _find_dates(text: str) -> list[Entity]:
        entities: list[Entity] = []
        for pattern in DATE_PATTERNS:
            for m in pattern.finditer(text):
                raw = m.group(1) if m.lastindex and m.lastindex >= 1 else m.group()
                normalized = EntityRecognizer._normalize_date(raw)
                entities.append(
                    Entity(
                        text=raw,
                        entity_type="date",
                        start=m.start(),
                        end=m.end(),
                        confidence=0.90,
                        normalized=normalized,
                    )
                )
        return entities

    @staticmethod
    def _normalize_date(raw: str) -> Optional[str]:
        """Attempt to normalise a date string to ISO 8601 format."""
        import re as _re

        try:
            from datetime import datetime

            formats = [
                "%Y-%m-%d",
                "%m/%d/%Y",
                "%m/%d/%y",
                "%d/%m/%Y",
                "%d.%m.%Y",
                "%B %d, %Y",
                "%B %d %Y",
                "%d %B %Y",
                "%b %d, %Y",
                "%b %d %Y",
            ]
            cleaned = _re.sub(r"\s+", " ", raw.strip().rstrip(","))
            for fmt in formats:
                try:
                    dt = datetime.strptime(cleaned, fmt)
                    return dt.strftime("%Y-%m-%d")
                except ValueError:
                    continue
        except Exception:
            pass
        return None

    @staticmethod
    def _find_amounts(text: str) -> list[Entity]:
        entities: list[Entity] = []
        for pattern in AMOUNT_PATTERNS:
            for m in pattern.finditer(text):
                raw = m.group()
                # Normalise: extract numeric value
                numeric = re.sub(r"[^\d.]", "", raw.replace(",", ""))
                entities.append(
                    Entity(
                        text=raw,
                        entity_type="amount",
                        start=m.start(),
                        end=m.end(),
                        confidence=0.85,
                        normalized=numeric if numeric else None,
                    )
                )
        return entities

    @staticmethod
    def _find_phones(text: str) -> list[Entity]:
        entities: list[Entity] = []
        for pattern in PHONE_PATTERNS:
            for m in pattern.finditer(text):
                raw = m.group()
                digits_only = re.sub(r"\D", "", raw)
                entities.append(
                    Entity(
                        text=raw,
                        entity_type="phone",
                        start=m.start(),
                        end=m.end(),
                        confidence=0.80,
                        normalized=digits_only,
                    )
                )
        return entities

    @staticmethod
    def _find_addresses(text: str) -> list[Entity]:
        entities: list[Entity] = []
        for m in ADDRESS_PATTERN.finditer(text):
            entities.append(
                Entity(
                    text=m.group().strip(),
                    entity_type="address",
                    start=m.start(),
                    end=m.end(),
                    confidence=0.70,
                )
            )
        return entities

    @staticmethod
    def _find_references(text: str) -> list[Entity]:
        entities: list[Entity] = []
        seen: set[str] = set()
        for pattern in REFERENCE_PATTERNS:
            for m in pattern.finditer(text):
                ref_text = m.group(1) if m.lastindex and m.lastindex >= 1 else m.group()
                if ref_text in seen:
                    continue
                seen.add(ref_text)
                entities.append(
                    Entity(
                        text=ref_text,
                        entity_type="reference_number",
                        start=m.start(1) if m.lastindex and m.lastindex >= 1 else m.start(),
                        end=m.end(1) if m.lastindex and m.lastindex >= 1 else m.end(),
                        confidence=0.80,
                        normalized=ref_text.upper(),
                    )
                )
        return entities

    @staticmethod
    def _find_persons(text: str) -> list[Entity]:
        entities: list[Entity] = []
        for m in PERSON_PATTERN.finditer(text):
            name = m.group(1).strip()
            entities.append(
                Entity(
                    text=name,
                    entity_type="person",
                    start=m.start(),
                    end=m.end(),
                    confidence=0.75,
                    normalized=name.title(),
                )
            )
        return entities

    @staticmethod
    def _find_organizations(text: str) -> list[Entity]:
        entities: list[Entity] = []
        for m in ORGANIZATION_PATTERN.finditer(text):
            org_name = m.group(1).strip()
            # Filter very short / generic matches
            if len(org_name) < 5:
                continue
            entities.append(
                Entity(
                    text=org_name,
                    entity_type="organization",
                    start=m.start(),
                    end=m.end(),
                    confidence=0.70,
                )
            )
        return entities

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate(entities: list[Entity]) -> list[Entity]:
        """Remove entities whose span is entirely contained within another entity."""
        if not entities:
            return []

        # Sort by span length descending (keep longest match), then start
        sorted_ents = sorted(entities, key=lambda e: (-(e.end - e.start), e.start))
        kept: list[Entity] = []
        for ent in sorted_ents:
            overlaps = any(
                k.start <= ent.start and k.end >= ent.end and k is not ent
                for k in kept
            )
            if not overlaps:
                kept.append(ent)

        return sorted(kept, key=lambda e: e.start)
