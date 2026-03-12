"""Entity linker — assign semantic roles to extracted entities."""

from __future__ import annotations

import logging
import re

from docminer.core.types import Entity

logger = logging.getLogger(__name__)

# Context window (characters) to look left of an entity
_CONTEXT_LEFT = 60

# Mapping: (context_keywords, entity_type) -> semantic_role
_LINKING_RULES: list[tuple[list[str], str, str]] = [
    # Invoice fields
    (["invoice date", "date of invoice", "billed date"], "date", "invoice_date"),
    (["due date", "payment due", "due by"], "date", "due_date"),
    (["issue date", "issued on", "issued:"], "date", "issue_date"),
    (["date:", "as of", "dated"], "date", "document_date"),
    (["total", "amount due", "grand total", "balance due"], "amount", "invoice_total"),
    (["subtotal", "sub-total", "net amount"], "amount", "subtotal"),
    (["tax", "vat", "gst", "hst"], "amount", "tax_amount"),
    (["discount"], "amount", "discount_amount"),
    (["unit price", "unit cost", "rate"], "amount", "unit_price"),
    (["invoice number", "invoice no", "inv #", "invoice #"], "reference_number", "invoice_number"),
    (["purchase order", "po number", "p.o."], "reference_number", "po_number"),
    # Contract / letter fields
    (["from:", "sender:", "signed by", "submitted by"], "person", "sender"),
    (["to:", "recipient:", "addressed to", "attention:"], "person", "recipient"),
    (["bill to", "billed to", "ship to"], "organization", "customer"),
    (["vendor", "supplier", "sold by", "from:"], "organization", "vendor"),
    (["effective date", "commencement date", "start date"], "date", "effective_date"),
    (["expiry date", "expiration date", "end date", "termination date"], "date", "expiry_date"),
    # Contact fields
    (["email:", "e-mail:", "contact:"], "email", "contact_email"),
    (["phone:", "tel:", "telephone:", "mobile:"], "phone", "contact_phone"),
    (["address:", "located at", "office:"], "address", "office_address"),
    (["website:", "web:", "url:"], "url", "website"),
    # Report fields
    (["ref:", "reference:", "case number:"], "reference_number", "case_reference"),
    (["author:", "prepared by", "written by"], "person", "author"),
    (["organization:", "company:", "institution:"], "organization", "issuing_org"),
]


class EntityLinker:
    """Augment entities with semantic roles based on surrounding context.

    For each entity, the linker inspects the text immediately preceding
    the entity and attempts to match context keywords.  When a match is
    found the entity's metadata is updated with a ``"role"`` key.
    """

    def link(self, entities: list[Entity], text: str) -> list[Entity]:
        """Assign semantic roles to *entities* in place.

        Parameters
        ----------
        entities:
            List of :class:`Entity` objects previously extracted from *text*.
        text:
            The full source text.

        Returns
        -------
        list[Entity]
            The same list with ``metadata["role"]`` populated where possible.
        """
        for entity in entities:
            role = self._resolve_role(entity, text)
            if role:
                entity.metadata["role"] = role
                logger.debug(
                    "Entity '%s' (%s) linked to role '%s'",
                    entity.text,
                    entity.entity_type,
                    role,
                )
        return entities

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_role(entity: Entity, full_text: str) -> str | None:
        """Return the semantic role for *entity* or *None*."""
        # Extract the context window to the left of the entity
        ctx_start = max(0, entity.start - _CONTEXT_LEFT)
        context_left = full_text[ctx_start: entity.start].lower()

        for keywords, ent_type, role in _LINKING_RULES:
            if entity.entity_type != ent_type:
                continue
            if any(kw in context_left for kw in keywords):
                return role

        # Second pass: proximity to colon-delimited labels on same line
        line_start = full_text.rfind("\n", 0, entity.start) + 1
        line_context = full_text[line_start: entity.start].lower().strip()
        # e.g. "Date: " or "Total Amount:"
        label_match = re.match(r"^([a-z][a-z\s]{1,30}):\s*$", line_context)
        if label_match:
            label = label_match.group(1).strip()
            return f"field:{label}"

        return None
