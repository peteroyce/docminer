"""Output schemas for different document types."""

from __future__ import annotations

# Schema definitions describe the expected structured output fields
# per document type.  Each field has: description, entity_roles that
# map to it, and whether it is required.

INVOICE_SCHEMA: dict = {
    "document_type": "invoice",
    "fields": {
        "vendor": {
            "description": "Name of the issuing company or person",
            "roles": ["vendor"],
            "entity_types": ["organization", "person"],
            "required": False,
        },
        "customer": {
            "description": "Name of the billed party",
            "roles": ["customer"],
            "entity_types": ["organization", "person"],
            "required": False,
        },
        "invoice_number": {
            "description": "Unique invoice identifier",
            "roles": ["invoice_number"],
            "entity_types": ["reference_number"],
            "required": False,
        },
        "invoice_date": {
            "description": "Date the invoice was issued",
            "roles": ["invoice_date", "document_date"],
            "entity_types": ["date"],
            "required": False,
        },
        "due_date": {
            "description": "Payment due date",
            "roles": ["due_date"],
            "entity_types": ["date"],
            "required": False,
        },
        "subtotal": {
            "description": "Amount before tax",
            "roles": ["subtotal"],
            "entity_types": ["amount"],
            "required": False,
        },
        "tax": {
            "description": "Tax amount",
            "roles": ["tax_amount"],
            "entity_types": ["amount"],
            "required": False,
        },
        "total": {
            "description": "Total amount due",
            "roles": ["invoice_total"],
            "entity_types": ["amount"],
            "required": False,
        },
        "po_number": {
            "description": "Purchase order number",
            "roles": ["po_number"],
            "entity_types": ["reference_number"],
            "required": False,
        },
        "contact_email": {
            "description": "Vendor contact email",
            "roles": ["contact_email"],
            "entity_types": ["email"],
            "required": False,
        },
        "contact_phone": {
            "description": "Vendor contact phone",
            "roles": ["contact_phone"],
            "entity_types": ["phone"],
            "required": False,
        },
    },
}

CONTRACT_SCHEMA: dict = {
    "document_type": "contract",
    "fields": {
        "party_a": {
            "description": "First contracting party",
            "roles": ["sender"],
            "entity_types": ["organization", "person"],
            "required": False,
        },
        "party_b": {
            "description": "Second contracting party",
            "roles": ["recipient"],
            "entity_types": ["organization", "person"],
            "required": False,
        },
        "effective_date": {
            "description": "Date the contract takes effect",
            "roles": ["effective_date"],
            "entity_types": ["date"],
            "required": False,
        },
        "expiry_date": {
            "description": "Contract expiration date",
            "roles": ["expiry_date"],
            "entity_types": ["date"],
            "required": False,
        },
        "reference": {
            "description": "Contract reference number",
            "roles": ["case_reference"],
            "entity_types": ["reference_number"],
            "required": False,
        },
    },
}

RESUME_SCHEMA: dict = {
    "document_type": "resume",
    "fields": {
        "candidate_name": {
            "description": "Name of the candidate",
            "roles": ["sender"],
            "entity_types": ["person"],
            "required": False,
        },
        "email": {
            "description": "Candidate email address",
            "roles": ["contact_email"],
            "entity_types": ["email"],
            "required": False,
        },
        "phone": {
            "description": "Candidate phone number",
            "roles": ["contact_phone"],
            "entity_types": ["phone"],
            "required": False,
        },
        "address": {
            "description": "Candidate address",
            "roles": ["office_address"],
            "entity_types": ["address"],
            "required": False,
        },
        "website": {
            "description": "Personal website or LinkedIn URL",
            "roles": ["website"],
            "entity_types": ["url"],
            "required": False,
        },
    },
}

REPORT_SCHEMA: dict = {
    "document_type": "report",
    "fields": {
        "author": {
            "description": "Report author(s)",
            "roles": ["author"],
            "entity_types": ["person"],
            "required": False,
        },
        "issuing_organization": {
            "description": "Organization that issued the report",
            "roles": ["issuing_org"],
            "entity_types": ["organization"],
            "required": False,
        },
        "report_date": {
            "description": "Date the report was issued",
            "roles": ["document_date"],
            "entity_types": ["date"],
            "required": False,
        },
        "reference": {
            "description": "Report reference or case number",
            "roles": ["case_reference"],
            "entity_types": ["reference_number"],
            "required": False,
        },
    },
}

LETTER_SCHEMA: dict = {
    "document_type": "letter",
    "fields": {
        "sender": {
            "description": "Letter sender",
            "roles": ["sender"],
            "entity_types": ["person", "organization"],
            "required": False,
        },
        "recipient": {
            "description": "Letter recipient",
            "roles": ["recipient"],
            "entity_types": ["person", "organization"],
            "required": False,
        },
        "date": {
            "description": "Date of the letter",
            "roles": ["document_date"],
            "entity_types": ["date"],
            "required": False,
        },
        "subject": {
            "description": "Subject line",
            "roles": [],
            "entity_types": [],
            "required": False,
        },
    },
}

FORM_SCHEMA: dict = {
    "document_type": "form",
    "fields": {
        "submitter": {
            "description": "Person who filled out the form",
            "roles": ["sender"],
            "entity_types": ["person"],
            "required": False,
        },
        "date": {
            "description": "Date the form was completed",
            "roles": ["document_date"],
            "entity_types": ["date"],
            "required": False,
        },
        "reference": {
            "description": "Form reference number",
            "roles": ["case_reference"],
            "entity_types": ["reference_number"],
            "required": False,
        },
    },
}

GENERIC_SCHEMA: dict = {
    "document_type": "unknown",
    "fields": {},
}

_SCHEMA_MAP: dict[str, dict] = {
    "invoice": INVOICE_SCHEMA,
    "contract": CONTRACT_SCHEMA,
    "resume": RESUME_SCHEMA,
    "report": REPORT_SCHEMA,
    "letter": LETTER_SCHEMA,
    "form": FORM_SCHEMA,
    "unknown": GENERIC_SCHEMA,
}


def get_output_schema(document_type: str) -> dict:
    """Return the output schema for the given *document_type*."""
    return _SCHEMA_MAP.get(document_type, GENERIC_SCHEMA)
