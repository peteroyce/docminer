"""Compiled regex patterns for named entity extraction."""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Date patterns
# ---------------------------------------------------------------------------

DATE_PATTERNS: list[re.Pattern] = [
    # ISO 8601: 2024-01-15
    re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),
    # US style: 01/15/2024 or 1/15/24
    re.compile(r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b"),
    # European: 15.01.2024
    re.compile(r"\b(\d{1,2}\.\d{1,2}\.\d{4})\b"),
    # Long form: January 15, 2024 or 15 January 2024
    re.compile(
        r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{1,2},?\s+\d{4})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{4})\b",
        re.IGNORECASE,
    ),
    # Month abbreviations: Jan 15, 2024
    re.compile(
        r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+\d{4})\b",
        re.IGNORECASE,
    ),
]

# ---------------------------------------------------------------------------
# Monetary amount patterns
# ---------------------------------------------------------------------------

AMOUNT_PATTERNS: list[re.Pattern] = [
    # $1,234.56 or $1234.56
    re.compile(r"(\$\s*[\d,]+(?:\.\d{1,2})?)\b"),
    # USD 1,234.56 / EUR 1234.56 etc.
    re.compile(r"\b(USD|EUR|GBP|CAD|AUD|JPY)\s*([\d,]+(?:\.\d{1,2})?)\b"),
    # 1,234.56 USD
    re.compile(r"\b([\d,]+(?:\.\d{1,2})?)\s*(USD|EUR|GBP|CAD|AUD|JPY)\b"),
    # Amount with words: "total: 1,234.56"  (captured inline)
    re.compile(r"(?:total|amount|subtotal|tax|price)\s*[:\s]\s*(\$?[\d,]+(?:\.\d{1,2})?)"),
]

# ---------------------------------------------------------------------------
# Email address pattern
# ---------------------------------------------------------------------------

EMAIL_PATTERN: re.Pattern = re.compile(
    r"\b([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})\b"
)

# ---------------------------------------------------------------------------
# Phone number patterns (US + international)
# ---------------------------------------------------------------------------

PHONE_PATTERNS: list[re.Pattern] = [
    # (123) 456-7890
    re.compile(r"\((\d{3})\)\s*(\d{3})[.\-\s](\d{4})"),
    # 123-456-7890 or 123.456.7890
    re.compile(r"\b(\d{3})[.\-](\d{3})[.\-](\d{4})\b"),
    # +1 123 456 7890
    re.compile(r"\+(\d{1,3})\s(\d{1,4})\s(\d{3,4})\s(\d{4})"),
    # 1-800-123-4567
    re.compile(r"\b1[-.\s](\d{3})[-.\s](\d{3})[-.\s](\d{4})\b"),
]

# ---------------------------------------------------------------------------
# URL / web address
# ---------------------------------------------------------------------------

URL_PATTERN: re.Pattern = re.compile(
    r"\bhttps?://[^\s<>\"{}|\\^`\[\]]+"
    r"|www\.[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}[^\s]*"
)

# ---------------------------------------------------------------------------
# Postal address (heuristic multi-line)
# ---------------------------------------------------------------------------

ADDRESS_PATTERN: re.Pattern = re.compile(
    r"\b\d+\s+[A-Za-z0-9\s,\.]+(?:Street|St|Avenue|Ave|Boulevard|Blvd|Drive|Dr|Lane|Ln|Road|Rd"
    r"|Way|Court|Ct|Place|Pl|Suite|Ste|Floor|Fl)\b[^\n]*",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Reference / invoice numbers
# ---------------------------------------------------------------------------

REFERENCE_PATTERNS: list[re.Pattern] = [
    # Invoice #: INV-12345
    re.compile(r"(?:invoice|inv)[\s#\-]*:?\s*([A-Z0-9\-]{4,20})", re.IGNORECASE),
    # PO Number: PO-98765
    re.compile(r"(?:p\.?o\.?|purchase order)[\s#\-]*:?\s*([A-Z0-9\-]{4,20})", re.IGNORECASE),
    # Reference: REF-2024-001
    re.compile(r"(?:ref(?:erence)?|case)[\s#\-]*:?\s*([A-Z0-9\-]{4,20})", re.IGNORECASE),
    # Generic document number: #12345 or No. 12345
    re.compile(r"(?:no\.|number|#)\s*([A-Z]?\d{4,12})", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Person names (heuristic: title + capitalized words)
# ---------------------------------------------------------------------------

PERSON_PATTERN: re.Pattern = re.compile(
    r"\b((?:Mr|Mrs|Ms|Dr|Prof|Rev|Sir|Lady|Lord)\.?\s+"
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b"
)

# ---------------------------------------------------------------------------
# Organization names
# ---------------------------------------------------------------------------

ORGANIZATION_PATTERN: re.Pattern = re.compile(
    r"\b([A-Z][a-zA-Z&\s,\.]+(?:Inc\.|LLC|Ltd\.|Corp\.|Co\.|GmbH|PLC|LLP|LP|AG))\b"
)
