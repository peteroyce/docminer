"""Shared pytest fixtures for DocMiner tests."""

from __future__ import annotations

import io
import struct
import zlib
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Minimal PDF byte generator (no external deps)
# ---------------------------------------------------------------------------

def _make_minimal_pdf(text: str = "Hello World\nThis is a test document.") -> bytes:
    """Create a minimal valid single-page PDF in memory.

    Built from scratch using raw PDF syntax — no reportlab or PyMuPDF required
    in the test environment.
    """
    encoded_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream_content = (
        "BT\n"
        "/F1 12 Tf\n"
        "50 750 Td\n"
        f"({encoded_text}) Tj\n"
        "ET"
    )
    stream_bytes = stream_content.encode("latin-1")

    objects: list[bytes] = []

    # Object 1: Catalog
    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")

    # Object 2: Pages
    objects.append(
        b"2 0 obj\n"
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>\n"
        b"endobj\n"
    )

    # Object 3: Page
    objects.append(
        b"3 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R\n"
        b"   /MediaBox [0 0 612 792]\n"
        b"   /Contents 4 0 R\n"
        b"   /Resources << /Font << /F1 5 0 R >> >> >>\n"
        b"endobj\n"
    )

    # Object 4: Content stream
    stream_len = len(stream_bytes)
    objects.append(
        f"4 0 obj\n<< /Length {stream_len} >>\nstream\n".encode()
        + stream_bytes
        + b"\nendstream\nendobj\n"
    )

    # Object 5: Font
    objects.append(
        b"5 0 obj\n"
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\n"
        b"endobj\n"
    )

    # Build PDF
    header = b"%PDF-1.4\n"
    body = b""
    offsets: list[int] = []
    pos = len(header)
    for obj in objects:
        offsets.append(pos)
        body += obj
        pos += len(obj)

    xref_offset = len(header) + len(body)
    xref = f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        xref += f"{off:010d} 00000 n \n".encode()

    trailer = (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode()

    return header + body + xref + trailer


def _make_invoice_pdf() -> bytes:
    text = (
        "INVOICE\n"
        "Invoice Number: INV-2024-001\n"
        "Invoice Date: 2024-01-15\n"
        "Due Date: 2024-02-15\n"
        "\n"
        "Bill To:\n"
        "Acme Corp\n"
        "123 Main Street, Springfield, IL 62701\n"
        "\n"
        "From:\n"
        "Widget Supplies Inc.\n"
        "vendor@widgetsupplies.com\n"
        "\n"
        "Description    Qty   Unit Price   Total\n"
        "Widget A       10    $10.00       $100.00\n"
        "Widget B       5     $25.00       $125.00\n"
        "\n"
        "Subtotal: $225.00\n"
        "Tax (10%): $22.50\n"
        "Total: $247.50\n"
    )
    return _make_minimal_pdf(text)


def _make_contract_pdf() -> bytes:
    text = (
        "SERVICE AGREEMENT\n"
        "\n"
        "This Agreement is entered into as of January 1, 2024, by and between:\n"
        "Party A: Acme Corporation (hereinafter 'Client')\n"
        "Party B: Tech Solutions LLC (hereinafter 'Provider')\n"
        "\n"
        "WHEREAS, the Provider desires to provide services to the Client;\n"
        "NOW THEREFORE, in consideration of the mutual obligations herein,\n"
        "the parties agree as follows:\n"
        "\n"
        "1. Term. This Agreement shall commence on January 1, 2024 and\n"
        "   terminate on December 31, 2024.\n"
        "\n"
        "2. Governing Law. This Agreement shall be governed by the laws.\n"
        "\n"
        "IN WITNESS WHEREOF, the parties have executed this Agreement.\n"
    )
    return _make_contract_pdf_bytes(text)


def _make_contract_pdf_bytes(text: str) -> bytes:
    return _make_minimal_pdf(text)


# ---------------------------------------------------------------------------
# PIL image generator
# ---------------------------------------------------------------------------

def _make_test_image(width: int = 200, height: int = 100, text: str = "Test") -> "PIL.Image.Image":
    """Create a simple white PIL image with black text."""
    try:
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), text, fill=(0, 0, 0))
        return img
    except ImportError:
        pytest.skip("Pillow not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Minimal valid PDF bytes."""
    return _make_minimal_pdf("This is a sample PDF document for testing.")


@pytest.fixture
def invoice_pdf_bytes() -> bytes:
    """Minimal invoice-like PDF bytes."""
    return _make_invoice_pdf()


@pytest.fixture
def contract_pdf_bytes() -> bytes:
    """Minimal contract-like PDF bytes."""
    return _make_contract_pdf()


@pytest.fixture
def sample_pdf_path(tmp_path: Path, sample_pdf_bytes: bytes) -> Path:
    """Write sample PDF to a temp file and return the path."""
    p = tmp_path / "sample.pdf"
    p.write_bytes(sample_pdf_bytes)
    return p


@pytest.fixture
def invoice_pdf_path(tmp_path: Path, invoice_pdf_bytes: bytes) -> Path:
    p = tmp_path / "invoice.pdf"
    p.write_bytes(invoice_pdf_bytes)
    return p


@pytest.fixture
def contract_pdf_path(tmp_path: Path, contract_pdf_bytes: bytes) -> Path:
    p = tmp_path / "contract.pdf"
    p.write_bytes(contract_pdf_bytes)
    return p


@pytest.fixture
def sample_image_path(tmp_path: Path) -> Path:
    """Create a simple PNG test image."""
    try:
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (400, 200), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), "Sample OCR Test Image", fill=(0, 0, 0))
        draw.text((20, 60), "Invoice Total: $123.45", fill=(0, 0, 0))
        p = tmp_path / "sample.png"
        img.save(str(p))
        return p
    except ImportError:
        pytest.skip("Pillow not installed")


@pytest.fixture
def tmp_storage(tmp_path: Path):
    """Return a SQLiteBackend backed by a temp database."""
    from docminer.storage.backend import SQLiteBackend

    backend = SQLiteBackend(db_path=":memory:")
    yield backend
    backend.close()


@pytest.fixture
def sample_text() -> str:
    return (
        "This is a sample document for testing purposes. "
        "It contains multiple sentences for summarization. "
        "The document includes various entities like dates and amounts. "
        "Contact us at test@example.com or call (555) 123-4567. "
        "The invoice total is $1,234.56 due on 2024-03-15. "
        "Reference number: INV-2024-001."
    )


@pytest.fixture
def invoice_text() -> str:
    return (
        "INVOICE\n"
        "Invoice Number: INV-2024-001\n"
        "Invoice Date: 2024-01-15\n"
        "Due Date: 2024-02-15\n"
        "Bill To: Acme Corp\n"
        "Subtotal: $225.00\n"
        "Tax: $22.50\n"
        "Total: $247.50\n"
        "Contact: billing@vendor.com\n"
        "Phone: (555) 987-6543\n"
    )


@pytest.fixture
def contract_text() -> str:
    return (
        "SERVICE AGREEMENT\n"
        "This agreement is entered into whereas both parties agree.\n"
        "The terms and conditions hereinafter govern this contract.\n"
        "Governing law: State of California.\n"
        "Effective date: January 1, 2024\n"
        "Termination clause applies after 90 days notice.\n"
        "Indemnification and liability provisions are outlined herein.\n"
    )
