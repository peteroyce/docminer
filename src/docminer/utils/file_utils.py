"""File utility helpers — type detection, temp files, path helpers."""

from __future__ import annotations

import mimetypes
import os
import tempfile
from pathlib import Path
from typing import Optional

# Map of file extensions to docminer file_type strings
_EXTENSION_MAP: dict[str, str] = {
    ".pdf": "pdf",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".tiff": "image",
    ".tif": "image",
    ".bmp": "image",
    ".webp": "image",
    ".gif": "image",
}

# Magic bytes for file type detection
_MAGIC_BYTES: list[tuple[bytes, str]] = [
    (b"%PDF", "pdf"),
    (b"\x89PNG", "image"),
    (b"\xff\xd8\xff", "image"),   # JPEG
    (b"GIF87a", "image"),
    (b"GIF89a", "image"),
    (b"BM", "image"),             # BMP
    (b"II*\x00", "image"),        # TIFF LE
    (b"MM\x00*", "image"),        # TIFF BE
    (b"RIFF", "image"),           # WebP (container)
]


def detect_file_type(path: str | Path) -> str:
    """Detect the docminer file type string for *path*.

    Uses magic bytes first, falls back to extension.

    Returns
    -------
    str
        One of ``"pdf"``, ``"image"``, ``"scan"`` (unknown).
    """
    path = Path(path)

    # Magic bytes detection
    try:
        with open(path, "rb") as f:
            header = f.read(16)
        for magic, ftype in _MAGIC_BYTES:
            if header.startswith(magic):
                return ftype
    except OSError:
        pass

    # Extension fallback
    suffix = path.suffix.lower()
    return _EXTENSION_MAP.get(suffix, "scan")


def get_mime_type(path: str | Path) -> str:
    """Return the MIME type for the file at *path*."""
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def is_supported(path: str | Path) -> bool:
    """Return True if DocMiner can process this file."""
    return detect_file_type(path) in ("pdf", "image")


def make_temp_copy(data: bytes, suffix: str = ".pdf") -> Path:
    """Write *data* to a temporary file and return the path.

    The caller is responsible for deleting the file when done.
    """
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
    return Path(tmp_path)


def clean_filename(name: str, replacement: str = "_") -> str:
    """Sanitise a filename by replacing non-alphanumeric characters.

    Parameters
    ----------
    name:
        Original filename (without directory).
    replacement:
        Character to substitute for invalid characters.
    """
    import re

    clean = re.sub(r"[^\w\s\-.]", replacement, name)
    clean = re.sub(r"\s+", replacement, clean)
    return clean.strip(replacement)


def ensure_dir(path: str | Path) -> Path:
    """Create *path* as a directory if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def file_size_mb(path: str | Path) -> float:
    """Return the file size in megabytes."""
    return Path(path).stat().st_size / (1024 * 1024)


def list_documents(
    directory: str | Path,
    recursive: bool = False,
    extensions: Optional[set[str]] = None,
) -> list[Path]:
    """List all document files in *directory*.

    Parameters
    ----------
    directory:
        Root directory to scan.
    recursive:
        If True, scan sub-directories.
    extensions:
        Set of extensions to include (e.g. ``{".pdf", ".png"}``).
        If None, all supported extensions are included.
    """
    if extensions is None:
        extensions = set(_EXTENSION_MAP.keys())

    directory = Path(directory)
    glob_fn = directory.rglob if recursive else directory.glob

    paths: list[Path] = []
    for ext in extensions:
        paths.extend(glob_fn(f"*{ext}"))
        paths.extend(glob_fn(f"*{ext.upper()}"))

    return sorted(set(p for p in paths if p.is_file()))
