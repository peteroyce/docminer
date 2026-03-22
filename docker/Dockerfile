# Multi-stage Dockerfile for DocMiner
# Stage 1: builder — install Python deps
# Stage 2: runtime — lean image with Tesseract

# ---- Builder ----
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files needed for install
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

# Install the package into a prefix
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir ".[dev]"


# ---- Runtime ----
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="DocMiner"
LABEL org.opencontainers.image.description="Document intelligence pipeline"
LABEL org.opencontainers.image.version="0.1.0"

# System dependencies: Tesseract OCR + common language packs + MuPDF fonts
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY --from=builder /build/src /app/src
COPY configs/ /app/configs/

WORKDIR /app

ENV PYTHONPATH=/app/src \
    DOCMINER_CONFIG=/app/configs/default.yml \
    PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"

EXPOSE 8000

CMD ["python", "-m", "docminer.cli.main", "serve", "--host", "0.0.0.0", "--port", "8000"]
