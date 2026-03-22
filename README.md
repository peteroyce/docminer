# DocMiner

A document intelligence pipeline that extracts, classifies, and analyzes content from PDFs, images, and scanned documents.

Drop in a PDF of an invoice, contract, or research paper and get back structured data.

```
PDF / Image / Scan
       |
       v
  [Extraction]  ──── PyMuPDF (PDF text) / Tesseract OCR (images/scans)
       |
       v
  [Preprocessing]  ─── text cleaning, image deskew/denoise/binarize
       |
       v
  [Layout Analysis]  ── header/paragraph/list/footer/caption detection,
       |                  reading order, column detection
       v
  [Classification]  ─── TF-IDF + Logistic Regression → invoice / contract /
       |                  resume / report / letter / form / unknown
       v
  [Entity Recognition] ─ dates, amounts, emails, phones, addresses, persons,
       |                  organizations, reference numbers (regex + patterns)
       v
  [Entity Linking]  ─── context-aware role assignment (invoice_total, due_date…)
       |
       v
  [Analysis]  ──────── TextRank summarization, TF-IDF + RAKE keywords
       |
       v
  [Output]  ────────── JSON / CSV / Markdown structured output
       |
       v
  [Storage]  ──────── SQLite (SQLAlchemy) — query results later
```

## Features

- **PDF extraction** — PyMuPDF for native text PDFs with bounding boxes and font metadata
- **OCR** — Tesseract for scanned documents and images; configurable DPI, language, PSM
- **Image preprocessing** — deskew, bilateral filter denoising, CLAHE contrast, adaptive binarization
- **Layout analysis** — classifies blocks as header/paragraph/list/caption/footer, reading order, column detection
- **Document classification** — TF-IDF + Logistic Regression across 6 document types with rule-based fallback
- **Named entity recognition** — regex-based NER: dates, amounts, emails, phones, addresses, persons, organizations, reference numbers
- **Entity linking** — context-aware role assignment (e.g., amount near "Total:" → `invoice_total`)
- **Extractive summarization** — TextRank algorithm (NetworkX PageRank on sentence similarity graph)
- **Keyword extraction** — TF-IDF and RAKE algorithms, merged and ranked
- **Structured output** — type-specific JSON schemas (invoice fields, contract clauses, etc.)
- **REST API** — FastAPI with `POST /extract`, `POST /classify`, `POST /analyze`, `GET /documents`
- **CLI** — Click-based: `extract`, `classify`, `analyze`, `pipeline`, `serve`
- **SQLite storage** — persist results and query them later

## Supported Document Types

| Type | Keywords detected |
|---|---|
| `invoice` | invoice number, total, subtotal, tax, due date, bill to |
| `contract` | whereas, hereinafter, parties, governing law, termination |
| `resume` | education, experience, skills, GPA, references |
| `report` | executive summary, methodology, findings, conclusion |
| `letter` | dear, sincerely, regards, re:, enclosed |
| `form` | fill in, signature, check all that apply, required |

## Quick Start

### Install

```bash
pip install -e ".[dev]"
```

Requires Tesseract OCR to be installed for scanned documents/images:

```bash
# Ubuntu / Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download installer from https://github.com/UB-Mannheim/tesseract/wiki
```

### CLI

```bash
# Extract text and entities from a PDF
docminer extract invoice.pdf

# Classify document type
docminer classify contract.pdf

# Full pipeline with JSON output
docminer analyze report.pdf --output json

# Full pipeline with Markdown output saved to file
docminer analyze report.pdf --output markdown --out-file report.md

# Batch-process a directory of PDFs
docminer pipeline ./documents/ --recursive --pattern "*.pdf" --out-dir ./results/

# Start API server
docminer serve --port 8000
```

### Python API

```python
from docminer.core.pipeline import Pipeline
from docminer.output.formatter import OutputFormatter

pipeline = Pipeline()
result = pipeline.process_file("invoice.pdf")

print(result.classification.document_type)  # "invoice"
print(result.classification.confidence)     # 0.92

for entity in result.entities:
    print(f"{entity.entity_type}: {entity.text} -> {entity.metadata.get('role')}")

print(result.summary)
print(result.keywords)

formatter = OutputFormatter()
print(formatter.to_markdown(result))
```

### REST API

```bash
# Start server
docminer serve

# Extract a document
curl -X POST http://localhost:8000/api/v1/extract \
  -F "file=@invoice.pdf"

# Classify a document
curl -X POST http://localhost:8000/api/v1/classify \
  -F "file=@contract.pdf"

# List processed documents
curl http://localhost:8000/api/v1/documents

# Get document by ID
curl http://localhost:8000/api/v1/documents/abc123def456

# Health check
curl http://localhost:8000/api/v1/health
```

Interactive API docs available at `http://localhost:8000/docs`.

## Configuration

Override settings with a YAML config file:

```yaml
# my_config.yml
extraction:
  preprocess_images: true
  ocr:
    language: eng+fra   # multi-language OCR
    dpi: 400

classification:
  confidence_threshold: 0.45

analysis:
  summary_sentences: 8
  top_keywords: 20

pipeline:
  enable_layout: true
  enable_classification: true
  enable_entities: true
  enable_analysis: true

storage:
  backend: sqlite
  db_path: /data/docminer.db

server:
  port: 8080
  workers: 4
```

```bash
docminer analyze document.pdf --config my_config.yml
```

## CLI Reference

```
docminer [OPTIONS] COMMAND [ARGS]...

Commands:
  extract    Extract text and entities from a file
  classify   Classify the document type
  analyze    Run the full pipeline on a file
  pipeline   Batch-process a directory of documents
  serve      Start the REST API server

Options:
  -v, --verbose  Enable verbose logging
  --version      Show version
  --help         Show help
```

### `docminer extract`

```
docminer extract [OPTIONS] FILE

Arguments:
  FILE  Path to the document (PDF or image)

Options:
  -o, --output [json|csv|markdown]  Output format  [default: json]
  -c, --config PATH                 YAML config file
  --out-file PATH                   Write output to file
```

### `docminer pipeline`

```
docminer pipeline [OPTIONS] DIRECTORY

Arguments:
  DIRECTORY  Directory to scan for documents

Options:
  -r, --recursive                   Include sub-directories
  --pattern TEXT                    Glob pattern  [default: *.pdf]
  -o, --output [json|csv|markdown]  Output format  [default: json]
  --out-dir PATH                    Directory for output files
  -c, --config PATH                 YAML config file
```

## API Reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/extract` | Upload file, get extraction result |
| `POST` | `/api/v1/classify` | Upload file, get classification only |
| `POST` | `/api/v1/analyze` | Run full pipeline (same as extract) |
| `GET` | `/api/v1/documents` | List processed documents |
| `GET` | `/api/v1/documents/{id}` | Get document by ID |
| `GET` | `/api/v1/health` | Service health check |

## Development

```bash
# Install with dev dependencies
make install-dev

# Run tests
make test

# Run tests with coverage
make test-cov

# Lint
make lint

# Format
make format

# Build Docker image
make docker-build
```

## Architecture

```
src/docminer/
├── core/           types.py, pipeline.py
├── extraction/     pdf.py, ocr.py, image.py, table.py
├── layout/         analyzer.py, regions.py, geometry.py
├── classification/ classifier.py, features.py, labels.py
├── entities/       recognizer.py, patterns.py, linker.py
├── analysis/       summarizer.py, keywords.py, similarity.py
├── output/         formatter.py, schema.py
├── storage/        backend.py, models.py
├── preprocessing/  cleaning.py, image_prep.py
├── config/         schema.py, loader.py
├── api/            app.py, routes.py, schemas.py
├── cli/            main.py
└── utils/          file_utils.py, text_utils.py
```

## License

MIT — see [LICENSE](LICENSE).
