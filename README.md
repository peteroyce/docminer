# docminer

A document intelligence pipeline for PDFs, images, and scans. It extracts text and
tables, works out what kind of document it is, pulls out entities such as dates, amounts
and reference numbers, assigns each entity a semantic role, and emits JSON, CSV, or
Markdown. Usable as a library, a CLI, or a REST API.

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)

## Features

- Native PDF extraction via PyMuPDF, keeping bounding boxes and font metadata per text
  block, with Tesseract OCR for scans and images.
- Optional image preprocessing before OCR: deskew by projection-profile angle estimation,
  denoise, contrast enhancement, and binarisation.
- Layout analysis that labels each block as title, header, paragraph, list item, caption,
  or footer, detects one to three column layouts, and reorders blocks into reading order.
- Table extraction from ruling lines, with a whitespace-alignment fallback for borderless
  tables.
- Classification into invoice, contract, resume, report, letter, or form, with per-type
  confidence thresholds; anything below its threshold becomes `unknown`.
- Regex entity recognition — dates, amounts, emails, phones, URLs, addresses, persons,
  organisations, reference numbers — followed by entity linking, which reads the 60
  characters preceding each entity and assigns a role, so an amount after "Total:"
  becomes `invoice_total` and a date after "Due:" becomes `due_date`.
- Extractive summarisation by TextRank over a sentence-similarity graph, and keyword
  extraction that merges TF-IDF and RAKE rankings.
- JSON, CSV, and Markdown output, with a field schema per document type and a generic
  fallback for unclassified documents.
- Optional SQLite persistence through SQLAlchemy, queryable afterwards over the API.

## Architecture

```
PDF · image · scan
   │
   ├─▶ extraction/      PyMuPDF for native text, Tesseract for scans, table.py for grids
   │                    preprocessing/image_prep.py runs first on image input
   ├─▶ layout/          block roles, column detection, reading order
   ├─▶ classification/  TF-IDF + logistic regression, keyword scoring as fallback
   ├─▶ entities/        regex NER → linker.py assigns roles from left context
   ├─▶ analysis/        TextRank summary + TF-IDF/RAKE keywords
   │
   ▼
output/  JSON | CSV | Markdown          storage/  optional SQLite
```

`core/pipeline.py` owns this sequence. Every stage after extraction is switchable in
config, and each is wrapped so that a failure is appended to `result.errors` and the
pipeline continues — a document that defeats the layout analyser still yields entities
and a summary. Only an extraction failure aborts the run, since nothing downstream has
anything to work with. Components are built lazily behind properties: loading
scikit-learn or Tesseract costs real time, and a caller who only wants text extraction
should not pay for either.

## Quickstart

```bash
pip install -e ".[dev]"
```

Tesseract must be installed separately for OCR on scans and images
(`apt-get install tesseract-ocr`, or `brew install tesseract`).

OpenCV is an optional extra (`pip install -e ".[cv]"`). Without it, preprocessing falls
back to PIL equivalents: a median filter instead of a bilateral filter, histogram
equalisation instead of CLAHE, and Otsu thresholding instead of adaptive thresholding.
Classification likewise falls back to rule-based keyword scoring if scikit-learn is
unavailable.

## Usage

### CLI

```bash
docminer extract invoice.pdf                              # JSON to stdout
docminer classify contract.pdf                            # type + all class scores
docminer analyze report.pdf -o markdown --out-file report.md
docminer pipeline ./documents/ -r --pattern "*.pdf" --out-dir ./results/
docminer serve --port 8000
```

`-o/--output` accepts `json`, `csv`, or `markdown`; `-c/--config` takes a YAML file;
`-v/--verbose` raises logging from warnings to debug. `analyze` is `extract` with the
full pipeline enabled.

### Python

```python
from docminer.core.pipeline import Pipeline
from docminer.output.formatter import OutputFormatter

result = Pipeline().process_file("invoice.pdf")

print(result.classification.document_type, result.classification.confidence)

for entity in result.entities:
    print(entity.entity_type, entity.text, entity.metadata.get("role"))

print(result.summary, result.keywords, result.errors)
print(OutputFormatter().to_markdown(result))
```

### REST API

Routes are mounted under `/api/v1`; OpenAPI docs are at `/docs`.

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/extract` | Upload a document, run the pipeline, persist if storage is enabled |
| `POST` | `/api/v1/classify` | Classify only; returns all class scores and elapsed time |
| `POST` | `/api/v1/analyze` | Alias of `/extract` |
| `GET` | `/api/v1/documents` | Paginated list of stored documents (`limit`, `offset`) |
| `GET` | `/api/v1/documents/{id}` | Stored record for one document |
| `GET` | `/api/v1/health` | Version plus availability of Tesseract, PyMuPDF, and scikit-learn |

Uploads are capped at 50 MB and restricted to known document and image extensions; the
supplied filename is reduced to a safe suffix before a temporary file is written, and the
temporary file is removed in a `finally` block. CORS origins come from the
`ALLOWED_ORIGINS` environment variable and default to `http://localhost:3000`.

## Configuration

`configs/default.yml` documents every setting; pass your own with `--config`. The schema
lives in `src/docminer/config/schema.py` and is validated by pydantic, so an out-of-range
DPI or page-segmentation mode is rejected at load time rather than inside Tesseract.

```yaml
extraction:
  ocr: { language: eng+fra, dpi: 400 }
classification:
  confidence_threshold: 0.45
analysis:
  summary_sentences: 8
storage:
  backend: sqlite        # or "none"
  db_path: docminer.db
```

## Tech stack

Python 3.10+ · PyMuPDF · pytesseract · Pillow · scikit-learn · NetworkX · NumPy ·
FastAPI · Uvicorn · SQLAlchemy · Click · pydantic · PyYAML · Hatchling · ruff · pytest

## Testing

```bash
make test        # pytest tests -q
make test-cov    # with coverage
make lint        # ruff check
```

Tests are grouped by subsystem under `tests/` (extraction, layout, classification,
entities, analysis, output, api). CI runs lint, format check, and the suite with coverage
on Python 3.10, 3.11, and 3.12, then builds and smoke-tests the Docker image on `main`
(`.github/workflows/ci.yml`).

## Docker

```bash
make docker-build && make docker-run     # serves on :8000
```

The image is multi-stage: dependencies are installed in a builder layer, and the runtime
layer carries only Python, Tesseract with English data, and the application, with a
healthcheck against `/api/v1/health`.

## License

MIT — see [LICENSE](LICENSE).
