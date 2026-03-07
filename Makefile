.PHONY: install install-dev test test-cov lint format clean serve docker-build help

PYTHON := python
PIP    := pip
SRC    := src/docminer
TESTS  := tests

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

test:
	pytest $(TESTS) -q

test-cov:
	pytest $(TESTS) --cov=$(SRC) --cov-report=term-missing --cov-report=html

test-fast:
	pytest $(TESTS) -q -x --no-header

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------

lint:
	ruff check $(SRC) $(TESTS)

format:
	ruff format $(SRC) $(TESTS)

format-check:
	ruff format --check $(SRC) $(TESTS)

check: lint format-check

# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

serve:
	$(PYTHON) -m docminer.cli.main serve --reload

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

docker-build:
	docker build -f docker/Dockerfile -t docminer:latest .

docker-run:
	docker run --rm -p 8000:8000 docminer:latest

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

clean:
	rm -rf build/ dist/ *.egg-info/ .eggs/ .pytest_cache/ .ruff_cache/ htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

help:
	@echo "DocMiner Makefile targets:"
	@echo ""
	@echo "  install        Install package in editable mode"
	@echo "  install-dev    Install with dev dependencies"
	@echo "  test           Run test suite"
	@echo "  test-cov       Run tests with coverage report"
	@echo "  lint           Run ruff linter"
	@echo "  format         Auto-format with ruff"
	@echo "  format-check   Check formatting without modifying files"
	@echo "  check          Run lint + format check"
	@echo "  serve          Start API server with hot-reload"
	@echo "  docker-build   Build Docker image"
	@echo "  docker-run     Run Docker container on port 8000"
	@echo "  clean          Remove build artifacts and caches"
