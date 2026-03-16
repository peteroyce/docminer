"""Pipeline — chains extractors, classifiers, and analyzers."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from docminer.core.types import Document, ExtractionResult

logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestrate the full document intelligence pipeline.

    Steps (all optional and configurable):
      1. Extraction      — PDF/OCR/image text extraction
      2. Layout          — block type classification and reading order
      3. Classification  — document type detection
      4. Entities        — named entity recognition and linking
      5. Analysis        — summarization and keyword extraction
      6. Output          — structured serialisation

    Parameters
    ----------
    config:
        A :class:`~docminer.config.schema.DocMinerConfig` instance.  If
        *None* the default config is used.
    """

    def __init__(self, config=None) -> None:
        if config is None:
            from docminer.config.schema import DocMinerConfig

            config = DocMinerConfig()
        self.config = config
        self._extractor = None
        self._layout_analyzer = None
        self._classifier = None
        self._entity_recognizer = None
        self._entity_linker = None
        self._summarizer = None
        self._keyword_extractor = None
        self._table_extractor = None

    # ------------------------------------------------------------------
    # Lazy component accessors
    # ------------------------------------------------------------------

    def _get_extractor(self, file_type: str):
        from docminer.extraction import create_extractor

        return create_extractor(file_type, self.config.extraction)

    @property
    def layout_analyzer(self):
        if self._layout_analyzer is None:
            from docminer.layout.analyzer import LayoutAnalyzer

            self._layout_analyzer = LayoutAnalyzer()
        return self._layout_analyzer

    @property
    def classifier(self):
        if self._classifier is None:
            from docminer.classification.classifier import DocumentClassifier

            self._classifier = DocumentClassifier()
        return self._classifier

    @property
    def entity_recognizer(self):
        if self._entity_recognizer is None:
            from docminer.entities.recognizer import EntityRecognizer

            self._entity_recognizer = EntityRecognizer()
        return self._entity_recognizer

    @property
    def entity_linker(self):
        if self._entity_linker is None:
            from docminer.entities.linker import EntityLinker

            self._entity_linker = EntityLinker()
        return self._entity_linker

    @property
    def summarizer(self):
        if self._summarizer is None:
            from docminer.analysis.summarizer import DocumentSummarizer

            self._summarizer = DocumentSummarizer()
        return self._summarizer

    @property
    def keyword_extractor(self):
        if self._keyword_extractor is None:
            from docminer.analysis.keywords import KeywordExtractor

            self._keyword_extractor = KeywordExtractor()
        return self._keyword_extractor

    # ------------------------------------------------------------------
    # Public processing API
    # ------------------------------------------------------------------

    def process_file(self, path: str | Path) -> ExtractionResult:
        """Run the full pipeline on *path* and return an :class:`ExtractionResult`."""
        path = Path(path)
        start = time.perf_counter()
        errors: list[str] = []

        # ---- 1. Determine file type and extract ----
        from docminer.utils.file_utils import detect_file_type

        file_type = detect_file_type(path)
        logger.info("Processing %s as file_type=%s", path.name, file_type)

        try:
            extractor = self._get_extractor(file_type)
            document = extractor.extract_document(path)
        except Exception as exc:
            logger.exception("Extraction failed for %s", path)
            errors.append(f"extraction: {exc}")
            raise

        # ---- 2. Layout analysis ----
        if self.config.pipeline.enable_layout:
            try:
                for page in document.pages:
                    self.layout_analyzer.analyze_page(page)
            except Exception as exc:
                logger.warning("Layout analysis failed: %s", exc)
                errors.append(f"layout: {exc}")

        # ---- 3. Classification ----
        classification = None
        if self.config.pipeline.enable_classification:
            try:
                classification = self.classifier.classify(document)
                logger.info(
                    "Document classified as '%s' (%.2f)",
                    classification.document_type,
                    classification.confidence,
                )
            except Exception as exc:
                logger.warning("Classification failed: %s", exc)
                errors.append(f"classification: {exc}")

        # ---- 4. Entity recognition ----
        entities = []
        if self.config.pipeline.enable_entities:
            try:
                entities = self.entity_recognizer.recognize(document.text)
                if self.config.pipeline.enable_entity_linking:
                    entities = self.entity_linker.link(entities, document.text)
            except Exception as exc:
                logger.warning("Entity recognition failed: %s", exc)
                errors.append(f"entities: {exc}")

        # ---- 5. Analysis ----
        summary: Optional[str] = None
        keywords: list[str] = []
        if self.config.pipeline.enable_analysis:
            try:
                summary = self.summarizer.summarize(
                    document.text,
                    num_sentences=self.config.analysis.summary_sentences,
                )
                keywords = self.keyword_extractor.extract(
                    document.text,
                    top_n=self.config.analysis.top_keywords,
                )
            except Exception as exc:
                logger.warning("Analysis failed: %s", exc)
                errors.append(f"analysis: {exc}")

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("Pipeline completed in %.1f ms", elapsed_ms)

        return ExtractionResult(
            document=document,
            entities=entities,
            tables=document.all_tables,
            classification=classification,
            summary=summary,
            keywords=keywords,
            processing_time_ms=elapsed_ms,
            errors=errors,
        )

    def process_directory(
        self,
        directory: str | Path,
        recursive: bool = False,
        pattern: str = "*",
    ) -> list[ExtractionResult]:
        """Process all documents in *directory*.

        Parameters
        ----------
        directory:
            Root directory to scan.
        recursive:
            If *True*, recurse into sub-directories.
        pattern:
            Glob pattern for file names, e.g. ``"*.pdf"``.
        """
        directory = Path(directory)
        glob_fn = directory.rglob if recursive else directory.glob
        paths = sorted(glob_fn(pattern))
        results: list[ExtractionResult] = []
        for p in paths:
            if not p.is_file():
                continue
            try:
                result = self.process_file(p)
                results.append(result)
            except Exception as exc:
                logger.error("Failed to process %s: %s", p, exc)
        return results
