"""Tests for document classification."""

from __future__ import annotations

import pytest

from docminer.classification.classifier import DocumentClassifier
from docminer.classification.features import FeatureExtractor
from docminer.core.types import ClassificationResult, Document, Page, TextBlock


def make_document(text: str, file_type: str = "pdf") -> Document:
    from docminer.extraction.base import BaseExtractor

    doc_id = BaseExtractor.make_document_id(text)
    block = TextBlock(text=text, page_num=1)
    page = Page(number=1, width=612, height=792, blocks=[block])
    return Document(
        id=doc_id,
        source_path="test.pdf",
        file_type=file_type,
        pages=[page],
        text=text,
    )


class TestDocumentClassifier:
    def test_classify_returns_result(self, invoice_text: str) -> None:
        classifier = DocumentClassifier()
        doc = make_document(invoice_text)
        result = classifier.classify(doc)
        assert isinstance(result, ClassificationResult)
        assert result.document_type in (
            "invoice", "contract", "resume", "report", "letter", "form", "unknown"
        )
        assert 0.0 <= result.confidence <= 1.0

    def test_classify_invoice(self, invoice_text: str) -> None:
        classifier = DocumentClassifier()
        doc = make_document(invoice_text)
        result = classifier.classify(doc)
        # Invoice keywords should dominate
        assert result.document_type == "invoice"

    def test_classify_contract(self, contract_text: str) -> None:
        classifier = DocumentClassifier()
        doc = make_document(contract_text)
        result = classifier.classify(doc)
        assert result.document_type == "contract"

    def test_classify_empty_text(self) -> None:
        classifier = DocumentClassifier()
        doc = make_document("")
        result = classifier.classify(doc)
        assert result.document_type == "unknown"
        assert result.confidence == 0.0

    def test_all_scores_sum_to_reasonable_value(self, invoice_text: str) -> None:
        classifier = DocumentClassifier()
        doc = make_document(invoice_text)
        result = classifier.classify(doc)
        assert result.all_scores  # non-empty
        for score in result.all_scores.values():
            assert 0.0 <= score <= 1.0

    def test_features_used_populated(self, invoice_text: str) -> None:
        classifier = DocumentClassifier()
        doc = make_document(invoice_text)
        result = classifier.classify(doc)
        assert len(result.features_used) > 0

    def test_classify_text_directly(self) -> None:
        classifier = DocumentClassifier()
        invoice_text = "invoice total amount due payment subtotal tax"
        result = classifier.classify_text(invoice_text)
        assert isinstance(result, ClassificationResult)

    def test_resume_classification(self) -> None:
        classifier = DocumentClassifier()
        resume_text = (
            "Resume\n"
            "Education: Bachelor of Science in Computer Science\n"
            "Work Experience: Software Engineer at Tech Corp\n"
            "Skills: Python, Java, SQL, Machine Learning\n"
            "GPA: 3.8, University of Technology\n"
            "References available upon request\n"
        )
        doc = make_document(resume_text)
        result = classifier.classify(doc)
        assert result.document_type == "resume"


class TestFeatureExtractor:
    def test_extract_returns_dict(self, invoice_text: str) -> None:
        extractor = FeatureExtractor()
        doc = make_document(invoice_text)
        features = extractor.extract(doc)
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_keyword_features_present(self, invoice_text: str) -> None:
        extractor = FeatureExtractor()
        doc = make_document(invoice_text)
        features = extractor.extract(doc)
        assert "keyword_invoice_hits" in features
        assert features["keyword_invoice_hits"] > 0

    def test_structural_features_present(self, invoice_text: str) -> None:
        extractor = FeatureExtractor()
        doc = make_document(invoice_text)
        features = extractor.extract(doc)
        assert "struct_page_count" in features
        assert "struct_has_numbers" in features
        assert features["struct_has_numbers"] == 1.0

    def test_metadata_features(self) -> None:
        extractor = FeatureExtractor()
        doc = make_document("test text")
        doc.metadata = {"title": "Annual Report 2024"}
        features = extractor.extract(doc)
        assert "meta_report_title" in features
        assert features["meta_report_title"] == 1.0
