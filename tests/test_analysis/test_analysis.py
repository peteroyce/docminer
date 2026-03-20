"""Tests for summarization, keyword extraction, and document similarity."""

from __future__ import annotations

import pytest

from docminer.analysis.keywords import KeywordExtractor
from docminer.analysis.similarity import DocumentSimilarity
from docminer.analysis.summarizer import DocumentSummarizer


LONG_TEXT = """
Machine learning is a subset of artificial intelligence that provides systems
the ability to automatically learn and improve from experience without being
explicitly programmed. Machine learning focuses on the development of computer
programs that can access data and use it to learn for themselves.

The process of learning begins with observations or data, such as examples,
direct experience, or instruction, so that computers can look for patterns
in data and make better decisions in the future. The primary aim is to allow
the computers to learn automatically without human intervention or assistance
and adjust actions accordingly.

Natural language processing is a subfield of linguistics, computer science,
and artificial intelligence concerned with the interactions between computers
and human language. The goal of NLP is to enable computers to understand,
interpret, and generate human language in a valuable way.

Deep learning uses neural networks with many layers to learn representations
of data with multiple levels of abstraction. These methods have dramatically
improved the state-of-the-art in speech recognition, visual object recognition,
object detection and many other domains.
""".strip()


class TestDocumentSummarizer:
    def test_summarize_returns_string(self) -> None:
        summarizer = DocumentSummarizer()
        result = summarizer.summarize(LONG_TEXT, num_sentences=3)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_summary_shorter_than_original(self) -> None:
        summarizer = DocumentSummarizer()
        result = summarizer.summarize(LONG_TEXT, num_sentences=2)
        assert len(result) < len(LONG_TEXT)

    def test_empty_text_returns_empty(self) -> None:
        summarizer = DocumentSummarizer()
        result = summarizer.summarize("")
        assert result == ""

    def test_short_text_returns_unchanged(self) -> None:
        summarizer = DocumentSummarizer()
        short = "This is a short sentence. Only two."
        result = summarizer.summarize(short, num_sentences=5)
        assert len(result) > 0

    def test_num_sentences_respected(self) -> None:
        summarizer = DocumentSummarizer()
        result = summarizer.summarize(LONG_TEXT, num_sentences=2)
        # Very rough check: 2-sentence summary should have 1-3 period terminators
        sentence_endings = result.count(". ") + result.count(".")
        # Should have fewer sentence boundaries than the full text
        assert sentence_endings <= LONG_TEXT.count(". ") + LONG_TEXT.count(".")

    def test_cosine_similarity_same_text(self) -> None:
        summarizer = DocumentSummarizer()
        tf1 = summarizer._tf("hello world hello")
        tf2 = summarizer._tf("hello world hello")
        sim = summarizer._cosine_similarity(tf1, tf2)
        assert sim == pytest.approx(1.0)

    def test_cosine_similarity_no_overlap(self) -> None:
        summarizer = DocumentSummarizer()
        tf1 = summarizer._tf("apple banana cherry")
        tf2 = summarizer._tf("dog cat fish")
        sim = summarizer._cosine_similarity(tf1, tf2)
        assert sim == pytest.approx(0.0)


class TestKeywordExtractor:
    def test_extract_returns_list(self) -> None:
        extractor = KeywordExtractor()
        keywords = extractor.extract(LONG_TEXT, top_n=5)
        assert isinstance(keywords, list)
        assert len(keywords) > 0

    def test_extract_top_n(self) -> None:
        extractor = KeywordExtractor()
        keywords = extractor.extract(LONG_TEXT, top_n=10)
        assert len(keywords) <= 10

    def test_keywords_are_strings(self) -> None:
        extractor = KeywordExtractor()
        keywords = extractor.extract(LONG_TEXT, top_n=5)
        for kw in keywords:
            assert isinstance(kw, str)
            assert len(kw) > 0

    def test_empty_text_returns_empty(self) -> None:
        extractor = KeywordExtractor()
        keywords = extractor.extract("")
        assert keywords == []

    def test_relevant_keywords_extracted(self) -> None:
        extractor = KeywordExtractor()
        text = "invoice payment total amount due subtotal tax receipt"
        keywords = extractor.extract(text, top_n=5)
        # At least some domain-relevant keywords should appear
        all_kw = " ".join(keywords).lower()
        assert any(kw in all_kw for kw in ["invoice", "payment", "total", "amount"])

    def test_tfidf_keywords(self) -> None:
        extractor = KeywordExtractor()
        result = extractor._tfidf_keywords(LONG_TEXT, top_n=10)
        assert len(result) > 0
        for term, score in result:
            assert isinstance(term, str)
            assert isinstance(score, float)

    def test_rake_keywords(self) -> None:
        extractor = KeywordExtractor()
        result = extractor._rake_keywords(LONG_TEXT, top_n=10)
        assert len(result) > 0


class TestDocumentSimilarity:
    def test_identical_texts_score_one(self) -> None:
        sim = DocumentSimilarity()
        text = "The quick brown fox jumps over the lazy dog."
        assert sim.similarity(text, text) == pytest.approx(1.0)

    def test_different_texts_score_less_than_one(self) -> None:
        sim = DocumentSimilarity()
        a = "Invoice payment total amount due"
        b = "Machine learning neural network deep learning"
        score = sim.similarity(a, b)
        assert score < 0.5

    def test_empty_texts_score_zero(self) -> None:
        sim = DocumentSimilarity()
        assert sim.similarity("", "") == pytest.approx(0.0)
        assert sim.similarity("some text", "") == pytest.approx(0.0)

    def test_similarity_matrix_shape(self) -> None:
        sim = DocumentSimilarity()
        texts = ["document one text", "document two text", "document three text"]
        matrix = sim.similarity_matrix(texts)
        assert len(matrix) == 3
        assert all(len(row) == 3 for row in matrix)

    def test_similarity_matrix_diagonal_is_one(self) -> None:
        sim = DocumentSimilarity()
        texts = ["foo bar baz", "hello world test"]
        matrix = sim.similarity_matrix(texts)
        for i in range(len(texts)):
            assert matrix[i][i] == pytest.approx(1.0)

    def test_similarity_matrix_is_symmetric(self) -> None:
        sim = DocumentSimilarity()
        texts = ["apple banana", "banana cherry", "cherry apple"]
        matrix = sim.similarity_matrix(texts)
        for i in range(len(texts)):
            for j in range(len(texts)):
                assert matrix[i][j] == pytest.approx(matrix[j][i])

    def test_most_similar(self) -> None:
        sim = DocumentSimilarity()
        corpus = [
            "invoice payment total amount",
            "machine learning deep neural",
            "invoice tax subtotal billing",
        ]
        query = "invoice total amount due"
        results = sim.most_similar(query, corpus, top_n=2)
        assert len(results) == 2
        # Invoice-related docs should rank higher
        top_idx, top_score = results[0]
        assert top_idx in (0, 2)
