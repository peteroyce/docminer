"""Extractive summarizer using TextRank (NetworkX PageRank)."""

from __future__ import annotations

import logging
import math
import re
from collections import Counter

logger = logging.getLogger(__name__)

# Minimum sentence length to be included in the graph
_MIN_SENTENCE_WORDS = 5
_MAX_SENTENCE_WORDS = 80


class DocumentSummarizer:
    """Extractive document summarisation using TextRank.

    Algorithm:
    1. Split text into sentences.
    2. Build a similarity graph where nodes are sentences and edge weights
       are TF-IDF cosine similarities.
    3. Run PageRank on the graph.
    4. Return the top-k sentences in their original order.
    """

    def __init__(self, damping: float = 0.85, iterations: int = 100) -> None:
        self.damping = damping
        self.iterations = iterations

    def summarize(self, text: str, num_sentences: int = 5) -> str:
        """Return an extractive summary of *text*.

        Parameters
        ----------
        text:
            The document's full text.
        num_sentences:
            Number of sentences to include in the summary.

        Returns
        -------
        str
            Summary as a single string.
        """
        if not text.strip():
            return ""

        sentences = self._split_sentences(text)
        filtered = [
            s for s in sentences
            if _MIN_SENTENCE_WORDS <= len(s.split()) <= _MAX_SENTENCE_WORDS
        ]

        if len(filtered) == 0:
            return sentences[0] if sentences else ""

        if len(filtered) <= num_sentences:
            return " ".join(filtered)

        try:
            scores = self._textrank(filtered)
        except Exception as exc:
            logger.warning("TextRank failed (%s); falling back to first-N strategy", exc)
            return " ".join(filtered[:num_sentences])

        # Select top-k by score, then restore original order
        top_indices = sorted(
            range(len(filtered)), key=lambda i: scores[i], reverse=True
        )[:num_sentences]
        top_indices_ordered = sorted(top_indices)
        return " ".join(filtered[i] for i in top_indices_ordered)

    # ------------------------------------------------------------------
    # TextRank
    # ------------------------------------------------------------------

    def _textrank(self, sentences: list[str]) -> list[float]:
        """Run TextRank and return a score per sentence."""
        try:
            import networkx as nx  # type: ignore
        except ImportError:
            logger.warning("networkx not available; using positional scoring")
            return self._positional_scores(sentences)

        tf_vectors = [self._tf(s) for s in sentences]
        n = len(sentences)

        G = nx.Graph()
        G.add_nodes_from(range(n))

        for i in range(n):
            for j in range(i + 1, n):
                sim = self._cosine_similarity(tf_vectors[i], tf_vectors[j])
                if sim > 0.0:
                    G.add_edge(i, j, weight=sim)

        if G.number_of_edges() == 0:
            return self._positional_scores(sentences)

        pr = nx.pagerank(G, alpha=self.damping, max_iter=self.iterations)
        return [pr.get(i, 0.0) for i in range(n)]

    @staticmethod
    def _positional_scores(sentences: list[str]) -> list[float]:
        """Simple fallback: weight sentences by inverse position."""
        n = len(sentences)
        return [1.0 / (i + 1) for i in range(n)]

    # ------------------------------------------------------------------
    # Text processing utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences using punctuation heuristics."""
        # Simple sentence splitter — handles common abbreviations
        text = re.sub(r"\s+", " ", text)
        raw = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        return [s.strip() for s in raw if s.strip()]

    @staticmethod
    def _tf(text: str) -> dict[str, float]:
        """Compute raw term frequencies for a sentence."""
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        if not words:
            return {}
        counts = Counter(words)
        total = len(words)
        return {w: c / total for w, c in counts.items()}

    @staticmethod
    def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
        """Compute cosine similarity between two TF dicts."""
        if not a or not b:
            return 0.0
        common = set(a) & set(b)
        if not common:
            return 0.0
        dot = sum(a[w] * b[w] for w in common)
        norm_a = math.sqrt(sum(v**2 for v in a.values()))
        norm_b = math.sqrt(sum(v**2 for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
