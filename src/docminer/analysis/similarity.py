"""Document similarity — TF-IDF cosine similarity between documents."""

from __future__ import annotations

import logging
import math
import re
from typing import Sequence

logger = logging.getLogger(__name__)

_STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be
    because been before being below between both but by can cannot could
    did do does doing don down during each few for from further get got has
    have having he her here hers herself him himself his how i if in into
    is it its itself let me more most my myself no nor not of off on once
    only or other our ours ourselves out over own same she should so some
    such than that the their theirs them themselves then there these they
    this those through to too under until up very was we were what when
    where which while who will with would you your yours yourself""".split()
)


class DocumentSimilarity:
    """Compare pairs or batches of documents using TF-IDF cosine similarity.

    Usage::

        sim = DocumentSimilarity()
        score = sim.similarity("text of document A", "text of document B")
        matrix = sim.similarity_matrix(["doc A", "doc B", "doc C"])
    """

    def similarity(self, text_a: str, text_b: str) -> float:
        """Return the cosine similarity score between two texts (0–1).

        Parameters
        ----------
        text_a, text_b:
            Raw text of the two documents.
        """
        vec_a = self._tfidf_vector(text_a)
        vec_b = self._tfidf_vector(text_b)
        return self._cosine(vec_a, vec_b)

    def similarity_matrix(self, texts: Sequence[str]) -> list[list[float]]:
        """Compute a pairwise similarity matrix for a list of texts.

        Returns
        -------
        list[list[float]]
            ``n x n`` matrix where ``result[i][j]`` is the similarity
            between ``texts[i]`` and ``texts[j]``.
        """
        vectors = [self._tfidf_vector(t) for t in texts]
        n = len(texts)
        matrix: list[list[float]] = []
        for i in range(n):
            row: list[float] = []
            for j in range(n):
                if i == j:
                    row.append(1.0)
                elif i < j:
                    row.append(self._cosine(vectors[i], vectors[j]))
                else:
                    row.append(matrix[j][i])  # symmetric
            matrix.append(row)
        return matrix

    def most_similar(
        self,
        query: str,
        corpus: Sequence[str],
        top_n: int = 5,
    ) -> list[tuple[int, float]]:
        """Find the most similar documents in *corpus* to *query*.

        Returns
        -------
        list[tuple[int, float]]
            List of ``(index, score)`` tuples sorted by descending similarity.
        """
        query_vec = self._tfidf_vector(query)
        scores = [
            (i, self._cosine(query_vec, self._tfidf_vector(doc)))
            for i, doc in enumerate(corpus)
        ]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]

    # ------------------------------------------------------------------
    # TF-IDF (single document, no corpus needed — use raw TF)
    # ------------------------------------------------------------------

    def _tfidf_vector(self, text: str) -> dict[str, float]:
        """Compute a normalised TF vector for *text* (IDF is implicit from the raw TF)."""
        words = self._tokenize(text)
        if not words:
            return {}
        counts: dict[str, int] = {}
        for w in words:
            counts[w] = counts.get(w, 0) + 1
        total = len(words)
        tf = {w: c / total for w, c in counts.items()}
        return tf

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        return [w for w in words if w not in _STOPWORDS]

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
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
