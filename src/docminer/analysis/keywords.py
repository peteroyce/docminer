"""Keyword extractor — TF-IDF and RAKE-style extraction."""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from typing import Sequence

logger = logging.getLogger(__name__)

# Common English stopwords for filtering
_STOPWORDS = frozenset(
    """a about above after again against all am an and any are aren't as at be
    because been before being below between both but by can't cannot could
    couldn't did didn't do does doesn't doing don't down during each few for
    from further get got has hasn't have haven't having he he'd he'll he's
    her here here's hers herself him himself his how how's i i'd i'll i'm
    i've if in into is isn't it it's its itself let's me more most mustn't my
    myself no nor not of off on once only or other ought our ours ourselves
    out over own same shan't she she'd she'll she's should shouldn't so some
    such than that that's the their theirs them themselves then there there's
    these they they'd they'll they're they've this those through to too under
    until up very was wasn't we we'd we'll we're we've were weren't what
    what's when when's where where's which while who who's whom why why's
    will with won't would wouldn't you you'd you'll you're you've your yours
    yourself yourselves""".split()
)


class KeywordExtractor:
    """Extract keywords from document text.

    Two strategies are used:

    1. **TF-IDF single-document**: treats each sentence as a "document" and
       computes term importance relative to the corpus of sentences.
    2. **RAKE** (Rapid Automatic Keyword Extraction): extracts phrase-level
       keywords by splitting on stopwords and punctuation.

    Results are merged and ranked by combined score.
    """

    def __init__(self, min_word_length: int = 3) -> None:
        self.min_word_length = min_word_length

    def extract(self, text: str, top_n: int = 15) -> list[str]:
        """Return the top-N keywords from *text*.

        Parameters
        ----------
        text:
            Source document text.
        top_n:
            Number of keywords to return.

        Returns
        -------
        list[str]
            Keywords sorted by relevance (most important first).
        """
        if not text.strip():
            return []

        tfidf_kws = self._tfidf_keywords(text, top_n=top_n * 2)
        rake_kws = self._rake_keywords(text, top_n=top_n * 2)

        # Merge scores: RAKE phrases and TF-IDF single words
        combined: dict[str, float] = {}
        for kw, score in tfidf_kws:
            combined[kw] = combined.get(kw, 0.0) + score
        for kw, score in rake_kws:
            combined[kw] = combined.get(kw, 0.0) + score

        ranked = sorted(combined, key=combined.get, reverse=True)  # type: ignore[arg-type]
        return ranked[:top_n]

    # ------------------------------------------------------------------
    # TF-IDF (intra-document, sentence-level)
    # ------------------------------------------------------------------

    def _tfidf_keywords(self, text: str, top_n: int) -> list[tuple[str, float]]:
        """Compute TF-IDF over sentences and return top single-word keywords."""
        sentences = self._tokenize_sentences(text)
        if not sentences:
            return []

        # IDF over the sentence corpus
        df: Counter[str] = Counter()
        sent_tfs: list[dict[str, float]] = []
        for sent in sentences:
            words = self._tokenize_words(sent)
            tf: dict[str, float] = {}
            for w in words:
                tf[w] = tf.get(w, 0.0) + 1.0
            total = len(words) or 1
            tf = {w: c / total for w, c in tf.items()}
            sent_tfs.append(tf)
            for w in set(words):
                df[w] += 1

        n = len(sentences)
        idf: dict[str, float] = {
            w: math.log((n + 1) / (c + 1)) + 1.0 for w, c in df.items()
        }

        # Aggregate TF-IDF across all sentences
        scores: dict[str, float] = defaultdict(float)
        for tf in sent_tfs:
            for w, tf_val in tf.items():
                scores[w] += tf_val * idf.get(w, 1.0)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]

    # ------------------------------------------------------------------
    # RAKE
    # ------------------------------------------------------------------

    def _rake_keywords(self, text: str, top_n: int) -> list[tuple[str, float]]:
        """Rapid Automatic Keyword Extraction."""
        # Split on stopwords, punctuation, and sentence boundaries
        stop_pat = re.compile(
            r"\b(?:" + "|".join(re.escape(w) for w in _STOPWORDS) + r")\b",
            re.IGNORECASE,
        )
        phrases_raw = stop_pat.split(text)
        phrases = [
            re.sub(r"[^\w\s\-]", "", p).strip()
            for p in phrases_raw
        ]
        phrases = [p for p in phrases if p and len(p) >= self.min_word_length]

        # Score each phrase: sum of word degree / word frequency
        word_freq: Counter[str] = Counter()
        word_degree: dict[str, float] = defaultdict(float)

        for phrase in phrases:
            words = phrase.lower().split()
            words = [w for w in words if len(w) >= self.min_word_length]
            if not words:
                continue
            degree = len(words) - 1
            for w in words:
                word_freq[w] += 1
                word_degree[w] += degree

        word_score: dict[str, float] = {
            w: (word_degree[w] + word_freq[w]) / max(word_freq[w], 1)
            for w in word_freq
        }

        phrase_scores: dict[str, float] = {}
        for phrase in phrases:
            words = phrase.lower().split()
            words = [w for w in words if len(w) >= self.min_word_length]
            if not words:
                continue
            key = " ".join(words)
            if key and key not in phrase_scores:
                phrase_scores[key] = sum(word_score.get(w, 0.0) for w in words)

        ranked = sorted(phrase_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def _tokenize_words(self, text: str) -> list[str]:
        words = re.findall(r"\b[a-zA-Z]{%d,}\b" % self.min_word_length, text.lower())
        return [w for w in words if w not in _STOPWORDS]

    @staticmethod
    def _tokenize_sentences(text: str) -> list[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
