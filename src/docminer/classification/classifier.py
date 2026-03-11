"""Document classifier — TF-IDF + Logistic Regression with rule-based fallback."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from docminer.classification.labels import (
    CONFIDENCE_THRESHOLD,
    DOCUMENT_TYPES,
    KEYWORD_SIGNATURES,
)
from docminer.core.types import ClassificationResult

if TYPE_CHECKING:
    from docminer.core.types import Document

logger = logging.getLogger(__name__)


class DocumentClassifier:
    """Classify documents into one of the supported types.

    Primary approach: TF-IDF + Logistic Regression trained on synthetic
    keyword documents.  Falls back to rule-based keyword scoring when
    the sklearn model is not available.
    """

    _SUPPORTED_TYPES = [t for t in DOCUMENT_TYPES if t != "unknown"]

    def __init__(self) -> None:
        self._model = None
        self._vectorizer = None
        self._is_trained = False
        self._feature_extractor = None
        self._try_build_model()

    # ------------------------------------------------------------------
    # Training / model building
    # ------------------------------------------------------------------

    def _try_build_model(self) -> None:
        """Build and train the TF-IDF + LR model using synthetic training data."""
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import LabelEncoder
            from sklearn.feature_extraction.text import TfidfVectorizer

            X_train, y_train = self._generate_training_data()
            if not X_train:
                return

            self._vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=5000,
                min_df=1,
                sublinear_tf=True,
            )
            X_vec = self._vectorizer.fit_transform(X_train)
            self._model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver="lbfgs",
            )
            self._model.fit(X_vec, y_train)
            self._is_trained = True
            logger.info("DocumentClassifier: sklearn model trained on %d examples", len(X_train))
        except ImportError:
            logger.warning("scikit-learn not available; using rule-based classifier")
        except Exception as exc:
            logger.warning("Failed to train classifier: %s", exc)

    @staticmethod
    def _generate_training_data() -> tuple[list[str], list[str]]:
        """Generate synthetic training texts from keyword signatures."""
        X: list[str] = []
        y: list[str] = []
        for doc_type, keywords in KEYWORD_SIGNATURES.items():
            if doc_type == "unknown":
                continue
            # Create multiple varied training examples per type
            for i in range(30):
                # Rotate keywords to create variation
                rotation = i % max(len(keywords), 1)
                kw_subset = keywords[rotation:] + keywords[:rotation]
                text = " ".join(kw_subset * 3)
                X.append(text)
                y.append(doc_type)
        return X, y

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, document: "Document") -> ClassificationResult:
        """Classify *document* and return a :class:`ClassificationResult`."""
        text = document.text
        if not text.strip():
            return ClassificationResult(
                document_type="unknown",
                confidence=0.0,
                all_scores={t: 0.0 for t in self._SUPPORTED_TYPES},
                features_used=[],
            )

        if self._is_trained:
            return self._classify_ml(text, document)
        return self._classify_rules(text)

    def classify_text(self, text: str) -> ClassificationResult:
        """Classify raw text without a Document object."""
        if self._is_trained:
            return self._classify_ml(text, document=None)
        return self._classify_rules(text)

    # ------------------------------------------------------------------
    # ML classification
    # ------------------------------------------------------------------

    def _classify_ml(self, text: str, document=None) -> ClassificationResult:
        """Use TF-IDF + LR model."""
        assert self._vectorizer is not None
        assert self._model is not None

        X_vec = self._vectorizer.transform([text])
        proba = self._model.predict_proba(X_vec)[0]
        classes = self._model.classes_

        all_scores = {cls: float(prob) for cls, prob in zip(classes, proba)}
        best_type = max(all_scores, key=all_scores.get)  # type: ignore[arg-type]
        best_score = all_scores[best_type]

        # Apply confidence threshold
        threshold = CONFIDENCE_THRESHOLD.get(best_type, 0.35)
        if best_score < threshold:
            best_type = "unknown"

        # Also run rule scoring for features_used
        rule_result = self._classify_rules(text)

        return ClassificationResult(
            document_type=best_type,
            confidence=best_score,
            all_scores=all_scores,
            features_used=["tfidf", "logistic_regression"] + rule_result.features_used,
        )

    # ------------------------------------------------------------------
    # Rule-based classification
    # ------------------------------------------------------------------

    def _classify_rules(self, text: str) -> ClassificationResult:
        """Score each document type by keyword density."""
        text_lower = text.lower()
        word_count = max(len(text_lower.split()), 1)

        scores: dict[str, float] = {}
        features_used: list[str] = []

        for doc_type, keywords in KEYWORD_SIGNATURES.items():
            if doc_type == "unknown":
                continue
            hits = sum(1 for kw in keywords if kw in text_lower)
            density = hits / word_count
            scores[doc_type] = density * 100  # scale for readability
            if hits > 0:
                features_used.append(f"keyword:{doc_type}:{hits}")

        if not scores or all(v == 0 for v in scores.values()):
            return ClassificationResult(
                document_type="unknown",
                confidence=0.0,
                all_scores={t: 0.0 for t in self._SUPPORTED_TYPES},
                features_used=["rule_based"],
            )

        best_type = max(scores, key=scores.get)  # type: ignore[arg-type]
        total = sum(scores.values())
        normalized = {k: v / total for k, v in scores.items()} if total > 0 else scores
        best_conf = normalized.get(best_type, 0.0)

        threshold = CONFIDENCE_THRESHOLD.get(best_type, 0.35)
        if best_conf < threshold:
            best_type = "unknown"

        return ClassificationResult(
            document_type=best_type,
            confidence=best_conf,
            all_scores=normalized,
            features_used=["rule_based"] + features_used,
        )
