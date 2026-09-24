"""Baseline 1: TF-IDF features with calibrated logistic regression.

The honest classical baseline. Two details matter more than the model choice:

* The vectoriser is fitted **inside** the pipeline, on training rows only. The
  2023 notebook fitted TF-IDF on all reviews before splitting, which leaks the
  test vocabulary into training.
* Probabilities are calibrated with cross-validation on the training split, so
  thresholds chosen on validation data mean what they say.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

from spoiler_shield.data.schema import Example


def model_text(ex: Example) -> str:
    """Text the model sees: the previous sentence gives light context."""
    return f"{ex.context} [SEP] {ex.sentence}" if ex.context else ex.sentence


class TfidfLogReg:
    name = "tfidf-logreg"

    def __init__(
        self,
        *,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        c: float = 4.0,
        calibration_folds: int = 3,
        seed: int = 0,
    ) -> None:
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.c = c
        self.calibration_folds = calibration_folds
        self.seed = seed
        self._pipeline: Pipeline | None = None

    def fit(self, train: Sequence[Example]) -> TfidfLogReg:
        labelled = [ex for ex in train if ex.label is not None]
        y = np.array([ex.label for ex in labelled], dtype=int)
        if len(set(y.tolist())) < 2:
            raise ValueError("training data needs both spoilers and non-spoilers")
        base = LogisticRegression(
            C=self.c, class_weight="balanced", max_iter=2000, random_state=self.seed
        )
        self._pipeline = make_pipeline(
            TfidfVectorizer(ngram_range=self.ngram_range, min_df=self.min_df, sublinear_tf=True),
            CalibratedClassifierCV(base, method="sigmoid", cv=self.calibration_folds),
        )
        self._pipeline.fit([model_text(ex) for ex in labelled], y)
        return self

    def predict_proba(self, examples: Sequence[Example]) -> NDArray[np.float64]:
        if self._pipeline is None:
            raise RuntimeError("call fit() before predict_proba()")
        return self._pipeline.predict_proba([model_text(ex) for ex in examples])[:, 1]
