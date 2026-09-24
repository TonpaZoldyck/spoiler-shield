"""Baseline 0: hand-written spoiler cue phrases.

The floor every learned model has to clear. Each matched cue raises the score:
one cue gives 0.5, two give 0.75, and so on, so scores stay in [0, 1) and can
be thresholded like any other model's.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from spoiler_shield.data.schema import Example

DEFAULT_CUES: tuple[str, ...] = (
    r"\b(dies|died|killed|kills|death of|murdered)\b",
    r"\b(the|at the|in the) (ending|end)\b",
    r"\bturns out\b",
    r"\btwist\b",
    r"\b(reveals?|revealed|revelation)\b",
    r"\ball along\b",
    r"\bbehind it all\b",
    r"\bwas (the|actually) (killer|villain|traitor|informant|mole)\b",
    r"\b(betrays?|betrayed|betrayal)\b",
    r"\b(survives?|survived)\b",
    r"\b(finale|last episode|final scene|final shot)\b",
    r"\bspoilers?\b",
)


class KeywordRules:
    """Score = 1 - 0.5 ** (number of distinct cues matched)."""

    name = "keyword-rules"

    def __init__(self, cues: Sequence[str] = DEFAULT_CUES) -> None:
        self._patterns = [re.compile(c, re.IGNORECASE) for c in cues]

    def fit(self, train: Sequence[Example]) -> KeywordRules:
        return self

    def matches(self, text: str) -> int:
        return sum(1 for p in self._patterns if p.search(text))

    def predict_proba(self, examples: Sequence[Example]) -> NDArray[np.float64]:
        hits = np.array([self.matches(ex.sentence) for ex in examples], dtype=float)
        return 1.0 - 0.5**hits
