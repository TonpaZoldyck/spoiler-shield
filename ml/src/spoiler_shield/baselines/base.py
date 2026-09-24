"""The interface shared by every model in the ladder."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from spoiler_shield.data.schema import Example


class SpoilerModel(Protocol):
    """Anything that can be fitted on examples and return spoiler probabilities."""

    name: str

    def fit(self, train: Sequence[Example]) -> SpoilerModel: ...

    def predict_proba(self, examples: Sequence[Example]) -> NDArray[np.float64]:
        """Probability of 'spoiler' for each example, in [0, 1]."""
        ...
