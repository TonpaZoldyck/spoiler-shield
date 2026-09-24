"""Title-grouped, hash-based train/validation/test splits.

Rows are assigned to a split by hashing their ``title_id``, never at random per
row. That gives three guarantees the 2023 notebook lacked:

1. **No title leakage.** Every sentence about a work lands in the same split, so
   the model can't pass the test by memorising character names.
2. **Determinism.** The same title always lands in the same split, on any
   machine, without storing a split file.
3. **Stability.** Adding new titles never moves existing ones, so old test
   numbers stay comparable as the dataset grows.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from typing import Literal

from spoiler_shield.data.schema import Example

Split = Literal["train", "val", "test"]
SPLITS: tuple[Split, ...] = ("train", "val", "test")
DEFAULT_SALT = "spoiler-shield-v1"


def split_of(
    title_id: str,
    *,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    salt: str = DEFAULT_SALT,
) -> Split:
    """Return the split for a title. Pure function of its arguments."""
    if not (val_frac >= 0 and test_frac >= 0 and val_frac + test_frac < 1):
        raise ValueError("fractions must be non-negative and sum to less than 1")
    digest = hashlib.sha256(f"{salt}:{title_id}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64  # uniform in [0, 1)
    if u < test_frac:
        return "test"
    if u < test_frac + val_frac:
        return "val"
    return "train"


def assign_splits(
    examples: Iterable[Example],
    *,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    salt: str = DEFAULT_SALT,
) -> dict[Split, list[Example]]:
    """Group examples into splits by title."""
    out: dict[Split, list[Example]] = {s: [] for s in SPLITS}
    for ex in examples:
        out[split_of(ex.title_id, val_frac=val_frac, test_frac=test_frac, salt=salt)].append(ex)
    return out


class TitleLeakError(AssertionError):
    """Raised when one title appears in more than one split."""


def check_no_title_leakage(splits: dict[Split, list[Example]]) -> None:
    """Fail loudly if any title appears in two splits."""
    seen: dict[str, set[Split]] = defaultdict(set)
    for name, rows in splits.items():
        for ex in rows:
            seen[ex.title_id].add(name)
    leaked = {t: sorted(s) for t, s in seen.items() if len(s) > 1}
    if leaked:
        sample = dict(list(leaked.items())[:5])
        raise TitleLeakError(f"{len(leaked)} titles appear in several splits, e.g. {sample}")


def subsample(rows: list[Example], n: int, *, salt: str = DEFAULT_SALT) -> list[Example]:
    """Deterministic subsample of at most ``n`` rows, chosen by hashing example ids.

    Useful to iterate quickly on millions of Goodreads sentences. Stable across
    runs and machines, unlike random sampling without a stored seed.
    """
    if len(rows) <= n:
        return list(rows)

    def key(ex: Example) -> bytes:
        return hashlib.sha256(f"{salt}:sample:{ex.example_id}".encode()).digest()

    return sorted(rows, key=key)[:n]
