"""Sentence splitting for training data.

Mirrors the extension's splitter (``extension/lib/segment.ts``): split at
sentence-ending punctuation, then merge pieces that end in a known
abbreviation or a single-letter initial. Keeping training and inference
segmentation alike matters, because the model only ever sees one sentence.
The browser uses ICU via ``Intl.Segmenter``; small differences on unusual
punctuation are expected and measured on the gold set.
"""

from __future__ import annotations

import re

_ABBREVIATION_WORDS = (
    "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "capt", "lt", "sgt", "gen", "col",
    "vs", "etc", "e.g", "i.e", "approx", "no", "vol", "ep", "eps", "pt", "ch", "fig", "mt",
)  # fmt: skip
ABBREVIATIONS = frozenset(f"{a}." for a in _ABBREVIATION_WORDS)

# A sentence ends at . ! ? optionally followed by one closing quote or bracket,
# then whitespace. Lookbehinds keep the punctuation and quote in the sentence.
_CLOSERS = "\"'\u2019\u201d)\\]"
_BOUNDARY = re.compile(rf"(?<=[.!?])\s+|(?<=[.!?][{_CLOSERS}])\s+")


def _ends_with_abbreviation(piece: str) -> bool:
    words = piece.split()
    last = words[-1].lower() if words else ""
    return last in ABBREVIATIONS or re.fullmatch(r"[a-z]\.", last) is not None


def split_sentences(text: str) -> list[str]:
    pieces = [p for p in _BOUNDARY.split(text) if p.strip()]
    out: list[str] = []
    carry = ""
    for piece in pieces:
        carry = f"{carry} {piece}" if carry else piece
        if not _ends_with_abbreviation(carry):
            out.append(carry.strip())
            carry = ""
    if carry.strip():
        out.append(carry.strip())
    return out
