"""Text cleaning applied to every source before splitting.

The most important job here is removing **spoiler markup**. Review sites wrap
spoilers in tags such as Goodreads' "(view spoiler)[ ... (hide spoiler)]" or
Reddit's ">! ... !<". If those markers survive into training data, a model can
score well by detecting the markup instead of the content: a label leak that
looks great offline and fails on real, untagged comments.
"""

from __future__ import annotations

import html
import re
import unicodedata

_SPOILER_MARKUP = [
    re.compile(r"\(view spoiler\)\s*\[?", re.IGNORECASE),  # Goodreads open
    re.compile(r"\(hide spoiler\)\s*\]?", re.IGNORECASE),  # Goodreads close
    re.compile(r">!|!<"),  # Reddit inline spoiler
    re.compile(r"\[/?spoilers?\]", re.IGNORECASE),  # BBCode-style tags
    re.compile(r"\*\*?spoilers?( alert)?\*?\*?:?", re.IGNORECASE),  # "**SPOILER ALERT**:"
]
_WS = re.compile(r"\s+")
_MIN_CHARS = 8
_MIN_WORDS = 2


def strip_spoiler_markup(text: str) -> str:
    """Remove explicit spoiler markers, keeping the text they wrapped."""
    for pattern in _SPOILER_MARKUP:
        text = pattern.sub(" ", text)
    return text


def normalise(text: str) -> str:
    """Unescape HTML, normalise Unicode and collapse whitespace."""
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    return _WS.sub(" ", text).strip()


def clean_sentence(text: str) -> str:
    """Full cleaning pipeline for one sentence."""
    return normalise(strip_spoiler_markup(text))


def is_usable(sentence: str) -> bool:
    """Drop fragments too short to judge, such as '!!' or 'Wow.'."""
    return len(sentence) >= _MIN_CHARS and len(sentence.split()) >= _MIN_WORDS
