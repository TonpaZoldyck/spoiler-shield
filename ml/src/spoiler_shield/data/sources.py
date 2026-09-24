"""Converters from raw public datasets to ``Example`` rows.

Formats follow each dataset's published description. They have not yet been
checked against the real files, because the dataset hosts are unreachable from
the build environment (tracker task P1.1). Re-verify field names when the
downloads land.

* **Goodreads spoilers** (Wan et al., ACL 2019): one JSON object per line, with
  ``review_sentences`` as ``[[label, sentence], ...]`` where label 1 means the
  reviewer tagged that sentence as a spoiler.
* **IMDB Spoiler Dataset** (Misra): one JSON object per line with
  ``is_spoiler`` for the **whole review**. Sentences from reviews flagged as
  spoilers stay unlabelled until the Phase 2 LLM labeller assigns
  sentence-level labels. Sentences from unflagged reviews are labelled 0, a
  weak but mostly reliable negative.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any

from spoiler_shield.data.cleaning import clean_sentence, is_usable
from spoiler_shield.data.schema import Example
from spoiler_shield.data.segment import split_sentences


def _open_text(path: str | Path) -> IO[str]:
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def _json_lines(path: str | Path) -> Iterator[dict[str, Any]]:
    with _open_text(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def goodreads_examples(path: str | Path, titles: dict[str, str] | None = None) -> Iterator[Example]:
    """Sentence-level examples from ``goodreads_reviews_spoiler.json(.gz)``."""
    titles = titles or {}
    for review in _json_lines(path):
        book_id = str(review["book_id"])
        review_id = str(review["review_id"])
        previous = ""
        for i, (label, raw) in enumerate(review["review_sentences"]):
            sentence = clean_sentence(raw)
            if not is_usable(sentence):
                continue
            yield Example(
                example_id=f"goodreads:{review_id}:{i}",
                source="goodreads",
                title_id=f"goodreads:{book_id}",
                title=titles.get(book_id, ""),
                sentence=sentence,
                context=previous,
                label=1 if int(label) == 1 else 0,
                review_id=review_id,
            )
            previous = sentence


def imdb_examples(path: str | Path, titles: dict[str, str] | None = None) -> Iterator[Example]:
    """Examples from ``IMDB_reviews.json``: labelled 0 or left unlabelled."""
    titles = titles or {}
    for n, review in enumerate(_json_lines(path)):
        movie_id = str(review["movie_id"])
        review_id = f"{movie_id}:{review.get('user_id', 'anon')}:{n}"
        flagged = bool(review["is_spoiler"])
        previous = ""
        for i, raw in enumerate(split_sentences(review["review_text"])):
            sentence = clean_sentence(raw)
            if not is_usable(sentence):
                continue
            yield Example(
                example_id=f"imdb:{review_id}:{i}",
                source="imdb",
                title_id=f"imdb:{movie_id}",
                title=titles.get(movie_id, ""),
                sentence=sentence,
                context=previous,
                label=None if flagged else 0,
                review_id=review_id,
            )
            previous = sentence
