"""One sentence-level schema shared by every data source.

Every dataset (Goodreads, IMDb, our gold set, user feedback) is converted into
``Example`` rows so that splitting, training and evaluation code never has to
know where a sentence came from.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

Source = Literal["goodreads", "imdb", "gold", "feedback", "synthetic"]


class Example(BaseModel):
    """A single sentence to classify, with the title it might spoil."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    example_id: str
    source: Source
    title_id: str
    """Stable id of the work (book id, IMDb id). Splits are assigned by this."""
    title: str = ""
    """Human-readable title, used as model context when known."""
    sentence: str
    context: str = ""
    """The previous sentence in the same review or comment, if any."""
    label: Literal[0, 1] | None = None
    """1 = spoiler, 0 = not a spoiler, None = unlabelled."""
    review_id: str | None = None

    @field_validator("sentence")
    @classmethod
    def _sentence_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("sentence must not be blank")
        return v

    @field_validator("title_id")
    @classmethod
    def _title_id_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("title_id must not be blank")
        return v


def write_jsonl(path: str | Path, examples: Iterable[Example]) -> int:
    """Write examples as JSON lines. Returns the number written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.model_dump_json() + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[Example]:
    """Stream examples back from a JSON-lines file."""
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if line.strip():
                try:
                    yield Example.model_validate(json.loads(line))
                except ValueError as err:
                    raise ValueError(f"{path}:{line_no}: {err}") from err
