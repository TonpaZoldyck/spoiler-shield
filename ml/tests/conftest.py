"""Shared fixtures. Synthetic data here exercises code paths only; it says
nothing about real model quality and is never used for reported results."""

from __future__ import annotations

import random

import pytest

from spoiler_shield.data.schema import Example

_NAMES = ["Mara", "Elias", "Quinn", "Tomas", "Ada", "Rook", "Iris", "Bram"]
_SPOILER = [
    "{n} dies in the finale saving the crew.",
    "It turns out {n} was the informant all along.",
    "{n} betrays everyone at the wedding.",
    "The twist is that {n} was dead the whole time.",
    "{n} survives the fire and takes the throne.",
    "In the end {n} is revealed as the killer.",
]
_BENIGN = [
    "The cinematography in the storm scene is gorgeous.",
    "I love how {n} is written this season.",
    "The soundtrack carries the slower episodes.",
    "Anyone know when the next season comes out?",
    "{n} has the best costumes in the show.",
    "Rewatching the first season before the new one drops.",
    "The pacing dragged a little in the middle.",
]


def make_synthetic(n_titles: int = 60, per_title: int = 30, seed: int = 0) -> list[Example]:
    rng = random.Random(seed)
    rows: list[Example] = []
    for t in range(n_titles):
        title = f"Show {t}"
        for i in range(per_title):
            spoiler = rng.random() < 0.25
            name = rng.choice(_NAMES)
            template = rng.choice(_SPOILER if spoiler else _BENIGN)
            # 10% label noise, like real reviewer tags.
            label = int(spoiler) if rng.random() > 0.1 else int(not spoiler)
            rows.append(
                Example(
                    example_id=f"synthetic:{t}:{i}",
                    source="synthetic",
                    title_id=f"synthetic:{t}",
                    title=title,
                    sentence=template.format(n=name),
                    label=label,  # type: ignore[arg-type]
                )
            )
    return rows


@pytest.fixture
def synthetic() -> list[Example]:
    return make_synthetic()
