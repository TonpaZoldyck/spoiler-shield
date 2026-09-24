import pytest

from spoiler_shield.data.schema import Example
from spoiler_shield.data.splits import (
    TitleLeakError,
    assign_splits,
    check_no_title_leakage,
    split_of,
    subsample,
)


def test_split_is_deterministic_and_salted():
    assert split_of("imdb:tt0111161") == split_of("imdb:tt0111161")
    ids = [f"t{i}" for i in range(200)]
    assert [split_of(t) for t in ids] != [split_of(t, salt="other") for t in ids]


def test_split_proportions_match_fractions():
    ids = [f"title-{i}" for i in range(20_000)]
    counts = {"train": 0, "val": 0, "test": 0}
    for t in ids:
        counts[split_of(t, val_frac=0.1, test_frac=0.1)] += 1
    assert abs(counts["test"] / len(ids) - 0.1) < 0.01
    assert abs(counts["val"] / len(ids) - 0.1) < 0.01


def test_all_sentences_of_a_title_share_a_split(synthetic):
    splits = assign_splits(synthetic)
    check_no_title_leakage(splits)  # does not raise
    assert sum(len(v) for v in splits.values()) == len(synthetic)
    assert all(splits[s] for s in ("train", "val", "test"))


def test_adding_titles_never_moves_existing_ones():
    before = {f"t{i}": split_of(f"t{i}") for i in range(500)}
    _ = [split_of(f"new{i}") for i in range(500)]
    assert {t: split_of(t) for t in before} == before


def test_leak_check_catches_a_title_in_two_splits(synthetic):
    splits = assign_splits(synthetic)
    moved = splits["train"][0]
    splits["test"].append(moved)
    with pytest.raises(TitleLeakError, match=moved.title_id):
        check_no_title_leakage(splits)


def test_invalid_fractions_rejected():
    with pytest.raises(ValueError):
        split_of("t", val_frac=0.6, test_frac=0.5)


def test_subsample_is_deterministic_and_bounded(synthetic):
    a = subsample(synthetic, 100)
    assert len(a) == 100
    assert a == subsample(list(reversed(synthetic)), 100)
    assert subsample(synthetic[:10], 100) == synthetic[:10]
    assert all(isinstance(x, Example) for x in a)
