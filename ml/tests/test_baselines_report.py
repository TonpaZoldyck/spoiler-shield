import numpy as np
import pytest

from spoiler_shield.baselines import KeywordRules, TfidfLogReg
from spoiler_shield.data.schema import Example
from spoiler_shield.data.splits import TitleLeakError, assign_splits
from spoiler_shield.eval.report import render_markdown, run_ladder


def test_keyword_rules_scores():
    rules = KeywordRules()
    rows = [
        Example(example_id="a", source="gold", title_id="t", sentence="Loved the score here."),
        Example(example_id="b", source="gold", title_id="t", sentence="Turns out she dies."),
    ]
    assert rules.predict_proba(rows).tolist() == [0.0, 0.75]


def test_tfidf_needs_both_classes(synthetic):
    only_neg = [ex for ex in synthetic if ex.label == 0]
    with pytest.raises(ValueError):
        TfidfLogReg().fit(only_neg)
    with pytest.raises(RuntimeError):
        TfidfLogReg().predict_proba(synthetic)


def test_tfidf_probabilities_are_valid_and_informative(synthetic):
    splits = assign_splits(synthetic)
    model = TfidfLogReg().fit(splits["train"])
    val = [ex for ex in splits["val"] if ex.label is not None]
    p = model.predict_proba(val)
    y = np.array([ex.label for ex in val])
    assert ((p >= 0) & (p <= 1)).all()
    assert p[y == 1].mean() > p[y == 0].mean()


def test_ladder_reports_on_val_and_guards_the_test_split(synthetic):
    splits = assign_splits(synthetic)
    reports = run_ladder([KeywordRules(), TfidfLogReg()], splits, target_recall=0.8)
    assert [r.name for r in reports] == ["keyword-rules", "tfidf-logreg"]
    for r in reports:
        assert r.split == "val"
        assert r.overall.n == sum(1 for ex in splits["val"] if ex.label is not None)
        assert set(r.slices) == {"source", "length", "title mention"}
    md = render_markdown(reports, title="Test report", notes=["synthetic"])
    assert "| keyword-rules |" in md and "| tfidf-logreg |" in md
    assert "## Slices: tfidf-logreg" in md

    with pytest.raises(ValueError, match="releases only"):
        run_ladder([KeywordRules()], splits, report_split="test")
    assert run_ladder([KeywordRules()], splits, report_split="test", allow_test=True)


def test_ladder_refuses_leaky_splits(synthetic):
    splits = assign_splits(synthetic)
    splits["val"].append(splits["train"][0])
    with pytest.raises(TitleLeakError):
        run_ladder([KeywordRules()], splits)
