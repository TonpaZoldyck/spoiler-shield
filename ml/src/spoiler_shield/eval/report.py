"""Run the model ladder on the splits and write a markdown report.

Thresholds are always chosen on the **validation** split for a target recall.
Reports run on validation data by default; the test split is only used when
explicitly requested for a release, so it can't quietly steer model choices.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from spoiler_shield.baselines.base import SpoilerModel
from spoiler_shield.data.schema import Example
from spoiler_shield.data.splits import Split, check_no_title_leakage
from spoiler_shield.eval.metrics import EvalResult, evaluate, threshold_for_recall

SliceFn = Callable[[Example], str]


def _length_bucket(ex: Example) -> str:
    n = len(ex.sentence.split())
    return "short (<10 words)" if n < 10 else "medium (10-25)" if n <= 25 else "long (>25)"


def _mentions_title(ex: Example) -> str:
    if not ex.title:
        return "title unknown"
    return "names the title" if ex.title.lower() in ex.sentence.lower() else "doesn't name it"


DEFAULT_SLICES: dict[str, SliceFn] = {
    "source": lambda ex: ex.source,
    "length": _length_bucket,
    "title mention": _mentions_title,
}


@dataclass
class ModelReport:
    name: str
    split: Split
    target_recall: float
    overall: EvalResult
    slices: dict[str, dict[str, EvalResult]] = field(default_factory=dict)


def _labelled(rows: Sequence[Example]) -> list[Example]:
    return [ex for ex in rows if ex.label is not None]


def _labels(rows: Sequence[Example]) -> np.ndarray:
    return np.array([ex.label for ex in rows], dtype=int)


def run_ladder(
    models: Sequence[SpoilerModel],
    splits: dict[Split, list[Example]],
    *,
    target_recall: float = 0.9,
    report_split: Split = "val",
    allow_test: bool = False,
    slices: dict[str, SliceFn] | None = None,
) -> list[ModelReport]:
    """Fit each model on train, pick its threshold on val, report on ``report_split``."""
    if report_split == "test" and not allow_test:
        raise ValueError("the test split is for releases only: pass allow_test=True")
    check_no_title_leakage(splits)
    slices = DEFAULT_SLICES if slices is None else slices
    val = _labelled(splits["val"])
    target = _labelled(splits[report_split])
    if not val or not target:
        raise ValueError("validation and report splits need labelled examples")

    reports: list[ModelReport] = []
    for model in models:
        model.fit(splits["train"])
        threshold = threshold_for_recall(_labels(val), model.predict_proba(val), target_recall)
        scores = model.predict_proba(target)
        y = _labels(target)
        report = ModelReport(
            model.name, report_split, target_recall, evaluate(y, scores, threshold)
        )
        for slice_name, fn in slices.items():
            keys = np.array([fn(ex) for ex in target])
            report.slices[slice_name] = {
                str(k): evaluate(y[keys == k], scores[keys == k], threshold)
                for k in sorted(set(keys.tolist()))
            }
        reports.append(report)
    return reports


def _fmt(x: float, digits: int = 3) -> str:
    return "n/a" if math.isnan(x) else f"{x:.{digits}f}"


def render_markdown(
    reports: Sequence[ModelReport], *, title: str, notes: Sequence[str] = ()
) -> str:
    if not reports:
        raise ValueError("no reports to render")
    first = reports[0]
    lines = [
        f"# {title}",
        "",
        f"Generated {dt.date.today().isoformat()} on the **{first.split}** split. "
        f"Each model's threshold was chosen on validation data to catch "
        f"{first.target_recall:.0%} of spoilers. Compare models by false-blur rate "
        "and PR-AUC.",
        "",
        *[f"> {n}" for n in notes],
        *([""] if notes else []),
        "| Model | Sentences | Spoilers | Threshold | Recall | False-blur rate | "
        "Precision | PR-AUC | Calibration error |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in reports:
        o = r.overall
        lines.append(
            f"| {r.name} | {o.n} | {o.n_spoilers} | {_fmt(o.threshold)} | {_fmt(o.recall)} | "
            f"{_fmt(o.false_blur_rate)} | {_fmt(o.precision)} | {_fmt(o.pr_auc)} | {_fmt(o.ece)} |"
        )
    for r in reports:
        lines += ["", f"## Slices: {r.name}", ""]
        for slice_name, groups in r.slices.items():
            lines += [
                f"**By {slice_name}**",
                "",
                "| Group | Sentences | Spoilers | Recall | False-blur rate |",
                "| --- | --- | --- | --- | --- |",
            ]
            for group, res in groups.items():
                lines.append(
                    f"| {group} | {res.n} | {res.n_spoilers} | {_fmt(res.recall)} | "
                    f"{_fmt(res.false_blur_rate)} |"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"
