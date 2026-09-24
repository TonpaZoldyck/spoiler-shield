import math

import numpy as np
import pytest

from spoiler_shield.eval.metrics import (
    confusion_at,
    evaluate,
    expected_calibration_error,
    false_blur_rate,
    gate_recall,
    pr_auc,
    precision_at,
    recall_at,
    threshold_for_recall,
)

# Hand-worked case. Blurring scores >= 0.5 flags items 0, 4 and 5.
Y = [1, 1, 0, 0, 1, 0]
S = [0.9, 0.4, 0.35, 0.1, 0.8, 0.6]


def test_confusion_and_rates_by_hand():
    c = confusion_at(Y, S, 0.5)
    assert (c.tp, c.fp, c.tn, c.fn) == (2, 1, 2, 1)
    assert recall_at(Y, S, 0.5) == pytest.approx(2 / 3)
    assert precision_at(Y, S, 0.5) == pytest.approx(2 / 3)
    assert false_blur_rate(Y, S, 0.5) == pytest.approx(1 / 3)


def test_threshold_is_inclusive():
    assert recall_at([1], [0.5], 0.5) == 1.0


def test_threshold_for_recall_by_hand():
    # Spoiler scores, high to low: 0.9, 0.8, 0.4.
    assert threshold_for_recall(Y, S, 1.0) == 0.4
    assert threshold_for_recall(Y, S, 2 / 3) == 0.8
    assert threshold_for_recall(Y, S, 0.5) == 0.8  # need ceil(1.5) = 2 spoilers
    assert threshold_for_recall(Y, S, 0.3) == 0.9


@pytest.mark.parametrize("seed", range(20))
def test_threshold_for_recall_meets_target_and_is_the_highest_that_does(seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 300)
    s = np.clip(0.3 * y + rng.normal(0.4, 0.2, 300), 0, 1)
    if y.sum() == 0:
        return
    for target in (0.5, 0.8, 0.9, 0.95):
        t = threshold_for_recall(y, s, target)
        assert recall_at(y, s, t) >= target
        higher = s[s > t]
        if higher.size:
            assert recall_at(y, s, higher.min()) < target


def test_pr_auc_perfect_and_empty():
    assert pr_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert math.isnan(pr_auc([0, 0], [0.1, 0.2]))


def test_ece_perfect_and_worst():
    y = [1, 0, 0, 0] * 25
    assert expected_calibration_error(y, [0.25] * 100) == pytest.approx(0.0)
    assert expected_calibration_error([0] * 10, [1.0] * 10) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        expected_calibration_error([0], [1.5])


def test_gate_recall():
    assert gate_recall([1, 1, 1, 0], [True, False, True, False]) == pytest.approx(2 / 3)


def test_rates_are_nan_without_the_relevant_class():
    assert math.isnan(recall_at([0, 0], [0.9, 0.1], 0.5))
    assert math.isnan(false_blur_rate([1, 1], [0.9, 0.1], 0.5))


@pytest.mark.parametrize(
    ("y", "s"),
    [([1, 0], [0.5]), ([1, 2], [0.5, 0.5]), ([1, 0], [float("nan"), 0.1])],
)
def test_bad_inputs_rejected(y, s):
    with pytest.raises(ValueError):
        confusion_at(y, s, 0.5)


def test_evaluate_bundles_everything():
    r = evaluate(Y, S, 0.5)
    assert (r.n, r.n_spoilers) == (6, 3)
    assert r.recall == pytest.approx(2 / 3)
    assert set(r.as_dict()) >= {"recall", "false_blur_rate", "pr_auc", "ece"}
