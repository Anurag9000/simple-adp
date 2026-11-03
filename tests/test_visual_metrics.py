"""Calibration is post-selection, class-mapped and numerically defined."""
import numpy as np
import pytest

from simple_adp.tasks import ClassificationTask
from simple_adp.visual_metrics import visual_calibration_metrics


def task():
    t = ClassificationTask()
    t.fit_targets(np.asarray([7, 9, 7, 9]))
    return t


def test_ece_perfect_predictions_zero_brier_and_one_confidence_edge():
    p = np.asarray([[1., 0.], [0., 1.], [1., 0.]])
    result = visual_calibration_metrics(task(), np.array([7, 9, 7]), p)
    assert result == {"top_label_ece_10": 0.0, "multiclass_brier": 0.0}


def test_ece_known_gap_with_noncontiguous_class_labels():
    # Confidence .8, accuracy 1 => ECE .2; multiclass Brier = .08.
    p = np.array([[0.8, 0.2], [0.2, 0.8]], dtype=float)
    result = visual_calibration_metrics(task(), np.array([7, 9]), p)
    assert result["top_label_ece_10"] == pytest.approx(0.2)
    assert result["multiclass_brier"] == pytest.approx(0.08)


def test_brier_and_ece_are_not_equivalent():
    # Perfect calibration of aggregate confidence is compatible with 50%
    # prediction accuracy; the Brier score still penalizes uncertainty.
    p = np.asarray([[0.5, 0.5], [0.5, 0.5]])
    m = visual_calibration_metrics(task(), [7, 9], p)
    assert m["top_label_ece_10"] == pytest.approx(0.0)
    assert m["multiclass_brier"] == pytest.approx(0.5)


@pytest.mark.parametrize("invalid", [
    [[-0.2, 1.2]],
    [[float("nan"), 0.5]],
    [[0.4, 0.5]],
    [[0.9, 0.1, 0.0]],
])
def test_reject_invalid_probability_schema(invalid):
    with pytest.raises(ValueError):
        visual_calibration_metrics(task(), [7], invalid)


def test_bins_reject_boolean_and_low_values():
    with pytest.raises(ValueError, match="integer"):
        visual_calibration_metrics(task(), [7], [[1, 0]], bins=True)
    with pytest.raises(ValueError, match="integer"):
        visual_calibration_metrics(task(), [7], [[1, 0]], bins=1)
