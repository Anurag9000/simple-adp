"""Post-selection multiclass calibration metrics for visual classifiers.

Used *after* model selection. Never changes SearchController/TaskAdapter
objectives, class mappings, or the validation-only architecture decision.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .tasks import ClassificationTask


def visual_calibration_metrics(
    task: ClassificationTask,
    truth: Any,
    probabilities: Any,
    *,
    bins: int = 10,
) -> dict[str, float]:
    """Deterministic top-label ECE and multiclass Brier score.

    ECE groups maximum predicted confidence into equally sized intervals
    [0,1/bins), ... , [(bins-1)/bins,1] and computes a sample-weighted
    absolute gap between mean confidence and empirical accuracy. On tiny
    smoke sets it is descriptive only and is not evidence of calibration.
    """
    if not isinstance(task, ClassificationTask):
        raise TypeError("calibration requires the fitted ClassificationTask label map")
    if not isinstance(bins, int) or isinstance(bins, bool) or bins < 2:
        raise ValueError("calibration bins must be an integer >= 2")
    encoded = task.encode_targets(truth, device=torch.device("cpu")).numpy()
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape != (
        len(encoded), task.output_dim,
    ) or not len(encoded):
        raise ValueError("calibration probabilities must be [N, num_training_classes]")
    if not np.isfinite(probabilities).all() or (
        np.any(probabilities < 0) or np.any(probabilities > 1)
    ):
        raise ValueError("calibration probabilities must be finite and between 0 and 1")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-5):
        raise ValueError("calibration probabilities must sum to one per example")

    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = (prediction == encoded).astype(np.float64)
    slots = np.minimum(np.floor(confidence * bins).astype(np.int64), bins - 1)
    ece = 0.0
    for index in range(bins):
        mask = slots == index
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    brier = float(np.mean(np.sum(
        np.square(probabilities - np.eye(task.output_dim)[encoded]), axis=1,
    )))
    return {
        f"top_label_ece_{bins}": float(ece),
        "multiclass_brier": brier,
    }
