"""Held-out materialization must follow visual ADP search, not precede it.

Both opt-in coordinated adapter families are exercised using the real toy
provider and a genuine fixed-model training stage. A guarded row source fails
if any official test image or target is accessed before fit() returns.
"""
from __future__ import annotations

import numpy as np
import pytest

from simple_adp.config import SearchSettings
from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.vision import (
    VisionDatasetProvider, VisionInnerModelAdapter, VisionModelAdapter,
)
from simple_adp.visual import VisualADPEstimator
from simple_adp.visual_search import VisualResidualSearch


class GuardedRows:
    def __init__(self, rows, forbidden, fitted):
        self.rows = rows
        self.forbidden = set(int(i) for i in forbidden)
        self.fitted = fitted
        self.held_out_reads = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        requested = set(int(i) for i in np.asarray(index).reshape(-1))
        if requested & self.forbidden:
            if not self.fitted["completed"]:
                raise AssertionError("held-out image/label accessed before ADP fit returned")
            self.held_out_reads += 1
        return self.rows[index]


@pytest.mark.parametrize(
    ("adapter_class", "fit_class"),
    [
        (VisionModelAdapter, VisualADPEstimator),
        (VisionInnerModelAdapter, VisualResidualSearch),
    ],
)
def test_coordinated_visual_test_rows_remain_unread_until_fit(
    monkeypatch, adapter_class, fit_class,
):
    provider = VisionDatasetProvider()
    ref = provider.list_tasks("toy")[0]
    data = provider.load_task(ref)
    runner_settings = RunnerSettings(
        tasks=("vision",), width_to_depth_width=2, width_to_depth_depth=1,
    )
    split = provider.list_splits(ref, data, runner_settings)[0]
    fitted = {"completed": False}
    data.X = GuardedRows(data.X, split.test_indices, fitted)
    data.y = GuardedRows(data.y, split.test_indices, fitted)

    original_fit = fit_class.fit

    def tracked_fit(self, *args, **kwargs):
        value = original_fit(self, *args, **kwargs)
        fitted["completed"] = True
        return value

    monkeypatch.setattr(fit_class, "fit", tracked_fit)
    adapter = adapter_class()
    route = next(
        item for item in adapter.enumerate_routes(settings=runner_settings)
        if item.model_id == "MLP-Fixed"
    )
    result = adapter.fit_predict(
        route, data=data, split=split, seed=7,
        settings=SearchSettings(
            device="cpu", seed=7,
            init_width=2, init_depth=1,
            max_width=3, max_neurons=32,
            max_epochs=1, max_candidates=1,
            batch_size=16, patience_es=1,
        ),
    )
    assert fitted["completed"]
    assert data.X.held_out_reads > 0
    assert data.y.held_out_reads > 0
    assert result["test_metrics"]["accuracy"] >= 0
    assert result["metric_directions"]["top_label_ece_10"] == "min"
    assert result["metric_directions"]["multiclass_brier"] == "min"
    for partition in ("validation_metrics", "test_metrics"):
        assert 0.0 <= result[partition]["top_label_ece_10"] <= 1.0
        assert 0.0 <= result[partition]["multiclass_brier"] <= 2.0
    accounting = result["parameter_accounting"]
    assert accounting["total_parameters"] == (
        accounting["frozen_parameters"] + accounting["trainable_parameters"]
    )
    assert accounting["trainable_parameters"] == (
        accounting["currently_trainable_eligible_mlp_parameters"]
        + accounting["other_trainable_parameters"]
    )
    expected_site = "head" if adapter_class is VisionModelAdapter else "adapter:block1"
    assert expected_site in accounting["eligible_growth_sites"]
    compute = result["inference_compute"]
    assert compute["batch_size"] == 1
    assert compute["observed_linear_conv_macs_per_image"] > 0
    assert compute["whole_model_flops"] is None
    if adapter_class is VisionInnerModelAdapter:
        assert result["training_budget"]["trained_stages"] >= 1
        assert result["training_budget"]["optimizer_steps"] >= 1
