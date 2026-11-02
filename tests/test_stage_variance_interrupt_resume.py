"""An actual mid-epoch variance-stage crash must resume identically."""
from __future__ import annotations

import copy
import os

import pytest
import torch

from simple_adp.config import SearchSettings
from simple_adp.models import TabularMLP
from simple_adp.tasks import RegressionTask
from simple_adp.training import StageTrainer


def _train_settings():
    return SearchSettings(
        init_width=3,
        init_depth=1,
        max_width=4,
        max_depth=2,
        max_neurons=16,
        max_epochs=3,
        batch_size=4,
        patience_es=4,
        expansion_criterion="variance_stagnation",
        seed=71,
    )


def _observer_equal(a, b):
    assert a["schema"] == b["schema"]
    assert a["config"] == b["config"]
    assert a["observations"] == b["observations"]
    assert a["streak"] == b["streak"]
    assert a["contraction"] == b["contraction"]
    assert set(a["sigma"]) == set(b["sigma"])
    assert set(a["rng"]) == set(b["rng"])
    for key in a["sigma"]:
        assert torch.equal(a["sigma"][key], b["sigma"][key]), key
    for key in a["rng"]:
        assert torch.equal(a["rng"][key], b["rng"][key]), key


def test_variance_mid_epoch_crash_matches_uninterrupted_training(tmp_path):
    torch.manual_seed(100)
    original = TabularMLP(3, [3], 1)
    x = torch.randn(16, 3)
    y = torch.randn(16, 1)
    task = RegressionTask()
    task.fit_targets([0.0, 1.0])
    config = _train_settings()

    # The uninterrupted run and the crash run start with identical weights,
    # model-local seed, data, and global RNG.
    starting_rng = torch.get_rng_state().clone()
    full_model = copy.deepcopy(original)
    full = StageTrainer(task, config)
    torch.set_rng_state(starting_rng.clone())
    full_result = full.train_stage(full_model, x, y, x, y)

    checkpoint = tmp_path / "stage.pt"
    interrupted_model = copy.deepcopy(original)

    def save_then_crash(payload):
        temporary = checkpoint.with_suffix(".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, checkpoint)
        if payload["batches_done"] == 2:
            raise RuntimeError("intentional failure after durable minibatch")

    torch.set_rng_state(starting_rng.clone())
    with pytest.raises(RuntimeError, match="intentional failure"):
        StageTrainer(task, config, checkpoint_callback=save_then_crash).train_stage(
            interrupted_model, x, y, x, y
        )
    persisted = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert persisted["epoch"] == 0
    assert persisted["batch_cursor"] == 2
    assert persisted["batches_done"] == 2
    assert persisted["best_shadow_bgd_variance"] is None
    assert persisted["shadow_bgd_variance"]["observations"] == 2

    resumed_model = copy.deepcopy(original)
    resumed_result = StageTrainer(task, config).train_stage(
        resumed_model, x, y, x, y, resume_state=persisted
    )

    assert resumed_result.best_score == full_result.best_score
    assert resumed_result.epochs == full_result.epochs
    assert resumed_result.batches == full_result.batches
    assert resumed_result.variance_stagnated == full_result.variance_stagnated
    assert resumed_result.posterior_contraction == full_result.posterior_contraction
    for name, weights in full_model.state_dict().items():
        assert torch.equal(weights, resumed_model.state_dict()[name]), name

    for field in ("shadow_bgd_variance", "best_shadow_bgd_variance"):
        _observer_equal(full_result.checkpoint_state[field],
                        resumed_result.checkpoint_state[field])
    assert torch.equal(
        full_result.checkpoint_state["generator_state"],
        resumed_result.checkpoint_state["generator_state"],
    )
