"""Signed actual minibatch-forward operator work in native visual training."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from simple_adp.config import SearchSettings
from simple_adp.tasks import ClassificationTask
from simple_adp.models import TabularMLP
from simple_adp.training import StageTrainer
from simple_adp.visual_inner import ResidualAdapterSpec
from simple_adp.visual_search import VisualResidualSearch


def _toy():
    rng = np.random.default_rng(401)
    images = rng.normal(size=(24, 3, 8, 8)).astype(np.float32)
    labels = np.asarray([0, 1] * 12)
    return images[:16], labels[:16], (images[16:], labels[16:])


def _spec():
    return ResidualAdapterSpec(
        input_channels=3, backbone_channels=4, local_dim=4,
        classes=2, image_height=8, image_width=8,
        site_id="adapter:block1", fixed_seed=17,
    )


def _settings():
    return SearchSettings(
        seed=42, device="cpu", init_width=2, init_depth=1,
        max_width=4, max_depth=2, max_neurons=12,
        max_candidates=1, max_epochs=1, batch_size=4,
        patience_es=2, patience_width_exp=1, patience_depth_exp=1,
    )


def test_stage_observer_counts_exact_batch_sizes_and_ignores_validation():
    task = ClassificationTask()
    targets = np.asarray([0, 1, 0, 1, 0, 1, 0])
    task.fit_targets(targets)
    train_x = torch.arange(28, dtype=torch.float32).reshape(7, 4) / 100
    train_y = task.encode_targets(targets, device="cpu")
    valid_x, valid_y = train_x[:4], train_y[:4]
    model = TabularMLP(4, (3,), 2)
    settings = SearchSettings(seed=19, device="cpu", max_epochs=2, batch_size=3, patience_es=4)
    state = StageTrainer(
        task, settings, track_visual_training_forward_macs=True,
    ).train_stage(model, train_x, train_y, valid_x, valid_y)
    cost = state.training_forward_macs
    assert cost is not None and cost["complete"] is True
    assert cost["recorded_training_forward_batches"] == 6
    assert cost["recorded_training_samples"] == 14
    # Each sample traverses 4->3 then 3->2: 12+6 MACs.
    assert cost["observed_linear_conv_training_forward_macs"] == 14 * 18
    assert cost["analytical_mha_training_forward_macs"] == 0
    assert cost["full_training_flops"] is None
    assert state.checkpoint_state["training_forward_macs"] == cost


def test_opt_in_visual_native_trace_covers_all_trained_candidates_and_work():
    x, y, valid = _toy()
    search = VisualResidualSearch(
        "ADP-W-RR", _spec(), settings=_settings(),
        track_training_forward_macs=True,
    )
    search.fit(x, y, validation_data=valid)
    ledger = search.search_state_["training_forward_compute"]
    stages = search.search_state_["trained_forward_mac_trace"]
    assert len(stages) == search.search_state_["training_budget"]["trained_stages"] == 2
    assert ledger["complete"] and search.search_state_["trained_forward_mac_trace_complete"]
    assert ledger["recorded_training_forward_batches"] == 8
    assert ledger["recorded_training_samples"] == 32
    assert ledger["partial_training_forward_macs"] == sum(
        record["partial_training_forward_macs"] for record in stages
    )
    assert ledger["partial_training_forward_macs"] > 0
    assert ledger["full_training_flops"] is None
    assert all(row["complete"] for row in stages)
    assert [row["stage"] for row in stages] == ["initial", "width"]
    assert not any(model._forward_hooks for model in search.model_.modules())


def test_exact_interruption_and_relaunch_preserve_observed_forward_work(tmp_path):
    x, y, valid = _toy()
    original = VisualResidualSearch(
        "ADP-W-RR", _spec(), settings=_settings(),
        track_training_forward_macs=True,
    ).fit(x, y, validation_data=valid)
    checkpoint = tmp_path / "track-actual-train-macs.pt"

    def pause_after_batch(info):
        if info["phase"] == "initial" and info["batch_cursor"] == 2:
            raise RuntimeError("simulated batch interruption")

    interrupted = VisualResidualSearch(
        "ADP-W-RR", _spec(), settings=_settings(),
        track_training_forward_macs=True,
    )
    with pytest.raises(RuntimeError, match="simulated batch interruption"):
        interrupted.fit(
            x, y, validation_data=valid, checkpoint_path=checkpoint,
            progress_callback=pause_after_batch,
        )
    assert checkpoint.is_file()

    restarted = VisualResidualSearch(
        "ADP-W-RR", _spec(), settings=_settings(),
        track_training_forward_macs=True,
    ).fit(x, y, validation_data=valid, checkpoint_path=checkpoint)
    assert restarted.search_state_["training_forward_compute"] == original.search_state_[
        "training_forward_compute"
    ]
    assert restarted.search_state_["trained_forward_mac_trace"] == original.search_state_[
        "trained_forward_mac_trace"
    ]
    assert restarted.history_ == original.history_
    assert np.array_equal(restarted.predict(valid[0]), original.predict(valid[0]))


def test_opt_in_training_compute_invalidates_older_untracked_checkpoint(tmp_path):
    x, y, valid = _toy()
    path = tmp_path / "original-legacy-untracked.pt"
    old = VisualResidualSearch("MLP-Fixed", _spec(), settings=_settings(), fixed_grid=((2,),))
    old.fit(x, y, validation_data=valid, checkpoint_path=path)
    assert "training_forward_compute" not in old.search_state_
    with pytest.raises(ValueError, match="signature"):
        VisualResidualSearch(
            "MLP-Fixed", _spec(), settings=_settings(),
            fixed_grid=((2,),), track_training_forward_macs=True,
        ).fit(x, y, validation_data=valid, checkpoint_path=path)


def test_training_forward_tracker_input_type_validation():
    with pytest.raises(ValueError, match="boolean"):
        VisualResidualSearch(
            "ADP-W-RR", _spec(), settings=_settings(),
            track_training_forward_macs=1,
        )
    with pytest.raises(ValueError, match="boolean"):
        StageTrainer(
            ClassificationTask(), _settings(),
            track_visual_training_forward_macs="yes",
        )
