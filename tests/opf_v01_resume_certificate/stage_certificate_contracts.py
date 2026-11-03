"""Execution certificate for verbatim OPF_ADP V01 AdamW stage and BGD source.

The exact functions are mirrored into vision_adp/native_stage_snapshot.py.
Only the minimum surrounding model is synthetic; the real trainer's AdamW
loop, optimizer serialization, sampler replay and BGD observer run unchanged.
"""
from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from vision_adp.native_stage_snapshot import train_stage
from vision_adp.shadow_bgd_variance import ShadowBGDVariance
from vision_adp.v01_resume import SCHEMA, V01CheckpointStore


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(4, 5)
        self.second = nn.Linear(5, 2)

    def forward(self, x):
        return self.second(torch.relu(self.first(x)))

    def snapshot_arch_and_state(self):
        return {"widths": [5], "state_dict": deepcopy(self.state_dict())}

    def restore_arch_and_state(self, state):
        assert state["widths"] == [5]
        self.load_state_dict(state["state_dict"], strict=True)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = TinyHead()

    def forward(self, x):
        return self.head(x)


def setup(observer):
    torch.manual_seed(104)
    x = torch.linspace(-1.0, 1.0, 64).reshape(16, 4)
    y = torch.tensor([0, 1] * 8)
    train = DataLoader(
        TensorDataset(x, y), batch_size=4, shuffle=True,
        generator=torch.Generator().manual_seed(613),
        num_workers=0,
    )
    valid = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)
    model = TinyModel()
    shadow = (
        ShadowBGDVariance(
            initial_std=0.1, mc_samples=2, tolerance=1e-4,
            patience=3, seed=11,
        )
        if observer else None
    )
    return model, train, valid, shadow


def run(store, observer, *, restored=None, stop_at=None):
    model, train, val, shadow = setup(observer)
    def checkpoint(state):
        if store is not None:
            store.save({
                "schema": SCHEMA, "science_id": "3" * 64,
                "kind": "stage", "controller": {"next_stage": 0},
                "stage": state,
            })
        if stop_at == (state["epoch"], state["batch"]):
            raise RuntimeError("injected interruption after atomic minibatch commit")

    val_loss, trace = train_stage(
        model, train, val, torch.device("cpu"),
        epochs=2, lr=0.003, weight_decay=0.01,
        shadow=shadow,
        resume_state=restored,
        checkpoint_callback=checkpoint if store is not None else None,
    )
    return {
        "val_loss": val_loss,
        "trace": trace,
        "weights": {k: v.detach().clone() for k, v in model.state_dict().items()},
        "shadow": None if shadow is None else shadow.state_dict(),
        "rng": torch.get_rng_state().clone(),
    }


@pytest.mark.parametrize("observer", [False, True])
@pytest.mark.parametrize("stop_at", [(0, 2), (1, 0)])
def test_real_v01_stage_restart_equals_uninterrupted_cpu_training(
    tmp_path, observer, stop_at,
):
    original_store = V01CheckpointStore(
        tmp_path / "original", "3" * 64
    )
    baseline = run(original_store, observer)
    interrupted_store = V01CheckpointStore(
        tmp_path / "interrupted", "3" * 64
    )
    with pytest.raises(RuntimeError, match="injected interruption"):
        run(interrupted_store, observer, stop_at=stop_at)
    checkpoint = interrupted_store.load()
    assert checkpoint["kind"] == "stage"
    assert checkpoint["stage"]["epoch"] == stop_at[0]
    assert checkpoint["stage"]["batch"] == stop_at[1]
    assert interrupted_store.restored_generation == "current"

    restarted = run(
        interrupted_store, observer, restored=checkpoint["stage"]
    )
    assert baseline["val_loss"] == restarted["val_loss"]
    assert baseline["trace"] == restarted["trace"]
    assert baseline["weights"].keys() == restarted["weights"].keys()
    assert all(
        torch.equal(tensor, restarted["weights"][name])
        for name, tensor in baseline["weights"].items()
    )
    assert torch.equal(baseline["rng"], restarted["rng"])
    if observer:
        a, b = baseline["shadow"], restarted["shadow"]
        assert a["observations"] == b["observations"]
        assert a["streak"] == b["streak"]
        assert a["contraction"] == b["contraction"]
        assert all(torch.equal(t, b["sigma"][n]) for n,t in a["sigma"].items())
        assert all(torch.equal(t, b["rng"][n]) for n,t in a["rng"].items())


@pytest.mark.parametrize("observer", [False, True])
def test_real_v01_stage_can_replay_from_previous_valid_minibatch(
    tmp_path, observer,
):
    baseline_store = V01CheckpointStore(
        tmp_path / "original", "3" * 64
    )
    baseline = run(baseline_store, observer)
    broken = V01CheckpointStore(tmp_path / "broken", "3" * 64)
    with pytest.raises(RuntimeError, match="injected interruption"):
        run(broken, observer, stop_at=(0, 2))
    assert broken.previous_payload.exists()
    broken.payload.write_bytes(b"corrupted recent AdamW step")
    recovered = broken.load()
    assert broken.restored_generation == "previous"
    assert recovered["stage"]["batch"] == 1
    restarted = run(broken, observer, restored=recovered["stage"])
    assert baseline["val_loss"] == restarted["val_loss"]
    assert baseline["trace"] == restarted["trace"]
    assert all(
        torch.equal(tensor, restarted["weights"][name])
        for name, tensor in baseline["weights"].items()
    )
    if observer:
        a, b = baseline["shadow"], restarted["shadow"]
        assert all(torch.equal(t, b["sigma"][n]) for n,t in a["sigma"].items())
