"""Standalone durable V01 checkpoint journal tests; no OPF model imports."""
from __future__ import annotations

import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from vision_adp.v01_resume import (
    SCHEMA, V01CheckpointStore, capture_rng, restore_rng,
    science_checkpoint_identity,
)


def _state(identity: str, index: int) -> dict:
    return {
        "schema": SCHEMA, "science_id": identity,
        "kind": "stage",
        "controller": {"next_stage": 1},
        "stage": {
            "batch": index,
            "weights": torch.arange(6).reshape(2, 3).float() + index,
            "optimizer": {"moment": torch.ones(2, 3) * index},
        },
    }


def test_two_generation_journal_always_has_valid_last_commit(tmp_path):
    identity = "a" * 64
    journal = V01CheckpointStore(tmp_path / "output", identity)
    assert journal.load() is None
    journal.save(_state(identity, 1))
    assert journal.load()["stage"]["batch"] == 1
    assert journal.restored_generation == "current"
    journal.save(_state(identity, 2))
    assert journal.load()["stage"]["batch"] == 2
    assert journal._read_pair(
        journal.previous_payload, journal.previous_receipt,
    )["stage"]["batch"] == 1
    journal.save(_state(identity, 3))
    assert journal.load()["stage"]["batch"] == 3
    assert journal._read_pair(
        journal.previous_payload, journal.previous_receipt,
    )["stage"]["batch"] == 2
    journal.clear()
    assert journal.load() is None
    assert not journal.exists()


def test_corrupt_newest_falls_back_to_verified_older_and_never_corrupt_both(tmp_path):
    identity = "b" * 64
    journal = V01CheckpointStore(tmp_path / "output", identity)
    journal.save(_state(identity, 1))
    journal.save(_state(identity, 2))
    journal.payload.write_bytes(b"invalid-latest-payload")
    recovered = journal.load()
    assert recovered["stage"]["batch"] == 1
    assert journal.restored_generation == "previous"
    journal.save(_state(identity, 3))
    assert journal.load()["stage"]["batch"] == 3
    journal.previous_payload.write_bytes(b"corrupt-old")
    journal.payload.write_bytes(b"corrupt-new")
    with pytest.raises(ValueError, match="no valid committed generation"):
        journal.load()


def test_killed_interpreter_during_atomic_replace_keeps_verified_rollback(tmp_path):
    identity = "c" * 64
    directory = tmp_path / "output"
    journal = V01CheckpointStore(directory, identity)
    journal.save(_state(identity, 1))
    journal.save(_state(identity, 2))
    code = r'''
import os
import sys
from pathlib import Path
import torch
from vision_adp.v01_resume import V01CheckpointStore, SCHEMA
journal = V01CheckpointStore(Path(sys.argv[1]), "c" * 64)
replace = os.replace

def hard_exit(src, dst):
    replace(src, dst)
    if Path(dst) == journal.payload:
        os._exit(87)

os.replace = hard_exit
journal.save({
    "schema": SCHEMA, "science_id": "c" * 64,
    "kind": "stage", "controller": {"next_stage": 1},
    "stage": {"batch": 3, "weights": torch.ones(2), "optimizer": {}},
})
'''
    env = os.environ.copy()
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", code, str(directory)],
        cwd=root, env=env, capture_output=True, text=True,
        timeout=45, check=False,
    )
    assert result.returncode == 87, result.stderr
    prior = V01CheckpointStore(directory, identity)
    assert prior.load()["stage"]["batch"] == 2
    assert prior.restored_generation == "previous"
    prior.save(_state(identity, 4))
    assert prior.load()["stage"]["batch"] == 4


def test_wrong_science_and_unknown_checkpoint_files_fail_closed(tmp_path):
    directory = tmp_path / "output"
    journal = V01CheckpointStore(directory, "d" * 64)
    journal.save(_state("d" * 64, 1))
    with pytest.raises(ValueError, match="identity"):
        V01CheckpointStore(directory, "e" * 64).load()
    foreign = journal.root / "do-not-delete-research.txt"
    foreign.write_text("unowned")
    with pytest.raises(ValueError, match="unexpected files"):
        journal.load()
    assert foreign.read_text() == "unowned"
    foreign.unlink()
    marker = journal.root / ".pending-symlink.pt"
    marker.symlink_to(tmp_path)
    with pytest.raises(ValueError, match="symlink"):
        journal.load()
    marker.unlink()
    assert journal.load()["stage"]["batch"] == 1


def test_checkpoint_identity_covers_threading_determinism_and_sources(monkeypatch):
    from vision_adp import v01_resume

    arguments = {
        "matched_digest": "f" * 64,
        "criterion": "loss",
        "device": torch.device("cpu"),
        "source_digests": {"v01.py": "abc"},
    }
    first = science_checkpoint_identity(**arguments)
    assert len(first) == 64
    assert first != science_checkpoint_identity(
        **{**arguments, "criterion": "variance_stagnation"}
    )
    assert first != science_checkpoint_identity(
        **{**arguments, "source_digests": {"v01.py": "changed"}}
    )
    assert first != science_checkpoint_identity(
        **arguments, library_versions={"torchvision": "0.25.0"}
    )
    assert science_checkpoint_identity(
        **arguments, library_versions={"torchvision": "0.25.0"}
    ) != science_checkpoint_identity(
        **arguments, library_versions={"torchvision": "0.26.0"}
    )
    original = torch.get_num_threads()
    monkeypatch.setattr(v01_resume.torch, "get_num_threads", lambda: original + 1)
    assert first != science_checkpoint_identity(**arguments)


def test_global_rng_round_trip_is_exact_for_python_numpy_and_torch_cpu():
    random.seed(192)
    np.random.seed(192)
    torch.manual_seed(192)
    original = capture_rng()
    sampled = (random.random(), np.random.random(), torch.randn(5))
    restore_rng(original)
    replay = (random.random(), np.random.random(), torch.randn(5))
    assert sampled[:2] == replay[:2]
    assert torch.equal(sampled[2], replay[2])
