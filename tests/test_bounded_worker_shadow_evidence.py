"""Posterior growth gate diagnostic must read real-format checkpoint payloads."""
from __future__ import annotations

import hashlib
import copy

import pytest
import torch

from scripts.audit_bounded_worker_shadow_evidence import inspect_shadow_checkpoint


def _make(tmp_path, *, contraction=0.01, streak=0, observations=7):
    path = tmp_path / "estimator.checkpoint.pt"
    posterior = {
        "schema": 1,
        "config": (0.5, 2, 0.0001, 3, 73),
        "observations": observations,
        "streak": streak,
        "contraction": contraction,
        "sigma": {
            "layer.weight": torch.full((3, 4), 0.5),
            "layer.bias": torch.full((3,), 0.2),
        },
        "rng": {},
    }
    payload = {
        "stage_state": {
            "stage_status": "completed",
            "shadow_bgd_variance": posterior,
            "best_shadow_bgd_variance": copy.deepcopy(posterior),
        },
        "search_history": [
            {"architecture": [3], "score": 1.0, "accepted": True, "stage": "initial"}
        ],
    }
    torch.save({"format_version": 1, "signature": "receipt", "payload": payload}, path)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, checksum


def test_not_stagnated_best_epoch_is_not_eligible_for_growth(tmp_path):
    path, checksum = _make(tmp_path, observations=24, streak=0)
    proof = inspect_shadow_checkpoint(path, sha256=checksum)
    assert proof["gate_ready_at_best_stage"] is False
    assert proof["posterior_observations_at_best_stage"] == 24
    assert proof["minimum_sampled_posterior_forwards_from_successful_observations"] == 48
    assert proof["sigma_parameter_coordinates"] == 15


def test_stagnated_three_successive_observations_are_eligible(tmp_path):
    path, checksum = _make(tmp_path, contraction=0.0, streak=3, observations=3)
    assert inspect_shadow_checkpoint(path, sha256=checksum)["gate_ready_at_best_stage"]


def test_tampered_checkpoint_bytes_are_rejected(tmp_path):
    path, checksum = _make(tmp_path)
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="SHA-256"):
        inspect_shadow_checkpoint(path, sha256=checksum)


def test_nonpositive_sigma_fails_closed(tmp_path):
    path, _ = _make(tmp_path)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    raw["payload"]["stage_state"]["shadow_bgd_variance"]["sigma"]["layer.weight"][0, 0] = 0
    torch.save(raw, path)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="sigma"):
        inspect_shadow_checkpoint(path, sha256=checksum)
