"""Fail-closed identity, pair metric and checkpoint verification for Diabetes."""
from __future__ import annotations

import hashlib
import json

import pytest

from scripts.analyze_bounded_diabetes_pair_results import analyze, verify_receipt


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path, seed):
    parent = tmp_path / f"seed-{seed}"
    parent.mkdir()
    split = parent / "diabetes_split.npz"
    split.write_bytes(b"real-split-test-digest-fixture")
    arms = {}
    for name, criterion, rmse in (("original", "loss", 165.0),
                                   ("bayesian", "variance_stagnation", 165.5)):
        folder = parent / name
        folder.mkdir()
        model = folder / "estimator.model.pt"
        ckpt = folder / "checkpoint.pt"
        model.write_bytes((name + str(seed)).encode())
        ckpt.write_bytes((name + "-checkpoint" + str(seed)).encode())
        arms[name] = {
            "growth_trigger": criterion,
            "model_artifact_sha256": _sha(model),
            "model_artifact_size_bytes": model.stat().st_size,
            "checkpoint_sha256": _sha(ckpt),
            "checkpoint_size_bytes": ckpt.stat().st_size,
            "selected_architecture": [3],
            "test_metrics": {"rmse": rmse, "mae": rmse - 10, "r2": -3.0},
        }
    payload = {
        "schema": "bounded-diabetes-original-vs-shadow-bgd-v1",
        "evidence_grade": "BOUNDED_REAL_DATA_PASSED",
        "scientific_settings_diff": {"expansion_criterion": ["loss", "variance_stagnation"]},
        "source_commit": "a" * 40,
        "seed": seed,
        "dataset_source": "sklearn.datasets.load_diabetes",
        "dataset_sha256": "diabetes-sha",
        "sklearn_version": "1.9.1",
        "split_sha256": "legacy",
        "split_partition_sha256_v2": "stronger",
        "split_archive_sha256": _sha(split),
        "split_sizes": {"train": 264, "validation": 89, "test": 89},
        "route": {"model_id": "ADP-W-All"},
        "arms": arms,
    }
    file = parent / "pair_receipt.json"
    file.write_text(json.dumps(payload))
    return file


def test_verified_paired_regression_bootstrap(tmp_path):
    a, b = _fixture(tmp_path, 73), _fixture(tmp_path, 74)
    result = analyze([a, b], bootstrap_samples=1000)
    assert result["seed_count"] == 2
    assert result["paired_summary"]["rmse"]["mean_difference_bayesian_minus_original"] == pytest.approx(0.5)
    assert result["conditional_on_one_diabetes_split"]


@pytest.mark.parametrize("tamper", ["checkpoint", "model", "split", "trigger", "metric"])
def test_tampering_detected(tmp_path, tamper):
    receipt = _fixture(tmp_path, 71)
    if tamper == "checkpoint":
        (receipt.parent / "bayesian" / "checkpoint.pt").write_bytes(b"bad")
    elif tamper == "model":
        (receipt.parent / "original" / "estimator.model.pt").write_bytes(b"bad")
    elif tamper == "split":
        (receipt.parent / "diabetes_split.npz").write_bytes(b"bad")
    else:
        payload = json.loads(receipt.read_text())
        if tamper == "trigger":
            payload["arms"]["bayesian"]["growth_trigger"] = "loss"
        else:
            payload["arms"]["original"]["test_metrics"]["rmse"] = float("nan")
        receipt.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        verify_receipt(receipt)


def test_duplicate_seeds_and_mismatched_dataset_revisions_rejected(tmp_path):
    a = _fixture(tmp_path, 73)
    b = _fixture(tmp_path, 74)
    with pytest.raises(ValueError, match="duplicate model seed"):
        analyze([a, a], bootstrap_samples=1000)
    payload = json.loads(b.read_text())
    payload["source_commit"] = "b" * 40
    b.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="source_commit"):
        analyze([a, b], bootstrap_samples=1000)


def test_must_have_at_least_two_seeds(tmp_path):
    with pytest.raises(ValueError, match="two independent"):
        analyze([_fixture(tmp_path, 7)], bootstrap_samples=1000)
