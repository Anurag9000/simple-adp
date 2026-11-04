"""Fail-closed receipt verification and seed-level comparison contracts."""
import copy
import hashlib
import json

import pytest

from scripts.analyze_bounded_iris_pair_results import validate_receipt, analyze


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt(tmp_path, seed):
    path = tmp_path / f"seed-{seed}"
    path.mkdir()
    split = path / "iris_split.npz"
    split.write_bytes(b"fixed-source-split")
    arms = {}
    for name, trigger, accuracy in (("original", "loss", 0.67),
                                    ("bayesian", "variance_stagnation", 0.63)):
        folder = path / name
        folder.mkdir()
        model = folder / "estimator.model.pt"
        model.write_bytes((name + str(seed)).encode())
        arms[name] = {
            "growth_trigger": trigger,
            "selected_architecture": [3],
            "model_artifact_sha256": _hash(model),
            "model_artifact_size_bytes": model.stat().st_size,
            "test_metrics": {
                "accuracy": accuracy, "balanced_accuracy": accuracy,
                "macro_f1": accuracy, "log_loss": 1 - accuracy,
            },
        }
    data = {
        "schema": "bounded-iris-original-vs-shadow-bgd-v1",
        "evidence_grade": "BOUNDED_REAL_DATA_PASSED",
        "scientific_settings_diff": {"expansion_criterion": ["loss", "variance_stagnation"]},
        "source_commit": "a" * 40,
        "dataset_source": "sklearn.datasets.load_iris",
        "dataset_sha256": "data",
        "split_sizes": {"train": 90, "validation": 30, "test": 30},
        "split_archive_sha256": _hash(split),
        "split_legacy_sha256": "legacy",
        "split_partition_sha256_v2": "strong",
        "sklearn_version": "1.9",
        "route": {"model_id": "ADP-W-All"},
        "seed": seed,
        "arms": arms,
    }
    receipt = path / "pair_receipt.json"
    receipt.write_text(json.dumps(data))
    return receipt


def test_two_realistic_receipts_can_be_aggregated(tmp_path):
    a, b = _receipt(tmp_path, 17), _receipt(tmp_path, 19)
    result = analyze((a, b), bootstrap_samples=1000)
    assert result["distinct_seed_count"] == 2
    assert result["paired_summary"]["accuracy"]["paired_mean_bayesian_minus_original"] == pytest.approx(-0.04)
    assert result["conditioned_on_one_fixed_iris_split"]


@pytest.mark.parametrize("tamper", ["model", "split", "wrong_criterion"])
def test_tampered_receipts_fail_closed(tmp_path, tamper):
    file = _receipt(tmp_path, 21)
    if tamper == "model":
        (file.parent / "bayesian" / "estimator.model.pt").write_bytes(b"corrupt")
    elif tamper == "split":
        (file.parent / "iris_split.npz").write_bytes(b"wrong")
    else:
        payload = json.loads(file.read_text())
        payload["arms"]["bayesian"]["growth_trigger"] = "loss"
        file.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        validate_receipt(file)


def test_duplicate_seed_and_mismatched_source_revision_rejected(tmp_path):
    a, b = _receipt(tmp_path, 10), _receipt(tmp_path, 11)
    with pytest.raises(ValueError, match="duplicate model seeds"):
        analyze((a, a), bootstrap_samples=1000)
    payload = json.loads(b.read_text())
    payload["source_commit"] = "b" * 40
    b.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="source_commit"):
        analyze((a, b), bootstrap_samples=1000)


def test_comparison_requires_two_seeds(tmp_path):
    receipt = _receipt(tmp_path, 8)
    with pytest.raises(ValueError, match="at least two"):
        analyze((receipt,), bootstrap_samples=1000)
