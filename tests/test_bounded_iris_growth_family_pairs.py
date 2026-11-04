"""Four native growth-family source/variance receipt strict audit contracts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit_bounded_iris_growth_families import (
    EXPECTED_FAMILIES, audit_family_receipts,
)


def _hash(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def _fake(tmp_path, model_id, seed=73):
    folder = tmp_path / model_id
    folder.mkdir()
    split = folder / "iris_split.npz"
    split.write_bytes(b"same split across four families")
    arms = {}
    for key, trigger in (("original", "loss"), ("bayesian", "variance_stagnation")):
        sub = folder / key
        sub.mkdir()
        model = sub / "estimator.model.pt"
        model.write_bytes((model_id + key).encode())
        arms[key] = {
            "growth_trigger": trigger,
            "model_artifact_sha256": _hash(model),
            "model_artifact_size_bytes": model.stat().st_size,
            "selected_architecture": [3],
            "test_metrics": {
                "accuracy": 0.5, "balanced_accuracy": 0.5,
                "macro_f1": 0.5, "log_loss": 0.9,
            },
        }
    raw = {
        "schema": "bounded-iris-original-vs-shadow-bgd-v1",
        "evidence_grade": "BOUNDED_REAL_DATA_PASSED",
        "source_commit": "a"*40,
        "scientific_settings_diff": {"expansion_criterion": ["loss", "variance_stagnation"]},
        "dataset_source": "sklearn.datasets.load_iris",
        "dataset_sha256": "dataset",
        "split_sizes": {"train": 90, "validation": 30, "test": 30},
        "split_archive_sha256": _hash(split),
        "split_legacy_sha256": "legacy",
        "split_partition_sha256_v2": "strong",
        "sklearn_version": "1.9",
        "shared_settings_digest": "same",
        "growth_family": EXPECTED_FAMILIES[model_id],
        "seed": seed,
        "route": {"model_id": model_id},
        "arms": arms,
    }
    path = folder / "pair_receipt.json"
    path.write_text(json.dumps(raw))
    return path


def test_four_real_growth_method_receipts_validate(tmp_path):
    files = [_fake(tmp_path, model) for model in EXPECTED_FAMILIES]
    report = audit_family_receipts(files)
    assert report["distinct_native_growth_family_count"] == 4
    assert report["seed"] == 73


def test_missing_or_duplicate_method_rejected(tmp_path):
    files = [_fake(tmp_path, model) for model in EXPECTED_FAMILIES]
    with pytest.raises(ValueError, match="precisely four"):
        audit_family_receipts(files[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        audit_family_receipts(files[:3] + [files[0]])


def test_different_model_seed_rejected(tmp_path):
    files = [_fake(tmp_path, model) for model in EXPECTED_FAMILIES]
    p = files[0]
    data = json.loads(p.read_text())
    data["seed"] += 1
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="seed"):
        audit_family_receipts(files)


def test_corrupt_inference_model_fails_closed(tmp_path):
    files = [_fake(tmp_path, model) for model in EXPECTED_FAMILIES]
    (files[1].parent / "bayesian" / "estimator.model.pt").write_bytes(b"bad")
    with pytest.raises(ValueError, match="tampered"):
        audit_family_receipts(files)
