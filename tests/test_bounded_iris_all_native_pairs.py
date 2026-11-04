"""All 13 source-adaptive methods require distinct verified paired receipts."""
from __future__ import annotations

import hashlib
import json

import pytest

from scripts.audit_bounded_iris_all_native import (
    EXPECTED_NATIVE_METHOD_FAMILIES, audit_native_methods,
)
from scripts.run_bounded_iris_bayesian_pair import (
    ALL_NATIVE_METHOD_FAMILIES, GROWTH_FAMILY_PROOFS,
)
from simple_adp.registry import MODEL_REGISTRY


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake(tmp_path, method):
    root = tmp_path / method
    root.mkdir()
    split = root / "iris_split.npz"
    split.write_bytes(b"fixed-iris-split")
    arms = {}
    for arm, trigger in (("original", "loss"), ("bayesian", "variance_stagnation")):
        folder = root / arm
        folder.mkdir()
        model = folder / "estimator.model.pt"
        checkpoint = folder / "checkpoint.pt"
        model.write_bytes((method + arm + "model").encode())
        checkpoint.write_bytes((method + arm + "checkpoint").encode())
        arms[arm] = {
            "growth_trigger": trigger,
            "model_artifact_sha256": _sha(model),
            "model_artifact_size_bytes": model.stat().st_size,
            "checkpoint_sha256": _sha(checkpoint),
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "selected_architecture": [3],
            "test_metrics": {
                "accuracy": 0.5, "balanced_accuracy": 0.5,
                "macro_f1": 0.5, "log_loss": 1.0,
            },
        }
    family = EXPECTED_NATIVE_METHOD_FAMILIES[method]
    d = {
        "schema": "bounded-iris-original-vs-shadow-bgd-v1",
        "evidence_grade": "BOUNDED_REAL_DATA_PASSED",
        "source_commit": "a" * 40,
        "scientific_settings_diff": {
            "expansion_criterion": ["loss", "variance_stagnation"],
        },
        "dataset_source": "sklearn.datasets.load_iris",
        "dataset_sha256": "data",
        "split_sizes": {"train": 90, "validation": 30, "test": 30},
        "split_archive_sha256": _sha(split),
        "split_legacy_sha256": "legacy",
        "split_partition_sha256_v2": "v2",
        "sklearn_version": "1.9",
        "shared_settings_digest": "settings",
        "seed": 73,
        "route": {
            "model_id": method, "suite_group": family,
            "generic_profile": (
                "corrected" if family in {"width_to_depth", "depth_to_width"} else None
            ),
        },
        "growth_family": family,
        "arms": arms,
    }
    path = root / "pair_receipt.json"
    path.write_text(json.dumps(d))
    return path


def test_all_13_canonical_adaptive_source_methods_have_distinct_routes():
    assert len(ALL_NATIVE_METHOD_FAMILIES) == 13
    assert ALL_NATIVE_METHOD_FAMILIES == EXPECTED_NATIVE_METHOD_FAMILIES
    assert set(ALL_NATIVE_METHOD_FAMILIES) == set(MODEL_REGISTRY) - {"MLP-Fixed"}
    assert all(ALL_NATIVE_METHOD_FAMILIES[x] == GROWTH_FAMILY_PROOFS[x]
               for x in GROWTH_FAMILY_PROOFS)


def test_exact_13_method_matrix_is_accepted(tmp_path):
    files = [_fake(tmp_path, x) for x in EXPECTED_NATIVE_METHOD_FAMILIES]
    report = audit_native_methods(files)
    assert report["method_count"] == 13
    assert report["execution_records"] == 26
    assert report["family_counts"] == {
        "depth_only": 2, "depth_to_width": 4,
        "width_only": 3, "width_to_depth": 4,
    }


def test_missing_and_duplicate_native_method_rejected(tmp_path):
    files = [_fake(tmp_path, x) for x in EXPECTED_NATIVE_METHOD_FAMILIES]
    with pytest.raises(ValueError, match="exactly 13"):
        audit_native_methods(files[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        audit_native_methods(files[:-1] + [files[0]])


def test_tampered_checkpoint_fails_even_if_model_inference_bundle_is_intact(tmp_path):
    files = [_fake(tmp_path, x) for x in EXPECTED_NATIVE_METHOD_FAMILIES]
    (files[0].parent / "bayesian" / "checkpoint.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint"):
        audit_native_methods(files)


def test_different_seed_rejected_across_models(tmp_path):
    files = [_fake(tmp_path, x) for x in EXPECTED_NATIVE_METHOD_FAMILIES]
    mutated = json.loads(files[-1].read_text())
    mutated["seed"] = 74
    files[-1].write_text(json.dumps(mutated))
    with pytest.raises(ValueError, match="seed"):
        audit_native_methods(files)


def test_wrong_generic_profile_or_growth_family_rejected(tmp_path):
    files = [_fake(tmp_path, x) for x in EXPECTED_NATIVE_METHOD_FAMILIES]
    item = json.loads(files[2].read_text())
    item["route"]["suite_group"] = "depth_only"
    files[2].write_text(json.dumps(item))
    with pytest.raises(ValueError, match="suite group"):
        audit_native_methods(files)
