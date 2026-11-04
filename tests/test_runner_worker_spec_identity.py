"""Canonical serialized IDs and actual loaded split must bind to source jobs."""
from __future__ import annotations

from copy import deepcopy

import pytest

from simple_adp.runner.datasets import DatasetTaskRef
from simple_adp.runner.plan import RunIdentity, SplitSpec, _sha_indices
from simple_adp.runner.route import ModelRoute
from simple_adp.runner.spec_validation import validate_loaded_split
from simple_adp.runner.worker import _rebuild_spec


def _spec(*, bayesian=False):
    route = ModelRoute(
        "ADP-W-All", label="ADP-W-All@depth-1-width-3",
        suite_group="width_only", init_depth=1, init_width=3,
    )
    partitions = ((0, 1, 2), (3,), (4,))
    digest = _sha_indices(*partitions)
    source = RunIdentity(
        provider="test-real", suite="iris", task_id="iris-150",
        split_repeat=0, fold=0, sample=0, suite_repeat=1,
        model_route=route.route_label, seed=73, split_sha256=digest,
        run_signature="f" * 64,
    )
    settings = {"seed": 73, "expansion_criterion": "loss"}
    identity = source
    if bayesian:
        identity = RunIdentity(
            provider=source.provider, suite=source.suite,
            task_id=source.task_id, split_repeat=source.split_repeat,
            fold=source.fold, sample=source.sample,
            suite_repeat=source.suite_repeat,
            model_route=source.model_route + "__bgd_variance_stagnation",
            seed=source.seed, split_sha256=source.split_sha256,
            run_signature=source.run_signature,
        )
        settings = {
            "seed": 73, "expansion_criterion": "variance_stagnation",
            "counterpart_of": source.run_id,
            "variance_posterior": "shadow_bgd_sigma_not_exact_bayes",
        }
    return {
        "identity": identity.as_dict(),
        "task_ref": {
            "provider": "test-real", "suite": "iris", "task_id": "iris-150",
            "dataset_id": "sklearn-iris-150", "metadata": {},
        },
        "route": route.as_dict(),
        "worker_settings": settings,
        "model_adapter": "simple_adp",
        "split_file": "split.npz",
        "result_dir": "result",
    }, partitions


@pytest.mark.parametrize("bayesian", [False, True])
def test_exact_original_and_shadow_specs_accept_canonical_ids(bayesian):
    raw, parts = _spec(bayesian=bayesian)
    identity, _, route = _rebuild_spec(raw)
    assert identity.run_id == raw["identity"]["run_id"]
    assert route.route_label == raw["route"]["route_label"]
    validate_loaded_split(
        SplitSpec(0, 0, 0, *parts, identity.split_sha256),
        identity, row_count=5,
    )


@pytest.mark.parametrize("mutation", [
    "run_id", "route_label", "model_route", "seed", "original_variance",
    "fake_posterior", "missing_counterpart", "fixed_counterpart",
])
def test_internal_job_spec_conflicts_fail_before_training(mutation):
    raw, _ = _spec(bayesian=mutation in {
        "fake_posterior", "missing_counterpart", "fixed_counterpart",
    })
    if mutation == "run_id":
        raw["identity"]["run_id"] = "forged"
    elif mutation == "route_label":
        raw["route"]["route_label"] = "ADP-W-RR"
    elif mutation == "model_route":
        raw["identity"]["model_route"] = "ADP-W-RR"
        # A recomputed identity label is not sufficient to authorize route drift.
        from simple_adp.runner.plan import RunIdentity
        raw["identity"]["run_id"] = RunIdentity(
            **{key: value for key, value in raw["identity"].items()
               if key != "run_id"}
        ).run_id
    elif mutation == "seed":
        raw["worker_settings"]["seed"] += 1
    elif mutation == "original_variance":
        raw["worker_settings"]["expansion_criterion"] = "variance_stagnation"
    elif mutation == "fake_posterior":
        raw["worker_settings"]["variance_posterior"] = "exact_bayes"
    elif mutation == "missing_counterpart":
        raw["worker_settings"].pop("counterpart_of")
    else:
        raw["route"]["model_id"] = "MLP-Fixed"
        raw["route"]["route_label"] = "MLP-Fixed"
        raw["route"]["label"] = "MLP-Fixed"
    with pytest.raises(ValueError, match="run_id|route|seed|counterpart|posterior|fixed|criterion"):
        _rebuild_spec(raw)


def test_loaded_split_valid_internal_archive_must_match_worker_identity():
    raw, parts = _spec()
    identity, _, _ = _rebuild_spec(raw)
    shifted = ((0, 2), (1, 3), (4,))
    archive = SplitSpec(0, 0, 0, *shifted, _sha_indices(*shifted))
    with pytest.raises(ValueError, match="checksum differs"):
        validate_loaded_split(archive, identity, row_count=5)


@pytest.mark.parametrize("invalid", [(-1, 1, 2), (0, 1, 99)])
def test_negative_and_out_of_source_split_rows_fail_closed(invalid):
    raw, _ = _spec()
    identity, _, _ = _rebuild_spec(raw)
    partitions = (tuple(invalid), (3,), (4,))
    archive = SplitSpec(0, 0, 0, *partitions, identity.split_sha256)
    with pytest.raises(ValueError, match="outside source"):
        validate_loaded_split(archive, identity, row_count=5)
