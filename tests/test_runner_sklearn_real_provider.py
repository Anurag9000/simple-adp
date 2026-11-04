"""Real offline scikit-learn source provider and true planner/worker identities."""
from __future__ import annotations

import pytest

sklearn = pytest.importorskip("sklearn")

from simple_adp.runner.sklearn_real import SklearnRealDatasetProvider
from simple_adp.runner.plugins import load_registries
from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.plan import SuitePlan
from simple_adp.runner.pair_audit import audit_manifest
from scripts.run_bounded_sklearn_real_worker_pairs import _OnlyWidthAll


def test_offline_real_data_provider_is_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("SIMPLE_ADP_ENABLE_SKLEARN_REAL", raising=False)
    datasets, _ = load_registries()
    assert "sklearn_real" not in datasets.providers
    monkeypatch.setenv("SIMPLE_ADP_ENABLE_SKLEARN_REAL", "1")
    datasets, _ = load_registries()
    assert "sklearn_real" in datasets.providers


@pytest.mark.parametrize("suite,rows,kind", [
    ("iris", 150, "classification"), ("diabetes", 442, "regression"),
])
def test_real_data_identity_split_reproducibility(suite, rows, kind):
    provider = SklearnRealDatasetProvider()
    ref, = provider.list_tasks(suite)
    data = provider.load_task(ref)
    assert len(data.X) == len(data.y) == rows
    assert data.task_kind == kind
    assert data.content_sha256 == provider.load_task(ref).content_sha256
    split, = provider.list_splits(ref, data, None)
    assert split == provider.list_splits(ref, data, None)[0]
    assert len(set(split.train_indices) | set(split.validation_indices)
               | set(split.test_indices)) == rows


def test_real_plan_generates_two_true_bayesian_pairs_with_equal_sources(tmp_path):
    provider = SklearnRealDatasetProvider()
    settings = RunnerSettings(
        cpu_only=True, max_epochs=1,
        repeat_count=1, missing_present_task_repeats=1,
        model_seeds=(73,), generic_profiles=("corrected",),
        tasks=("width_only",), include_vanilla=False,
        include_width_to_depth=False, include_depth_to_width=False,
        include_depth_only=False, run_hardness_screen=False,
        width_only_depths=(1,), width_only_width=3,
        width_to_depth_depth=1, width_to_depth_width=3,
        width_to_depth_widths=(3,), depth_only_widths=(3,),
        vanilla_depths=(1,), vanilla_widths=(3,),
        screen_depths=(1,), screen_widths=(3,),
        deeper_screen_widths=(3,),
        max_width=6, max_depth=2, max_neurons=30,
        post_launch_sample_delay_seconds=0,
    )
    plan = SuitePlan.build(
        settings=settings,
        results_dir=tmp_path / "results",
        run_root=tmp_path / "runtime",
        dataset_providers=(provider,),
        dataset_suites=(("sklearn_real", "iris"), ("sklearn_real", "diabetes")),
        model_adapters=(_OnlyWidthAll(),),
    )
    audit = audit_manifest(plan.manifest, source_commit="f" * 40)
    assert audit["summary"]["matched_pairs_selected"] == 2
    assert len(plan.runs) == 4
    assert all(run.route.model_id == "ADP-W-All" for run in plan.runs)
    assert {run.task_ref.suite for run in plan.runs} == {"iris", "diabetes"}
    assert set(x["model_adapter"] for x in plan.manifest["jobs"]) == {"simple_adp"}
    assert all(pair["same_science_digest"] for pair in audit["pairs"])
