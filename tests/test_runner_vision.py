"""End-to-end opt-in coordinated V01 vision suite tests."""
from __future__ import annotations

import json

import pytest

from simple_adp.config import SearchSettings
from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.plan import SuitePlan
from simple_adp.runner.plugins import load_registries
from simple_adp.runner.worker import run_spec


def test_vision_is_opt_in_and_does_not_change_legacy_registries(monkeypatch):
    monkeypatch.delenv("SIMPLE_ADP_ENABLE_VISION", raising=False)
    datasets, models = load_registries()
    assert "vision" not in datasets.providers
    assert "vision_adp" not in models.adapters


def test_vision_provider_manifest_and_split_isolation(monkeypatch):
    monkeypatch.setenv("SIMPLE_ADP_ENABLE_VISION", "1")
    datasets, models = load_registries()
    provider = datasets.get("vision")
    data = provider.load_task(provider.list_tasks("toy")[0])
    assert data.X.ndim == 4 and data.X.shape[1] == 1
    assert data.task_family == "vision"
    split = provider.list_splits(provider.list_tasks("toy")[0], data, RunnerSettings())[0]
    a, b, c = (set(split.train_indices), set(split.validation_indices),
               set(split.test_indices))
    assert a.isdisjoint(b) and a.isdisjoint(c) and b.isdisjoint(c)
    assert len(a) + len(b) + len(c) == len(data.X)
    assert set(data.context["official_test_group"][list(split.test_indices)]) == {1}
    assert set(data.context["official_test_group"][list(split.validation_indices)]) == {0}


def test_coordinated_vision_job_is_routed_and_executed(monkeypatch, tmp_path):
    monkeypatch.setenv("SIMPLE_ADP_ENABLE_VISION", "1")
    datasets, models = load_registries()
    settings = RunnerSettings(
        tasks=("vision",), run_hardness_screen=False, repeat_count=1,
        missing_present_task_repeats=1, model_seeds=(0,),
        width_to_depth_width=2, width_to_depth_depth=1,
        max_candidates=1, max_epochs=1, batch_size=16,
        smoke_test=True, cpu_only=True,
    )
    plan = SuitePlan.build(
        settings=settings,
        results_dir=tmp_path / "results", run_root=tmp_path / "runs",
        dataset_providers=tuple(datasets.providers.values()),
        dataset_suites=(("vision", "toy"),),
        model_adapters=tuple(models.adapters.values()),
    )
    assert plan.runs
    assert all(run.model_adapter == "vision_adp" for run in plan.runs)
    assert all(run.task_ref.provider == "vision" for run in plan.runs)
    assert any(run.route.model_id == "MLP-Fixed" for run in plan.runs)
    assert not any(
        run.route.model_id == "MLP-Fixed"
        and run.worker_settings.get("expansion_criterion") == "variance_stagnation"
        for run in plan.runs
    )
    target = next(
        run for run in plan.runs
        if run.route.model_id == "ADP-W-RR"
        and run.worker_settings.get("expansion_criterion") != "variance_stagnation"
    )
    counterpart = next(
        run for run in plan.runs
        if run.worker_settings.get("counterpart_of") == target.identity.run_id
    )
    assert target.identity.seed == counterpart.identity.seed
    assert target.identity.split_sha256 == counterpart.identity.split_sha256
    assert counterpart.worker_settings["expansion_criterion"] == "variance_stagnation"
    spec_path = target.command[-1]
    output = run_spec(spec_path)
    assert output["status"] == "completed"
    assert output["task_family"] == "vision"
    assert output["test_metrics"]["accuracy"] >= 0
    assert output["model_id"] == "ADP-W-RR"
    assert (tmp_path / "results").exists()
