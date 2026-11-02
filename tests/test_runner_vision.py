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
    assert "vision_inner" not in models.adapters
    assert "vision_v03" not in models.adapters
    assert "vision_v04" not in models.adapters


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
    # In-process worker tests must not exec into a systemd-run user scope.
    # The production worker still keeps its original systemd behavior.
    monkeypatch.setenv("TABULAR_SKIP_SYSTEMD_SCOPE", "1")
    # In normal worker processes the scheduler explicitly admits a backend.
    monkeypatch.setenv("TABULAR_BACKEND", "cpu")
    monkeypatch.setenv("SIMPLE_ADP_DISABLE_GPU_ACCELERATORS", "1")
    monkeypatch.setenv("TRAINING_CONTROL_CPU_ONLY", "1")
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
    assert {run.model_adapter for run in plan.runs} == {"vision_adp", "vision_inner", "vision_v03", "vision_v04"}
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

    inner = next(
        run for run in plan.runs
        if run.model_adapter == "vision_inner"
        and run.route.model_id == "ADP-W-RR"
        and run.worker_settings.get("expansion_criterion") != "variance_stagnation"
    )
    inner_pair = next(
        run for run in plan.runs
        if run.worker_settings.get("counterpart_of") == inner.identity.run_id
    )
    assert inner_pair.model_adapter == "vision_inner"
    assert inner_pair.identity.seed == inner.identity.seed
    assert inner_pair.identity.split_sha256 == inner.identity.split_sha256
    assert inner_pair.worker_settings["expansion_criterion"] == "variance_stagnation"
    inner_result = run_spec(inner.command[-1])
    assert inner_result["status"] == "completed"
    assert inner_result["architecture_id"] == "V02:toy-frozen-residual-adapter"
    assert inner_result["growth_site"] == "adapter:block1"
    assert len(inner_result["selected_architecture"]) >= 1
    assert inner_result["test_metrics"]["accuracy"] >= 0
    assert inner_result["artifact_path"] is not None

    generic = next(
        run for run in plan.runs
        if run.model_adapter == "vision_inner"
        and run.route.model_id == "ADP-WD-RR-Dcur"
        and run.route.generic_profile == "corrected"
        and run.worker_settings.get("expansion_criterion") != "variance_stagnation"
    )
    generic_result = run_spec(generic.command[-1])
    assert generic_result["status"] == "completed"
    assert generic_result["architecture_id"] == "V02:toy-frozen-residual-adapter"
    assert generic_result["test_metrics"]["accuracy"] >= 0
    assert any(run.model_adapter == "vision_inner"
               and run.route.generic_profile == "faithful" for run in plan.runs)

    variance_result = run_spec(inner_pair.command[-1])
    assert variance_result["status"] == "completed"
    assert variance_result["expansion_criterion"] == "variance_stagnation"
    assert variance_result["selected_architecture"]





@pytest.mark.parametrize(
    ("family", "adapter_name", "expected_site"),
    [
        ("V03", "vision_v03", "ffn:block0"),
        ("V04", "vision_v04", "channel:stage0.block0"),
    ],
)
def test_native_ffn_runner_executes_and_pairs_variance(
    monkeypatch, tmp_path, family, adapter_name, expected_site,
):
    monkeypatch.setenv("SIMPLE_ADP_ENABLE_VISION", "1")
    monkeypatch.setenv("TABULAR_SKIP_SYSTEMD_SCOPE", "1")
    monkeypatch.setenv("TABULAR_BACKEND", "cpu")
    monkeypatch.setenv("SIMPLE_ADP_DISABLE_GPU_ACCELERATORS", "1")
    monkeypatch.setenv("TRAINING_CONTROL_CPU_ONLY", "1")
    datasets, models = load_registries()
    settings = RunnerSettings(
        tasks=("vision",), run_hardness_screen=False, repeat_count=1,
        missing_present_task_repeats=1, model_seeds=(0,),
        width_to_depth_width=2, width_to_depth_depth=1,
        max_candidates=1, max_epochs=1, batch_size=16,
        smoke_test=True, cpu_only=True,
    )
    plan = SuitePlan.build(
        settings=settings, results_dir=tmp_path / "results",
        run_root=tmp_path / "runs",
        dataset_providers=tuple(datasets.providers.values()),
        dataset_suites=(("vision", "toy"),),
        model_adapters=tuple(models.adapters.values()),
    )
    native = next(
        row for row in plan.runs
        if row.model_adapter == adapter_name
        and row.route.model_id == "ADP-W-RR"
        and row.worker_settings.get("expansion_criterion") != "variance_stagnation"
    )
    alternate = next(
        row for row in plan.runs
        if row.worker_settings.get("counterpart_of") == native.identity.run_id
    )
    assert alternate.model_adapter == adapter_name
    assert alternate.identity.seed == native.identity.seed
    assert alternate.identity.split_sha256 == native.identity.split_sha256
    assert alternate.worker_settings["expansion_criterion"] == "variance_stagnation"
    native_result = run_spec(native.command[-1])
    assert native_result["status"] == "completed"
    assert native_result["growth_site"] == expected_site
    assert native_result["architecture_id"].startswith(family + ":tiny-")
    assert len(native_result["selected_architecture"]) == 1
    assert native_result["test_metrics"]["accuracy"] >= 0
    assert native_result["artifact_path"] is not None
    assert len(native_result["frozen_sha256"]) == 64
    alternate_result = run_spec(alternate.command[-1])
    assert alternate_result["status"] == "completed"
    assert alternate_result["expansion_criterion"] == "variance_stagnation"
    generic = next(
        row for row in plan.runs
        if row.model_adapter == adapter_name
        and row.route.model_id == "ADP-WD-RR-Dcur"
        and row.route.generic_profile == "corrected"
        and row.worker_settings.get("expansion_criterion") != "variance_stagnation"
    )
    assert generic.route.route_label.startswith(family + ":")
    generic_result = run_spec(generic.command[-1])
    assert generic_result["status"] == "completed"
    assert generic_result["growth_site"] == expected_site
    assert any(
        row.model_adapter == adapter_name and row.route.generic_profile == "faithful"
        for row in plan.runs
    )
