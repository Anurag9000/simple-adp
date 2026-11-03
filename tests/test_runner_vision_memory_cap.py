"""The coordinated runner signs and enforces opt-in native model byte caps."""
from __future__ import annotations

import json

import pytest

from simple_adp.config import SearchSettings
from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.plan import SuitePlan
from simple_adp.runner.plugins import load_registries
from simple_adp.runner.vision import VisionDatasetProvider, VisionModelAdapter
from simple_adp.runner.worker import run_spec


@pytest.mark.parametrize("invalid", [True, False, -1, 0, 1.5, "100"])
def test_runner_visual_payload_cap_rejects_invalid_values(invalid):
    with pytest.raises(ValueError, match="positive integer"):
        RunnerSettings(visual_model_tensor_payload_cap_bytes=invalid)


def test_disabled_cap_is_omitted_from_legacy_run_signature_settings():
    default = RunnerSettings()
    assert default.visual_model_tensor_payload_cap_bytes is None
    assert "visual_model_tensor_payload_cap_bytes" not in default.as_dict()
    guarded = RunnerSettings(visual_model_tensor_payload_cap_bytes=2048)
    assert guarded.as_dict()["visual_model_tensor_payload_cap_bytes"] == 2048


def _prepare_native_plan(monkeypatch, tmp_path, *, cap):
    monkeypatch.setenv("SIMPLE_ADP_ENABLE_VISION", "1")
    monkeypatch.setenv("TABULAR_SKIP_SYSTEMD_SCOPE", "1")
    monkeypatch.setenv("TABULAR_BACKEND", "cpu")
    monkeypatch.setenv("SIMPLE_ADP_DISABLE_GPU_ACCELERATORS", "1")
    monkeypatch.setenv("TRAINING_CONTROL_CPU_ONLY", "1")
    datasets, models = load_registries()
    options = RunnerSettings(
        tasks=("vision",), run_hardness_screen=False, repeat_count=1,
        missing_present_task_repeats=1, model_seeds=(0,),
        width_to_depth_width=2, width_to_depth_depth=1,
        max_candidates=1, max_epochs=1, batch_size=16,
        smoke_test=True, cpu_only=True,
        visual_model_tensor_payload_cap_bytes=cap,
    )
    plan = SuitePlan.build(
        settings=options, results_dir=tmp_path / "results",
        run_root=tmp_path / "runs",
        dataset_providers=tuple(datasets.providers.values()),
        dataset_suites=(("vision", "toy"),),
        model_adapters=tuple(models.adapters.values()),
    )
    inner = next(
        job for job in plan.runs
        if job.model_adapter == "vision_inner"
        and job.route.model_id == "MLP-Fixed"
    )
    return options, plan, inner


def test_opt_in_cap_in_signed_worker_spec_allows_valid_native_cpu_job(monkeypatch, tmp_path):
    cap = 4_000_000  # Much higher than this tiny native model payload.
    options, plan, inner = _prepare_native_plan(monkeypatch, tmp_path, cap=cap)
    assert options.as_dict()["visual_model_tensor_payload_cap_bytes"] == cap
    assert inner.worker_settings["visual_model_tensor_payload_cap_bytes"] == cap
    saved_spec = json.loads(open(inner.command[-1], encoding="utf-8").read())
    assert saved_spec["worker_settings"]["visual_model_tensor_payload_cap_bytes"] == cap
    result = run_spec(inner.command[-1])
    assert result["status"] == "completed"
    assert result["model_tensor_payload_cap_bytes"] == cap
    assert result["memory_admission_rejections"] == []
    assert result["trained_model_memory_trace_complete"] is True
    assert len(result["trained_model_memory_trace"]) == result["training_budget"]["trained_stages"]
    mem = result["model_memory_accounting"]
    assert 0 < mem["total_model_tensor_payload_bytes"] < cap
    assert mem["total_model_tensor_payload_bytes"] == (
        mem["parameter_payload_bytes"] + mem["buffer_payload_bytes"]
    )
    assert mem["parameter_payload_bytes"] == (
        mem["trainable_parameter_bytes"] + mem["frozen_parameter_bytes"]
    )
    assert mem["peak_training_memory_bytes"] is None
    assert result["fit_process_resources"]["exact_fit_window_peak_rss_bytes"] is None
    assert result["fit_process_resources"]["device"] == "cpu"


def test_opt_in_cap_fails_closed_and_records_real_worker_failure(monkeypatch, tmp_path):
    cap = 1  # Smaller than the real frozen CNN even before any growth.
    _, _, inner = _prepare_native_plan(monkeypatch, tmp_path, cap=cap)
    with pytest.raises(ValueError, match="initial visual model logical tensor payload"):
        run_spec(inner.command[-1])
    result_path = tmp_path / "results" / inner.identity.run_id / "result.json"
    record = json.loads(result_path.read_text())
    assert record["status"] == "failed"
    assert "max_model_tensor_payload_bytes" in record["error"]


def test_legacy_v01_head_route_refuses_unimplemented_memory_admission():
    data_provider = VisionDatasetProvider()
    ref = data_provider.list_tasks("toy")[0]
    data = data_provider.load_task(ref)
    settings = RunnerSettings(tasks=("vision",))
    split = data_provider.list_splits(ref, data, settings)[0]
    route = next(
        item for item in VisionModelAdapter().enumerate_routes(settings=settings)
        if item.model_id == "MLP-Fixed"
    )
    with pytest.raises(ValueError, match="V01 original tabular-head"):
        VisionModelAdapter().fit_predict(
            route, data=data, split=split, seed=0, settings=SearchSettings(),
            max_model_tensor_payload_bytes=256,
        )
