"""Signed native visual worker integration of actual minibatch forward MACs."""
from __future__ import annotations

import json

import pytest

from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.plan import SuitePlan
from simple_adp.runner.plugins import load_registries
from simple_adp.runner.worker import run_spec
from simple_adp.visual_budget_audit import audit_visual_worker_pairs


def _plan(monkeypatch, tmp_path, *, track: bool):
    monkeypatch.setenv("SIMPLE_ADP_ENABLE_VISION", "1")
    monkeypatch.setenv("SIMPLE_ADP_ENABLE_V03_SOURCE", "0")
    monkeypatch.setenv("SIMPLE_ADP_ENABLE_V04_SOURCE", "0")
    monkeypatch.setenv("SIMPLE_ADP_DISABLE_GPU_ACCELERATORS", "1")
    monkeypatch.setenv("TRAINING_CONTROL_CPU_ONLY", "1")
    monkeypatch.setenv("TABULAR_SKIP_SYSTEMD_SCOPE", "1")
    monkeypatch.setenv("TABULAR_BACKEND", "cpu")
    providers, adapters = load_registries()
    settings = RunnerSettings(
        tasks=("vision",), run_hardness_screen=False,
        repeat_count=1, missing_present_task_repeats=1,
        model_seeds=(0,), width_to_depth_width=2,
        width_to_depth_depth=1, max_width=512, max_depth=5,
        max_neurons=128, max_candidates=1, max_epochs=None,
        max_epochs_vanilla=2, max_epochs_adp=1,
        batch_size=8, smoke_test=False, cpu_only=True,
        visual_training_forward_macs=track,
    )
    plan = SuitePlan.build(
        settings=settings, results_dir=tmp_path / "results",
        run_root=tmp_path / "runs",
        dataset_providers=tuple(providers.providers.values()),
        dataset_suites=(("vision", "toy"),),
        model_adapters=tuple(adapters.adapters.values()),
    )
    jobs = {
        (job.model_adapter, job.route.model_id): job
        for job in plan.runs
        if job.model_adapter in {"vision_inner", "vision_adp"}
        and job.route.model_id in {"MLP-Fixed", "ADP-W-RR"}
        and job.worker_settings.get("expansion_criterion", "loss") == "loss"
        and job.worker_settings.get("counterpart_of") is None
    }
    assert len(jobs) == 4
    return settings, jobs


def test_opt_in_signed_plan_observes_native_training_not_legacy_v01(monkeypatch, tmp_path):
    settings, jobs = _plan(monkeypatch, tmp_path, track=True)
    assert settings.as_dict()["visual_training_forward_macs"] is True
    fixed_job = jobs[("vision_inner", "MLP-Fixed")]
    growth_job = jobs[("vision_inner", "ADP-W-RR")]
    for job in (fixed_job, growth_job):
        assert job.worker_settings["visual_training_forward_macs"] is True
        signed = json.loads(open(job.command[-1], encoding="utf-8").read())
        assert signed["worker_settings"]["visual_training_forward_macs"] is True

    fixed, grown = run_spec(fixed_job.command[-1]), run_spec(growth_job.command[-1])
    for result in (fixed, grown):
        assert result["status"] == "completed"
        actual = result["training_forward_compute"]
        assert actual["complete"] is True
        assert actual["full_training_flops"] is None
        assert actual["partial_training_forward_macs"] > 0
        assert actual["recorded_training_forward_batches"] == result["training_budget"]["optimizer_steps"]
        assert len(result["trained_forward_mac_trace"]) == result["training_budget"]["trained_stages"]
        assert result["trained_forward_mac_trace_complete"] is True
        assert all(t["complete"] for t in result["trained_forward_mac_trace"])

    assert fixed["training_budget"]["optimizer_steps"] == grown["training_budget"]["optimizer_steps"]
    # Source architecture and per-training-step work differ after width
    # expansion, even when the optimizer-step budget is identical.
    assert fixed["training_forward_compute"]["partial_training_forward_macs"] != (
        grown["training_forward_compute"]["partial_training_forward_macs"]
    )
    rows = [
        {"adapter": "vision_inner", "method": method, **result}
        for method, result in (("MLP-Fixed", fixed), ("ADP-W-RR", grown))
    ]
    audit = audit_visual_worker_pairs(rows, expected_adapters=["vision_inner"])
    assert audit["same_optimizer_step_pairs"] == 1
    assert audit["full_training_flops_matched"] is None

    # V01 is a distinct original legacy head controller. It remains
    # runnable, but must never fabricate a native work counter.
    legacy = run_spec(jobs[("vision_adp", "MLP-Fixed")].command[-1])
    assert legacy["status"] == "completed"
    assert "training_forward_compute" not in legacy


def test_opt_out_preserves_original_worker_spec_and_old_plan_settings(monkeypatch, tmp_path):
    settings, jobs = _plan(monkeypatch, tmp_path, track=False)
    assert "visual_training_forward_macs" not in settings.as_dict()
    job = jobs[("vision_inner", "MLP-Fixed")]
    assert "visual_training_forward_macs" not in job.worker_settings
    output = run_spec(job.command[-1])
    assert output["status"] == "completed"
    assert "training_forward_compute" not in output


@pytest.mark.parametrize("invalid", [1, None, "true", 0])
def test_runner_tracking_flag_must_be_boolean(invalid):
    with pytest.raises(ValueError, match="boolean"):
        RunnerSettings(visual_training_forward_macs=invalid)
