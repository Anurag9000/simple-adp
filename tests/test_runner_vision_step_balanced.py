"""Native fixed/RR work counts can be balanced without changing ADP policies."""
from __future__ import annotations

from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.plan import SuitePlan
from simple_adp.runner.plugins import load_registries
from simple_adp.runner.worker import run_spec
from simple_adp.visual_budget_audit import audit_visual_worker_pairs


def test_native_fixed_two_epochs_vs_adp_two_one_epoch_stages(monkeypatch, tmp_path):
    monkeypatch.setenv("SIMPLE_ADP_ENABLE_VISION", "1")
    monkeypatch.setenv("TABULAR_SKIP_SYSTEMD_SCOPE", "1")
    monkeypatch.setenv("TABULAR_BACKEND", "cpu")
    monkeypatch.setenv("SIMPLE_ADP_DISABLE_GPU_ACCELERATORS", "1")
    monkeypatch.setenv("TRAINING_CONTROL_CPU_ONLY", "1")
    registry, models = load_registries()
    options = RunnerSettings(
        tasks=("vision",), run_hardness_screen=False,
        repeat_count=1, missing_present_task_repeats=1,
        model_seeds=(0,), width_to_depth_width=2, width_to_depth_depth=1,
        max_width=512, max_depth=5, max_neurons=128,
        max_candidates=1, max_epochs=None, max_epochs_vanilla=2,
        max_epochs_adp=1, batch_size=8, smoke_test=False, cpu_only=True,
    )
    plan = SuitePlan.build(
        settings=options, results_dir=tmp_path / "results",
        run_root=tmp_path / "runs",
        dataset_providers=tuple(registry.providers.values()),
        dataset_suites=(("vision", "toy"),),
        model_adapters=tuple(models.adapters.values()),
    )
    jobs = {
        job.route.model_id: job
        for job in plan.runs
        if job.model_adapter == "vision_inner"
        and job.route.model_id in {"MLP-Fixed", "ADP-W-RR"}
        and job.worker_settings.get("expansion_criterion", "loss") == "loss"
        and job.worker_settings.get("counterpart_of") is None
    }
    assert set(jobs) == {"MLP-Fixed", "ADP-W-RR"}
    assert all(job.worker_settings["max_epochs"] is None for job in jobs.values())
    assert all(job.worker_settings["max_epochs_vanilla"] == 2 for job in jobs.values())
    assert all(job.worker_settings["max_epochs_adp"] == 1 for job in jobs.values())
    assert all(not job.worker_settings["smoke_test"] for job in jobs.values())

    fixed = run_spec(jobs["MLP-Fixed"].command[-1])
    grown = run_spec(jobs["ADP-W-RR"].command[-1])
    assert fixed["status"] == grown["status"] == "completed"
    assert fixed["seed"] == grown["seed"]
    assert fixed["dataset_id"] == grown["dataset_id"]
    assert fixed["split_sha256"] == grown["split_sha256"]
    assert fixed["training_budget"]["trained_epochs"] == 2
    assert grown["training_budget"]["trained_epochs"] == 2
    assert fixed["training_budget"]["trained_stages"] == 1
    assert grown["training_budget"]["trained_stages"] == 2
    assert fixed["training_budget"]["optimizer_steps"] > 0
    assert fixed["training_budget"]["optimizer_steps"] == (
        grown["training_budget"]["optimizer_steps"]
    )
    records = [
        {"adapter": "vision_inner", "method": method, **record}
        for method, record in (("MLP-Fixed", fixed), ("ADP-W-RR", grown))
    ]
    audit = audit_visual_worker_pairs(records, expected_adapters=["vision_inner"])
    assert audit["same_optimizer_step_pairs"] == 1
    assert audit["mismatched_optimizer_step_pairs"] == 0
    assert audit["paired_budget_evidence"][0]["same_optimizer_step_count"] is True
    assert audit["full_training_flops_matched"] is None
    assert audit["scientific_performance_conclusion"] is None
