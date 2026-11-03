"""Genuine independent controlled vision seeds are planned by native SuitePlan."""
from __future__ import annotations

from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.plan import SuitePlan
from simple_adp.runner.plugins import load_registries


CONTROL_ADAPTERS = (
    "vision_adp", "vision_inner", "vision_v03", "vision_v04", "vision_v05",
    "vision_v06a", "vision_v06b", "vision_v07_token", "vision_v07_channel",
    "vision_v07_both", "vision_v08_early", "vision_v08_middle",
    "vision_v08_late", "vision_v08_schedule", "vision_v09_bank0",
    "vision_v09_bank1", "vision_v09_bank2", "vision_v10", "vision_v11",
)


def test_native_visual_plan_has_true_two_seed_fixed_rr_pairs(monkeypatch, tmp_path):
    monkeypatch.setenv("SIMPLE_ADP_ENABLE_VISION", "1")
    monkeypatch.setenv("SIMPLE_ADP_ENABLE_V03_SOURCE", "0")
    monkeypatch.setenv("SIMPLE_ADP_ENABLE_V04_SOURCE", "0")
    data_registry, model_registry = load_registries()
    options = RunnerSettings(
        tasks=("vision",), run_hardness_screen=False,
        repeat_count=1, missing_present_task_repeats=1,
        model_seeds=(0, 1), max_width=512,
        max_depth=5, max_neurons=128,
        width_to_depth_width=2, width_to_depth_depth=1,
        max_candidates=1, max_epochs=1, batch_size=16,
        smoke_test=True, cpu_only=True,
    )

    def build():
        return SuitePlan.build(
            settings=options,
            results_dir=tmp_path / "results",
            run_root=tmp_path / "runs",
            dataset_providers=tuple(data_registry.providers.values()),
            dataset_suites=(("vision", "toy"),),
            model_adapters=tuple(model_registry.adapters.values()),
        )

    plan = build()
    selected = {
        (job.model_adapter, job.route.model_id, job.identity.seed): job
        for job in plan.runs
        if job.model_adapter in CONTROL_ADAPTERS
        and job.route.model_id in {"MLP-Fixed", "ADP-W-RR"}
        and job.worker_settings.get("expansion_criterion", "loss") == "loss"
        and job.worker_settings.get("counterpart_of") is None
    }
    assert len(selected) == 76
    seeds = {entry[2] for entry in selected}
    assert len(seeds) == 2
    assert all(
        (adapter, method, seed) in selected
        for adapter in CONTROL_ADAPTERS for method in ("MLP-Fixed", "ADP-W-RR")
        for seed in seeds
    )
    assert len({job.identity.run_id for job in selected.values()}) == 76
    assert all(job.task_ref.suite == "toy" for job in selected.values())
    for adapter in CONTROL_ADAPTERS:
        for seed in seeds:
            fixed = selected[(adapter, "MLP-Fixed", seed)]
            grown = selected[(adapter, "ADP-W-RR", seed)]
            assert fixed.identity.seed == grown.identity.seed
            assert fixed.identity.split_sha256 == grown.identity.split_sha256
            assert fixed.task_ref.dataset_id == grown.task_ref.dataset_id
            assert fixed.identity.run_id != grown.identity.run_id

    again = build()
    second = {
        (job.model_adapter, job.route.model_id, job.identity.seed): job.identity.run_id
        for job in again.runs
        if job.model_adapter in CONTROL_ADAPTERS
        and job.route.model_id in {"MLP-Fixed", "ADP-W-RR"}
        and job.worker_settings.get("expansion_criterion", "loss") == "loss"
        and job.worker_settings.get("counterpart_of") is None
    }
    assert {key: job.identity.run_id for key, job in selected.items()} == second
