"""Bounded official Fashion-MNIST CPU worker matrix for every visual control.

This is a *real-image integration smoke*, not a multi-seed science benchmark.
It selects only fixed and ADP-W-RR routes from the authoritative SuitePlan,
then executes their normal signed worker specifications. All remaining
registered visual routes stay planned but are not executed by this script.
No model/algorithm/optimizer is reimplemented here.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.plan import SuitePlan
from simple_adp.runner.plugins import load_registries
from simple_adp.runner.worker import run_spec

CONTROL_ADAPTERS = (
    "vision_adp", "vision_inner", "vision_v03", "vision_v04", "vision_v05",
    "vision_v06a", "vision_v06b", "vision_v07_token", "vision_v07_channel",
    "vision_v07_both", "vision_v08_early", "vision_v08_middle",
    "vision_v08_late", "vision_v08_schedule", "vision_v09_bank0",
    "vision_v09_bank1", "vision_v09_bank2", "vision_v10", "vision_v11",
)


def _save(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--train-limit", type=int, default=40)
    parser.add_argument("--valid-limit", type=int, default=20)
    parser.add_argument("--test-limit", type=int, default=20)
    args = parser.parse_args(argv)
    if min(args.train_limit, args.valid_limit, args.test_limit) < 10:
        parser.error("official Fashion-MNIST 10-class balanced limits must be at least ten")
    root = Path(args.workdir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    data_root = root / "dataset"
    torch.set_num_threads(2)

    # Official torchvision source; this explicit download step is isolated
    # from the default runner and never modifies a historical provider.
    from torchvision.datasets import FashionMNIST
    FashionMNIST(root=str(data_root), train=True, download=True)
    FashionMNIST(root=str(data_root), train=False, download=True)
    os.environ.update({
        "SIMPLE_ADP_ENABLE_VISION": "1",
        "SIMPLE_ADP_ENABLE_V03_SOURCE": "0",
        "SIMPLE_ADP_ENABLE_V04_SOURCE": "0",
        "SIMPLE_ADP_VISION_ROOT": str(data_root),
        "SIMPLE_ADP_VISION_TRAIN_LIMIT": str(args.train_limit),
        "SIMPLE_ADP_VISION_VALID_LIMIT": str(args.valid_limit),
        "SIMPLE_ADP_VISION_TEST_LIMIT": str(args.test_limit),
        "SIMPLE_ADP_DISABLE_GPU_ACCELERATORS": "1",
        "TRAINING_CONTROL_CPU_ONLY": "1",
        "TABULAR_BACKEND": "cpu",
        "TABULAR_SKIP_SYSTEMD_SCOPE": "1",
    })
    datasets, models = load_registries()
    settings = RunnerSettings(
        tasks=("vision",), run_hardness_screen=False, repeat_count=1,
        missing_present_task_repeats=1, model_seeds=(0,),
        width_to_depth_width=2, width_to_depth_depth=1,
        # Keep the runner's global baseline width grids valid. The selected
        # one-step RR jobs are independently bounded by init width, candidate
        # count, epochs and max_neurons.
        max_width=512, max_depth=5, max_neurons=128,
        max_candidates=1, max_epochs=1, batch_size=16,
        smoke_test=True, cpu_only=True,
    )
    plan = SuitePlan.build(
        settings=settings,
        results_dir=root / "results", run_root=root / "runs",
        dataset_providers=tuple(datasets.providers.values()),
        dataset_suites=(("vision", "fashion_mnist"),),
        model_adapters=tuple(models.adapters.values()),
    )
    selected = {}
    for job in plan.runs:
        if job.model_adapter not in CONTROL_ADAPTERS:
            continue
        if job.route.model_id not in {"MLP-Fixed", "ADP-W-RR"}:
            continue
        if job.worker_settings.get("expansion_criterion", "loss") != "loss":
            continue
        if job.worker_settings.get("counterpart_of") is not None:
            continue
        key = (job.model_adapter, job.route.model_id)
        if key in selected:
            raise ValueError(f"duplicate official matrix job: {key}")
        selected[key] = job

    expected = {
        (name, method)
        for name in CONTROL_ADAPTERS
        for method in ("MLP-Fixed", "ADP-W-RR")
    }
    if set(selected) != expected:
        raise ValueError(f"official image matrix has missing/extra routes: {sorted(expected ^ set(selected))}")

    summary_path = root / "official_fashion_vision_matrix.json"
    summary: dict[str, object] = {
        "format": "simple-adp-official-fashion-visual-controls-v1",
        "source": "torchvision.datasets.FashionMNIST official train/test",
        "dataset_download": "explicit",
        "train_limit": args.train_limit,
        "validation_limit": args.valid_limit,
        "test_limit": args.test_limit,
        "fixed_vs_growth_methods": ["MLP-Fixed", "ADP-W-RR"],
        "planned_full_vision_jobs": len(plan.runs),
        "selected_worker_jobs": len(selected),
        "scientific_scope": "small one-seed, one-epoch official-image smoke; not a benchmark",
        "rows": [],
    }
    _save(summary_path, summary)
    failures = 0
    for adapter_name in CONTROL_ADAPTERS:
        for method in ("MLP-Fixed", "ADP-W-RR"):
            job = selected[(adapter_name, method)]
            started = time.perf_counter()
            row: dict[str, object] = {
                "adapter": adapter_name, "method": method,
                "job_id": job.identity.run_id,
                "split_sha256": job.identity.split_sha256,
                "worker_spec": str(job.command[-1]),
            }
            try:
                value = run_spec(job.command[-1])
                if value["status"] != "completed":
                    raise AssertionError("worker did not report completion")
                if value.get("task_family") != "vision":
                    raise AssertionError("worker did not use vision task contract")
                for part in ("validation_metrics", "test_metrics"):
                    metrics = value[part]
                    if not (0 <= metrics.get("top_label_ece_10", -1) <= 1):
                        raise AssertionError(f"{part}: missing valid top-label ECE")
                    if not (0 <= metrics.get("multiclass_brier", -1) <= 2):
                        raise AssertionError(f"{part}: missing valid multiclass Brier")
                if "parameter_accounting" not in value:
                    raise AssertionError("composite parameter accounting missing")
                counts = value["parameter_accounting"]
                compute = value.get("inference_compute")
                if not compute or compute.get("observed_linear_conv_macs_per_image", 0) <= 0:
                    raise AssertionError("observed whole-composite dense/conv inference probe missing")
                if compute.get("whole_model_flops") is not None:
                    raise AssertionError("partial operator MAC counts must not be reported as full FLOPs")
                if counts["total_parameters"] != counts["frozen_parameters"] + counts["trainable_parameters"]:
                    raise AssertionError("composite parameter counts do not close")
                if method == "ADP-W-RR" and value["history_entries"] < 2:
                    raise AssertionError("no actual structural candidate stage was executed")
                budget = value.get("training_budget")
                if adapter_name != "vision_adp":
                    if not budget or not budget.get("complete") or budget.get("optimizer_steps", 0) < 1:
                        raise AssertionError("native inner-MLP optimizer/search budget is missing")
                row.update({
                    "status": "completed",
                    "architecture_id": value["architecture_id"],
                    "growth_site": value["growth_site"],
                    "selected_architecture": value["selected_architecture"],
                    "candidate_stages_executed": value["history_entries"],
                    "training_budget": value.get("training_budget"),
                    "validation_metrics": value["validation_metrics"],
                    "test_metrics": value["test_metrics"],
                    "parameter_accounting": counts,
                    "inference_compute": compute,
                })
            except Exception as exc:
                failures += 1
                row.update({"status": "failed", "error": repr(exc)})
            row["elapsed_seconds"] = round(time.perf_counter() - started, 6)
            summary["rows"].append(row)
            _save(summary_path, summary)
            print(json.dumps({k: row.get(k) for k in ("adapter", "method", "status", "error")}, sort_keys=True), flush=True)
    summary["completed_jobs"] = len(selected) - failures
    summary["failed_jobs"] = failures
    summary["status"] = "completed" if failures == 0 else "incomplete"
    _save(summary_path, summary)
    if failures:
        raise RuntimeError(f"{failures} official-image workers failed; consult {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
