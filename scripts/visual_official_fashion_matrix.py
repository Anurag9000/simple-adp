"""Bounded official Fashion-MNIST/CIFAR CPU worker matrix for visual controls.

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
from simple_adp.visual_budget_audit import (
    audit_visual_worker_pairs, audit_visual_multiseed_worker_pairs,
)

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
    parser.add_argument("--dataset", choices=("fashion_mnist", "cifar10", "cifar100"),
                        default="fashion_mnist")
    parser.add_argument("--train-limit", type=int, default=40)
    parser.add_argument("--valid-limit", type=int, default=20)
    parser.add_argument("--test-limit", type=int, default=20)
    parser.add_argument("--model-seeds", type=int, default=1,
                        help="number of independently derived native runner seeds")
    parser.add_argument("--step-balanced", action="store_true",
                        help="request 2 fixed epochs vs two 1-epoch ADP stages; audit actual steps")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="used in step-balanced mode; standard smoke uses full train batch")
    args = parser.parse_args(argv)
    if args.model_seeds < 1:
        parser.error("--model-seeds must be a positive integer")
    if args.batch_size < 1:
        parser.error("--batch-size must be a positive integer")
    required_classes = 100 if args.dataset == "cifar100" else 10
    if min(args.train_limit, args.valid_limit, args.test_limit) < required_classes:
        parser.error(
            f"{args.dataset} requires at least {required_classes} samples "
            "in each train/validation/test partition to include all classes"
        )
    root = Path(args.workdir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    data_root = root / "dataset"
    torch.set_num_threads(2)

    # Official torchvision source; explicit download is isolated from the
    # default dataset provider, whose downloads remain disabled by design.
    from torchvision import datasets as torchvision_datasets
    dataset_type = {
        "fashion_mnist": torchvision_datasets.FashionMNIST,
        "cifar10": torchvision_datasets.CIFAR10,
        "cifar100": torchvision_datasets.CIFAR100,
    }[args.dataset]
    dataset_type(root=str(data_root), train=True, download=True)
    dataset_type(root=str(data_root), train=False, download=True)
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
        missing_present_task_repeats=1, model_seeds=tuple(range(args.model_seeds)),
        width_to_depth_width=2, width_to_depth_depth=1,
        # Keep the runner's global baseline width grids valid. The selected
        # one-step RR jobs are independently bounded by init width, candidate
        # count, epochs and max_neurons.
        max_width=512, max_depth=5, max_neurons=128,
        max_candidates=1, max_epochs=None if args.step_balanced else 1,
        max_epochs_vanilla=2 if args.step_balanced else 0,
        max_epochs_adp=1 if args.step_balanced else 0,
        batch_size=args.batch_size,
        smoke_test=not args.step_balanced, cpu_only=True,
    )
    plan = SuitePlan.build(
        settings=settings,
        results_dir=root / "results", run_root=root / "runs",
        dataset_providers=tuple(datasets.providers.values()),
        dataset_suites=(("vision", args.dataset),),
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
        key = (job.model_adapter, job.route.model_id, job.identity.seed)
        if key in selected:
            raise ValueError(f"duplicate official matrix job: {key}")
        selected[key] = job

    run_seeds = sorted({key[2] for key in selected})
    if len(run_seeds) != args.model_seeds:
        raise ValueError("actual derived visual model seeds do not match the requested seed count")
    expected = {
        (name, method, run_seed)
        for run_seed in run_seeds
        for name in CONTROL_ADAPTERS
        for method in ("MLP-Fixed", "ADP-W-RR")
    }
    if set(selected) != expected:
        raise ValueError(f"official image matrix has missing/extra routes: {sorted(expected ^ set(selected))}")

    matrix_basename = {
        "fashion_mnist": "official_fashion_vision_matrix.json",
        "cifar10": "official_cifar10_vision_matrix.json",
        "cifar100": "official_cifar100_vision_matrix.json",
    }[args.dataset]
    if args.model_seeds > 1:
        matrix_basename = matrix_basename.replace(
            ".json", f"_{args.model_seeds}seeds.json",
        )
    if args.step_balanced:
        matrix_basename = matrix_basename.replace(".json", "_stepbalanced.json")
    summary_path = root / matrix_basename
    summary: dict[str, object] = {
        "format": (
            "simple-adp-official-visual-stepbalanced-controls-v1"
            if args.step_balanced else
            "simple-adp-official-visual-multiseed-controls-v1"
            if args.model_seeds > 1 else
            "simple-adp-official-fashion-visual-controls-v1"
            if args.dataset == "fashion_mnist"
            else "simple-adp-official-cifar-visual-controls-v1"
        ),
        "dataset": args.dataset,
        "source": f"torchvision.datasets.{dataset_type.__name__} official train/test",
        "dataset_download": "explicit",
        "train_limit": args.train_limit,
        "validation_limit": args.valid_limit,
        "test_limit": args.test_limit,
        "fixed_vs_growth_methods": ["MLP-Fixed", "ADP-W-RR"],
        "planned_full_vision_jobs": len(plan.runs),
        "selected_worker_jobs": len(selected),
        "scientific_scope": (
            "explicit optimizer-step balancing attempt: two fixed epochs vs "
            "two one-epoch ADP stages, actual steps individually audited; "
            "not whole-model FLOP matching or a scientific benchmark"
            if args.step_balanced else
            "small multiple-seed, one-epoch-per-candidate official-image smoke; "
            "not a matched-compute benchmark"
            if args.model_seeds > 1 else
            "small one-seed, one-epoch official-image smoke; not a benchmark"
        ),
        **({
            "requested_step_balance": {
                "fixed_epoch_cap": 2, "adp_per_candidate_epoch_cap": 1,
                "max_structural_proposals": 1,
                "batch_size": args.batch_size,
                "actual_optimizer_steps_checked_by_pair_audit": True,
                "whole_model_training_flops_matched": None,
            },
        } if args.step_balanced else {}),
        **({
            "model_seed_count": args.model_seeds,
            "derived_run_seeds": run_seeds,
        } if args.model_seeds > 1 else {}),
        "rows": [],
    }
    _save(summary_path, summary)
    failures = 0
    for adapter_name, method, run_seed in (
        (adapter, policy, seed)
        for seed in run_seeds
        for adapter in CONTROL_ADAPTERS
        for policy in ("MLP-Fixed", "ADP-W-RR")
    ):
        job = selected[(adapter_name, method, run_seed)]
        started = time.perf_counter()
        row: dict[str, object] = {
            "adapter": adapter_name, "method": method,
            "seed": run_seed,
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
            if value["metric_directions"].get("top_label_ece_10") != "min":
                raise AssertionError("top-label ECE metric direction missing")
            if value["metric_directions"].get("multiclass_brier") != "min":
                raise AssertionError("multiclass Brier metric direction missing")
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
            dense = compute.get("observed_linear_conv_macs_per_image", 0)
            analytical = compute.get("analytical_multihead_attention_macs_per_image", 0)
            if compute.get("partial_dense_attention_inclusive_macs_per_image") != dense + analytical:
                raise AssertionError("total partial MAC components are inconsistent")
            if adapter_name == "vision_v03" and analytical <= 0:
                raise AssertionError("V03 transformer self-attention MAC estimate was omitted")
            if counts["total_parameters"] != counts["frozen_parameters"] + counts["trainable_parameters"]:
                raise AssertionError("composite parameter counts do not close")
            memory = value.get("model_memory_accounting")
            if not memory or memory.get("total_model_tensor_payload_bytes", 0) <= 0:
                raise AssertionError("real composite tensor payload accounting missing")
            if memory["total_model_tensor_payload_bytes"] != (
                memory["parameter_payload_bytes"] + memory["buffer_payload_bytes"]
            ):
                raise AssertionError("model tensor payload partition does not close")
            if memory.get("peak_training_memory_bytes") is not None:
                raise AssertionError("logical model bytes must not claim peak training RAM")
            measured = value.get("fit_process_resources")
            if not measured or measured.get("format") != "simple-adp-visual-fit-process-resources-v1":
                raise AssertionError("measured fit process resource window is missing")
            if measured.get("exact_fit_window_peak_rss_bytes") is not None:
                raise AssertionError("two RSS snapshots cannot establish a fit interval peak")
            if measured.get("device") != "cpu":
                raise AssertionError("official bounded CPU matrix must report CPU fit resources")
            nominal = memory.get("nominal_dense_adam_or_adamw_model_grad_moments_bytes")
            if nominal != (
                memory["total_model_tensor_payload_bytes"]
                + 3 * memory["trainable_parameter_bytes"]
            ):
                raise AssertionError("nominal Adam/AdamW payload does not match model and active trainables")
            if method == "ADP-W-RR" and value["history_entries"] < 2:
                raise AssertionError("no actual structural candidate stage was executed")
            budget = value.get("training_budget")
            if adapter_name != "vision_adp":
                if not budget or not budget.get("complete") or budget.get("optimizer_steps", 0) < 1:
                    raise AssertionError("native inner-MLP optimizer/search budget is missing")
                if not value.get("trained_model_memory_trace_complete"):
                    raise AssertionError("native trained memory trace is incomplete")
                if len(value.get("trained_model_memory_trace", [])) != budget["trained_stages"]:
                    raise AssertionError("native memory trace does not cover every stage")
            row.update({
                "status": "completed",
                "dataset_id": value["dataset_id"],
                "seed": value["seed"],
                "architecture_id": value["architecture_id"],
                "growth_site": value["growth_site"],
                "selected_architecture": value["selected_architecture"],
                "candidate_stages_executed": value["history_entries"],
                "training_budget": value.get("training_budget"),
                "trained_model_memory_trace": value.get("trained_model_memory_trace"),
                "validation_metrics": value["validation_metrics"],
                "test_metrics": value["test_metrics"],
                "parameter_accounting": counts,
                "model_memory_accounting": memory,
                "fit_process_resources": measured,
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
    # A one-epoch-per-candidate smoke is not a compute-matched comparison.
    # In particular, native ADP typically trains more than one candidate;
    # V01's legacy tabular head does not expose the native-stage budget.
    if args.model_seeds == 1:
        summary["fixed_vs_growth_budget_audit"] = audit_visual_worker_pairs(
            summary["rows"], expected_adapters=CONTROL_ADAPTERS,
        )
    else:
        summary["fixed_vs_growth_budget_audit"] = audit_visual_multiseed_worker_pairs(
            summary["rows"], expected_adapters=CONTROL_ADAPTERS,
            expected_seeds=run_seeds,
        )
    _save(summary_path, summary)
    if failures:
        raise RuntimeError(f"{failures} official-image workers failed; consult {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
