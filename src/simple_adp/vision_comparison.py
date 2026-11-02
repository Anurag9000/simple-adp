"""Paired V01 real-image comparison: fixed head vs loss ADP vs variance ADP.

A comparison is three *distinct* signed jobs per seed, sharing exactly one
official training/validation/test partition, source encoder, and hyperparameters
except the expansion criterion. This does not choose a winner by the official
test set. Results and paired invariants remain auditable on interruption.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from .checkpoint import stable_signature
from .config import SearchSettings
from .vision_data import ImagePartitions, load_torchvision_partitions
from .vision_experiment import _atomic_json, _file_sha256, run_vision_experiment

ROUTES = (
    ("fixed", "MLP-Fixed", "loss"),
    ("adp_loss", "ADP-W-RR", "loss"),
    ("adp_variance", "ADP-W-RR", "variance_stagnation"),
)


def _seeds(values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(v) for v in values)
    if not result or len(set(result)) != len(result) or any(v < 0 for v in result):
        raise ValueError("comparison seeds must be nonnegative, nonempty and unique")
    return result


def _with_criterion(settings: SearchSettings, criterion: str, seed: int) -> SearchSettings:
    return replace(settings, expansion_criterion=criterion, seed=seed)


def run_vision_comparison(
    *,
    dataset: str,
    root: str | Path,
    output_dir: str | Path,
    architecture_id: str,
    weights_path: str | Path | None,
    expected_weights_sha256: str | None = None,
    allow_random_backbone: bool = False,
    seeds: Sequence[int] = (11, 23, 37),
    settings: SearchSettings | None = None,
    fixed_width: int = 16,
    train_limit: int | None = None,
    validation_limit: int | None = None,
    test_limit: int | None = None,
    validation_fraction: float = 0.2,
    download: bool = False,
    encoder_batch_size: int = 32,
    # Test-only fixture hooks; never exposed by CLI.
    partitions: ImagePartitions | None = None,
    encoder: nn.Module | None = None,
    encoder_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if dataset not in {"fashion_mnist", "cifar10", "cifar100"}:
        raise ValueError("unsupported official image dataset")
    if architecture_id not in {"resnet18", "convnext_tiny", "vit_b_16"}:
        raise ValueError("unsupported frozen source architecture")
    if fixed_width <= 0:
        raise ValueError("fixed control width must be positive")
    if not allow_random_backbone and weights_path is None:
        raise ValueError("strict source comparison requires local source weights")
    seeds = _seeds(seeds)
    if encoder is not None and (encoder_metadata is None or weights_path is not None or not allow_random_backbone):
        raise ValueError("test fixture encoder requires explicit random control metadata")
    if partitions is not None and len(seeds) != 1:
        raise ValueError("one injected offline image partition supports exactly one seed")
    if settings is None:
        settings = SearchSettings(
            init_width=16, init_depth=1, max_width=24,
            max_depth=2, max_candidates=2, max_epochs=4,
            batch_size=32, device="cuda" if torch.cuda.is_available() else "cpu",
        )

    weights_sha = _file_sha256(Path(weights_path)) if weights_path is not None else None
    if (expected_weights_sha256 is not None and
            (weights_sha is None or expected_weights_sha256.lower() != weights_sha.lower())):
        raise ValueError("comparison source weights SHA256 mismatch")

    # Predeclare *all* partitions before running any job. These only use
    # original train-source labels to stratify architecture validation.
    subsets: dict[int, ImagePartitions] = {}
    for seed in seeds:
        subsets[seed] = (partitions if partitions is not None else
                         load_torchvision_partitions(
                             dataset, root, download=download, seed=seed,
                             validation_fraction=validation_fraction,
                             train_limit=train_limit,
                             validation_limit=validation_limit,
                             test_limit=test_limit,
                         ))
    plan: dict[str, Any] = {
        "format": "simple-adp-v01-paired-comparison-v1",
        "dataset": dataset, "architecture_id": architecture_id,
        "source_weights_sha256": weights_sha,
        "source_weights_path": str(Path(weights_path).resolve()) if weights_path else None,
        "random_backbone_explicit": bool(allow_random_backbone),
        "root": str(Path(root).resolve()),
        "seeds": list(seeds),
        "fixed_width": fixed_width,
        "limits": {"train": train_limit, "validation": validation_limit, "test": test_limit},
        "validation_fraction": validation_fraction,
        "encoder_batch_size": encoder_batch_size,
        "settings_base": settings.as_dict(),
        "routes": [{"name": name, "method": method, "criterion": criterion}
                   for name, method, criterion in ROUTES],
        "split_manifests": {str(seed): subsets[seed].manifest for seed in seeds},
        "official_test_use": "metrics only after each job completes; never select architecture or route",
        "comparison_decision": "report all metrics, no official-test-based winner selection",
    }
    plan["comparison_signature"] = stable_signature(plan)
    out = Path(output_dir)
    plan_path = out / "comparison_manifest.json"
    summary_path = out / "comparison_summary.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("comparison plan identity mismatch; use a new output directory")
    else:
        if summary_path.exists() or (out / "runs").exists():
            raise ValueError("comparison has artifacts but no signed plan")
        _atomic_json(plan_path, plan)

    summary_rows: list[dict[str, Any]] = []
    for seed in seeds:
        per_seed: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        for name, method, criterion in ROUTES:
            folder = out / "runs" / f"seed{seed}" / name
            result = run_vision_experiment(
                dataset=dataset,
                root=root,
                output_dir=folder,
                architecture_id=architecture_id,
                model_id=method,
                settings=_with_criterion(settings, criterion, seed),
                fixed_grid=((fixed_width,),) if method == "MLP-Fixed" else None,
                weights_path=weights_path,
                expected_weights_sha256=weights_sha,
                allow_random_backbone=allow_random_backbone,
                train_limit=train_limit,
                validation_limit=validation_limit,
                test_limit=test_limit,
                validation_fraction=validation_fraction,
                split_seed=seed,
                download=False,  # partitions already materialized once
                encoder_batch_size=encoder_batch_size,
                partitions=subsets[seed],
                encoder=encoder,
                encoder_metadata=encoder_metadata,
            )
            experiment_manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            if experiment_manifest["partitions"] != subsets[seed].manifest:
                raise ValueError("paired job changed its official source split")
            if result["run_signature"] != experiment_manifest["run_signature"]:
                raise ValueError("paired job result signature changed")
            per_seed.append((name, result, experiment_manifest))

        # Exact pair integrity, including the independent fixed control.
        reference = per_seed[0][2]
        for _, result, record in per_seed:
            if record["partitions"] != reference["partitions"]:
                raise ValueError("paired comparison uses inconsistent official data splits")
            if (record["source_encoder_state_sha256"] != reference["source_encoder_state_sha256"]
                    or record["input_transform"] != reference["input_transform"]):
                raise ValueError("paired comparison source backbone/preprocessing changed")
            if result["sample_counts"] != per_seed[0][1]["sample_counts"]:
                raise ValueError("paired comparison source sample counts changed")
            comparable = dict(record["settings"])
            comparable.pop("expansion_criterion")
            comparable.pop("seed")
            reference_comparable = dict(reference["settings"])
            reference_comparable.pop("expansion_criterion")
            reference_comparable.pop("seed")
            if comparable != reference_comparable:
                raise ValueError("paired comparison hyperparameters differ beyond criterion")
        for name, result, record in per_seed:
            folder = out / "runs" / f"seed{seed}" / name
            summary_rows.append({
                "seed": seed, "route": name, "method_id": record["model_id"],
                "criterion": record["settings"]["expansion_criterion"],
                "run_signature": result["run_signature"],
                "source_encoder_state_sha256": result["source_encoder_state_sha256"],
                "official_split_sha256": stable_signature(record["partitions"]),
                "result_file": str(Path("runs") / f"seed{seed}" / name / "result.json"),
                "result_file_sha256": _file_sha256(folder / "result.json"),
                "selected_architecture": result["selected_architecture"],
                "validation_metrics": result["validation_metrics"],
                "official_test_metrics_descriptive_only": result["test_metrics"],
            })

    summary = {
        "format": "simple-adp-v01-paired-summary-v1",
        "comparison_signature": plan["comparison_signature"],
        "status": "completed",
        "completed_jobs": len(summary_rows),
        "rows": summary_rows,
        "evaluation_policy": "Report every official test result descriptively; do not select a method from test results",
    }
    if summary_path.exists() and json.loads(summary_path.read_text(encoding="utf-8")) != summary:
        raise ValueError("completed comparison summary no longer matches verified jobs")
    _atomic_json(summary_path, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V01 matched frozen/loss/variance comparison")
    parser.add_argument("--dataset", required=True, choices=("fashion_mnist", "cifar10", "cifar100"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--architecture", required=True, choices=("resnet18", "convnext_tiny", "vit_b_16"))
    parser.add_argument("--weights", help="complete local source torchvision checkpoint")
    parser.add_argument("--weights-sha256")
    parser.add_argument("--allow-random-backbone", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--seeds", default="11,23,37")
    parser.add_argument("--fixed-width", type=int, default=16)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--valid-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=2)
    parser.add_argument("--init-width", type=int, default=16)
    parser.add_argument("--max-width", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--encoder-batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    seeds = tuple(int(x) for x in args.seeds.split(","))
    cfg = SearchSettings(
        seed=0, device=args.device, init_width=args.init_width, init_depth=1,
        max_width=args.max_width, max_depth=2,
        max_epochs=args.epochs, max_candidates=args.candidates,
        batch_size=args.batch_size,
    )
    summary = run_vision_comparison(
        dataset=args.dataset, root=args.root, output_dir=args.out,
        architecture_id=args.architecture,
        weights_path=args.weights, expected_weights_sha256=args.weights_sha256,
        allow_random_backbone=args.allow_random_backbone,
        seeds=seeds, settings=cfg, fixed_width=args.fixed_width,
        download=args.download, train_limit=args.train_limit,
        validation_limit=args.valid_limit, test_limit=args.test_limit,
        encoder_batch_size=args.encoder_batch_size,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
