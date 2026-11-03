"""Conservative paired visual accuracy/calibration summaries across true worker seeds.

This module reports descriptive *within-seed* differences for each controlled
architecture/site separately. It does NOT infer statistical significance,
compute confidence intervals, match training FLOPs, or compare across sites.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .visual_budget_audit import audit_visual_multiseed_worker_pairs

METRIC_DIRECTIONS = {
    "accuracy": "max",
    "balanced_accuracy": "max",
    "macro_f1": "max",
    "log_loss": "min",
    "multiclass_brier": "min",
    "top_label_ece_10": "min",
}
SPLITS = ("validation_metrics", "test_metrics")


def _metric_value(row: Mapping[str, Any], split: str, name: str) -> float:
    group = row.get(split)
    if not isinstance(group, Mapping) or name not in group:
        raise ValueError(f"missing {split}.{name} in worker metric record")
    value = group[name]
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError(f"{split}.{name} must be a finite numeric value")
    number = float(value)
    if name in {"accuracy", "balanced_accuracy", "macro_f1", "top_label_ece_10"}:
        if not 0 <= number <= 1:
            raise ValueError(f"{split}.{name} outside the valid [0,1] interval")
    elif number < 0 or (name == "multiclass_brier" and number > 2):
        raise ValueError(f"{split}.{name} outside its valid range")
    return number


def summarize_visual_paired_metrics(
    rows: Iterable[Mapping[str, Any]], *,
    expected_adapters: Iterable[str],
    expected_seeds: Iterable[int],
) -> dict[str, object]:
    """Compute descriptive paired deltas, using the *signed worker metadata*.

    All seeds must share a single dataset and split fingerprint. Inferred
    intervals from two seeds would be especially misleading, so uncertainty
    intervals are intentionally absent at *every* seed count. Full dataset
    and equal FLOP claims must be established by other provenance.
    """
    records = tuple(rows)
    adapters = tuple(expected_adapters)
    seeds = tuple(expected_seeds)
    audit = audit_visual_multiseed_worker_pairs(
        records, expected_adapters=adapters, expected_seeds=seeds,
    )
    for run in audit["per_seed_budget_audits"]:
        if any(
            pair["identity_status"] != "paired-input-identity"
            for pair in run["paired_budget_evidence"]
        ):
            raise ValueError("paired metric analysis requires completed matching worker inputs")

    ids = [r.get("job_id") for r in records]
    if any(not isinstance(job, str) or not job.strip() for job in ids) or (
        len(ids) != len(set(ids))
    ):
        raise ValueError("worker job IDs must be present and globally unique")

    datasets = {r.get("dataset_id") for r in records}
    splits = {r.get("split_sha256") for r in records}
    if len(datasets) != 1 or len(splits) != 1:
        raise ValueError(
            "paired seed aggregation requires identical dataset and split "
            "fingerprints across all controlled runs"
        )
    dataset_id, split_sha = next(iter(datasets)), next(iter(splits))
    if not isinstance(dataset_id, str) or not dataset_id or (
        not isinstance(split_sha, str) or not split_sha
    ):
        raise ValueError("missing signed dataset/split identity")

    index = {(r["adapter"], r["method"], r["seed"]): r for r in records}
    per_seed_audits = {
        group["run_seed"]: {
            pair["adapter"]: pair
            for pair in group["paired_budget_evidence"]
        }
        for group in audit["per_seed_budget_audits"]
    }
    adapters_out: list[dict[str, object]] = []
    for adapter in adapters:
        geometry = {
            (index[(adapter, "MLP-Fixed", seed)]["architecture_id"],
             index[(adapter, "MLP-Fixed", seed)]["growth_site"])
            for seed in seeds
        }
        if len(geometry) != 1:
            raise ValueError(
                "the same adapter label refers to different architecture/sites across seeds"
            )
        architecture_id, growth_site = next(iter(geometry))
        budgets = [per_seed_audits[seed][adapter] for seed in seeds]
        same = [b["same_optimizer_step_count"] for b in budgets]
        parity = (
            "all-equal-optimizer-steps" if all(v is True for v in same)
            else "unknown-optimizer-steps" if any(v is None for v in same)
            else "unequal-optimizer-steps"
        )
        by_split: dict[str, object] = {}
        for split in SPLITS:
            summarized_metrics: dict[str, object] = {}
            for name, direction in METRIC_DIRECTIONS.items():
                points = []
                for seed in seeds:
                    fixed = _metric_value(index[(adapter, "MLP-Fixed", seed)], split, name)
                    growth = _metric_value(index[(adapter, "ADP-W-RR", seed)], split, name)
                    improvement = (
                        growth - fixed if direction == "max" else fixed - growth
                    )
                    points.append({
                        "seed": seed,
                        "fixed": fixed,
                        "growth": growth,
                        "improvement_positive_is_better": float(improvement),
                    })
                values = [point["improvement_positive_is_better"] for point in points]
                mean = math.fsum(values) / len(values)
                summarized_metrics[name] = {
                    "direction": direction,
                    "seed_count": len(values),
                    "paired_seed_observations": points,
                    "mean_improvement": mean,
                    "median_improvement": statistics.median(values),
                    "sample_std_improvement": (
                        statistics.stdev(values) if len(values) >= 2 else None
                    ),
                    "wins": sum(value > 0 for value in values),
                    "ties": sum(value == 0 for value in values),
                    "losses": sum(value < 0 for value in values),
                    "confidence_interval": None,
                    "p_value": None,
                }
            by_split[split] = summarized_metrics
        adapters_out.append({
            "adapter": adapter,
            "architecture_id": architecture_id,
            "growth_site": growth_site,
            "model_seed_count": len(seeds),
            "observed_optimizer_step_pair_status": parity,
            "training_flops_matched": None,
            "metrics_by_split": by_split,
        })

    return {
        "format": "simple-adp-visual-descriptive-within-seed-paired-metrics-v1",
        "dataset_id": dataset_id,
        "split_sha256": split_sha,
        "actual_run_seeds": list(seeds),
        "adapter_count": len(adapters),
        "paired_worker_count": len(records),
        "metric_directions": dict(METRIC_DIRECTIONS),
        "adapters": adapters_out,
        "full_training_flops_matched": None,
        "confidence_intervals_computed": False,
        "multiple_comparisons_adjusted": False,
        "scientific_performance_conclusion": None,
        "limitations": (
            "Descriptive post-selection differences conditional on this exact "
            "held-out split and these model seeds, not population inference. "
            "No confidence intervals, p-values, full training FLOPs, causal "
            "attribution, or across-adapter pooling. Missing matching worker "
            "identities and malformed metrics fail closed."
        ),
    }


def verify_visual_worker_result_artifacts(
    rows: Iterable[Mapping[str, Any]], *, results_root: str | Path,
) -> dict[str, object]:
    """Re-read actual signed result.json files and fingerprint their exact bytes.

    This proves the *summary agrees with locally supplied original worker
    result artifacts*, and all workers share one suite plan signature.
    It does not authenticate GitHub or constitute a public-key
    signature: run_signature is the original runner's plan hash.
    """
    import hashlib
    import json
    import re
    root = Path(results_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("worker results root must be an existing directory")
    records = tuple(rows)
    if not records:
        raise ValueError("no worker result rows supplied for provenance verification")
    fingerprints = []
    seen: set[str] = set()
    signatures: set[str] = set()
    for row in records:
        if not isinstance(row, Mapping):
            raise ValueError("every source worker summary must be a mapping")
        job = row.get("job_id")
        if not isinstance(job, str) or not job or job in seen:
            raise ValueError("missing or duplicated original signed worker job ID")
        seen.add(job)
        relative = Path(job)
        if relative.is_absolute() or any(part in (".", "..") for part in relative.parts):
            raise ValueError("signed job ID must not escape its worker results directory")
        file = (root / relative / "result.json").resolve()
        if not file.is_relative_to(root) or not file.is_file():
            raise ValueError("original signed worker result is missing or escapes result root")
        payload_bytes = file.read_bytes()
        result = json.loads(payload_bytes)
        if not isinstance(result, dict):
            raise ValueError("worker result JSON must contain a single mapping")
        signature = result.get("run_signature")
        if not isinstance(signature, str) or not re.fullmatch(r"[0-9a-f]{64}", signature):
            raise ValueError("original worker result has an invalid or missing run signature")
        signatures.add(signature)
        for key in (
            "job_id", "status", "dataset_id", "split_sha256", "seed",
            "architecture_id", "growth_site",
            "validation_metrics", "test_metrics", "selected_architecture",
        ):
            if result.get(key) != row.get(key):
                raise ValueError(f"signed worker result mismatches matrix field {key!r}")
        if result.get("model_id") != row.get("method"):
            raise ValueError("worker model ID does not match matrix controller method")
        if result.get("training_budget") != row.get("training_budget"):
            raise ValueError("signed native optimizer work differs from matrix summary")
        fingerprints.append((job, hashlib.sha256(payload_bytes).hexdigest()))
    if len(signatures) != 1:
        raise ValueError("original worker result files belong to different signed suite plans")
    digest = hashlib.sha256()
    for job, checksum in sorted(fingerprints):
        digest.update(job.encode("utf-8") + bytes([0]) + checksum.encode("ascii") + bytes([10]))
    return {
        "format": "simple-adp-visual-original-worker-result-artifact-audit-v1",
        "verified_worker_results": len(records),
        "shared_suite_plan_run_signature": next(iter(signatures)),
        "shared_plan_signature_consistency_verified": True,
        "original_worker_result_bundle_sha256": digest.hexdigest(),
        "source_scope": (
            "Matrix row agreement with original signed-worker result.json "
            "content; not cryptographic authentication of the runner owner"
        ),
    }
