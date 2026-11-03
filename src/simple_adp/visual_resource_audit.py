"""Descriptive paired memory evidence from independently executed visual workers.

This is NOT a compute, VRAM, optimizer-memory, or training-peak benchmark.
RSS high-water marks include Python/PyTorch startup and data loading.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .visual_budget_audit import audit_visual_multiseed_worker_pairs


def summarize_isolated_visual_resource_pairs(
    rows: Iterable[Mapping[str, Any]], *,
    expected_adapters: Iterable[str],
    expected_seeds: Iterable[int],
    worker_process_scope: str,
) -> dict[str, object]:
    """Describe *observed* per-child VmHWM and endpoint fit RSS differences.

    Fail closed when the matrix did not actually run one signed worker per
    process. The existing full identity auditor validates datasets, split,
    seeds, and growth sites before comparing memory evidence. The returned
    memory deltas are descriptive CPU process counters, not model allocation.
    """
    if worker_process_scope != "one-process-per-signed-run":
        raise ValueError("process memory comparisons require independently executed signed workers")
    source = tuple(rows)
    names = tuple(expected_adapters)
    seeds = tuple(expected_seeds)
    identity = audit_visual_multiseed_worker_pairs(
        source, expected_adapters=names, expected_seeds=seeds,
    )
    for group in identity["per_seed_budget_audits"]:
        if any(
            pair["identity_status"] != "paired-input-identity"
            for pair in group["paired_budget_evidence"]
        ):
            raise ValueError("cross-dataset/seed/split/site worker data cannot be compared")

    index = {(r["adapter"], r["method"], r["seed"]): r for r in source}
    per_pair: list[dict[str, object]] = []
    for seed in seeds:
        for adapter in names:
            fixed = index[(adapter, "MLP-Fixed", seed)]
            grown = index[(adapter, "ADP-W-RR", seed)]
            pair = []
            for row in (fixed, grown):
                fit = row.get("fit_process_resources")
                if not isinstance(fit, Mapping) or fit.get("format") != "simple-adp-visual-fit-process-resources-v1":
                    raise ValueError("worker lacks a valid observed fit resource record")
                if fit.get("exact_fit_window_peak_rss_bytes") is not None:
                    raise ValueError("two RSS snapshots must never claim exact interval peak")
                rss_before = fit.get("rss_current_before_bytes")
                rss_after = fit.get("rss_current_after_bytes")
                high = fit.get("lifetime_peak_rss_after_bytes")
                pid = fit.get("pid")
                if not all(
                    type(v) is int and v > 0
                    for v in (rss_before, rss_after, high, pid)
                ):
                    raise ValueError("isolated CPU RSS and process ID evidence is unavailable")
                if high < rss_before or high < rss_after:
                    raise ValueError("process lifetime high-water mark below observed RSS")
                if fit.get("device") != "cpu" or fit.get("cuda_pytorch_allocator") is not None:
                    raise ValueError("this descriptive audit is CPU RSS only; CUDA remains separately sourced")
                pair.append({
                    "pid": pid,
                    "rss_current_before_bytes": rss_before,
                    "rss_current_after_bytes": rss_after,
                    "rss_current_delta_bytes": rss_after - rss_before,
                    "process_lifetime_highwater_after_bytes": high,
                    "new_lifetime_peak_during_fit_bytes": fit.get("new_lifetime_peak_rss_during_fit_bytes"),
                })
            first, second = pair
            per_pair.append({
                "adapter": adapter,
                "seed": seed,
                "fixed": first,
                "growth": second,
                "growth_minus_fixed_lifetime_highwater_bytes": (
                    second["process_lifetime_highwater_after_bytes"]
                    - first["process_lifetime_highwater_after_bytes"]
                ),
                "growth_minus_fixed_fit_endpoint_delta_bytes": (
                    second["rss_current_delta_bytes"] - first["rss_current_delta_bytes"]
                ),
                "exact_fit_peak_comparison_available": False,
                "training_flops_matched": None,
                "scientific_performance_conclusion": None,
            })
    return {
        "format": "simple-adp-descriptive-isolated-visual-process-rss-pairs-v1",
        "scope": "isolated signed CPU process lifetime RSS; not model or training peak",
        "expected_pairs": len(names) * len(seeds),
        "pairs": per_pair,
        "actual_optimizer_step_pairs_same": identity["same_optimizer_step_pairs"],
        "actual_optimizer_step_pairs_unequal": identity["mismatched_optimizer_step_pairs"],
        "unknown_step_pairs": identity["unknown_optimizer_step_pairs"],
        "independent_seed_count": len(seeds),
        "uncertainty_intervals_computed": False,
        "full_training_flops_matched": None,
        "scientific_performance_conclusion": None,
        "limitations": (
            "Descriptive per-child RSS includes Python/runtime and data loading. "
            "Endpoint differences are not interval peaks; process VmHWM is "
            "lifetime-wide and cannot attribute allocations to trainable FFNs. "
            "No GPU VRAM, CUDA allocation, training FLOPs or significance inference."
        ),
    }
