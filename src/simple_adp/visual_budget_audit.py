"""Conservative fixed-vs-growth work ledger for official visual worker pairs.

One epoch per candidate is NOT one equal-budget experiment: RR/D1 may train
multiple candidates or warmup stages. Equal optimizer-step counts alone also
do not establish equal FLOPs, GPU utilization or statistical significance.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def audit_visual_worker_pairs(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_adapters: Iterable[str],
) -> dict[str, object]:
    """Inspect fixed/RR pairs; fail closed on missing/partial provenance.

    This audits observed native optimizer steps, not matched full training
    compute. The V01 head's legacy controller has no native step report and
    is always classified as unknown rather than implicitly equal.
    """
    methods = ("MLP-Fixed", "ADP-W-RR")
    names = tuple(expected_adapters)
    if not names or len(set(names)) != len(names):
        raise ValueError("expected_adapters must be nonempty and distinct")
    index: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        adapter, method = row.get("adapter"), row.get("method")
        if adapter not in names or method not in methods:
            raise ValueError("unrecognized visual adapter/method in pair ledger")
        key = (adapter, method)
        if key in index:
            raise ValueError(f"duplicate visual worker pair entry: {key}")
        index[key] = row
    required = {(adapter, method) for adapter in names for method in methods}
    if index.keys() != required:
        raise ValueError(f"missing official worker pair entries: {sorted(required - index.keys())}")

    comparisons: list[dict[str, object]] = []
    same_step_pairs = 0
    mismatched_step_pairs = 0
    unknown_step_pairs = 0
    for adapter in names:
        fixed, grown = (index[(adapter, method)] for method in methods)
        if fixed.get("status") != "completed" or grown.get("status") != "completed":
            status = "worker-failed"
        elif not fixed.get("split_sha256") or (
            fixed.get("split_sha256") != grown.get("split_sha256")
        ):
            status = "split-mismatch"
        elif fixed.get("architecture_id") != grown.get("architecture_id") or (
            fixed.get("growth_site") != grown.get("growth_site")
        ):
            status = "architecture-or-site-mismatch"
        else:
            status = "paired-input-identity"
        fixed_budget, grown_budget = fixed.get("training_budget"), grown.get("training_budget")
        budgets_complete = bool(
            isinstance(fixed_budget, Mapping)
            and isinstance(grown_budget, Mapping)
            and fixed_budget.get("complete") is True
            and grown_budget.get("complete") is True
        )
        if budgets_complete:
            fixed_steps = fixed_budget.get("optimizer_steps")
            grown_steps = grown_budget.get("optimizer_steps")
            valid = (
                isinstance(fixed_steps, int) and not isinstance(fixed_steps, bool)
                and isinstance(grown_steps, int) and not isinstance(grown_steps, bool)
                and fixed_steps > 0 and grown_steps > 0
            )
            if not valid:
                budgets_complete = False
        if budgets_complete and status == "paired-input-identity":
            same_steps: bool | None = fixed_steps == grown_steps
            if same_steps:
                same_step_pairs += 1
            else:
                mismatched_step_pairs += 1
            ratio: float | None = float(grown_steps / fixed_steps)
        else:
            same_steps = None
            ratio = None
            unknown_step_pairs += 1
        comparisons.append({
            "adapter": adapter,
            "identity_status": status,
            "fixed_optimizer_steps": fixed_budget.get("optimizer_steps") if isinstance(fixed_budget, Mapping) else None,
            "growth_optimizer_steps": grown_budget.get("optimizer_steps") if isinstance(grown_budget, Mapping) else None,
            "both_budgets_complete": budgets_complete,
            "same_optimizer_step_count": same_steps,
            "growth_to_fixed_step_ratio": ratio,
            "fixed_candidate_stages": fixed.get("candidate_stages_executed"),
            "growth_candidate_stages": grown.get("candidate_stages_executed"),
            "full_training_flops_matched": None,
            "scientific_performance_conclusion": None,
        })
    return {
        "format": "simple-adp-visual-fixed-rr-pair-budget-audit-v1",
        "expected_pairs": len(names),
        "same_optimizer_step_pairs": same_step_pairs,
        "mismatched_optimizer_step_pairs": mismatched_step_pairs,
        "unknown_optimizer_step_pairs": unknown_step_pairs,
        "full_training_flops_matched": None,
        "scientific_performance_conclusion": None,
        "paired_budget_evidence": comparisons,
    }
