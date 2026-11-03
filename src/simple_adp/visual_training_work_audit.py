"""Conservative paired *actual training-forward* operator work audit.

This compares recorded executed Linear/Conv2d forward MACs and explicitly
analytical dense MHA work, never backwards, optimizers or full model FLOPs.
It only accepts pairs with the existing native signed dataset/split/site/seed
identity contract and refuses corrupted complete-work counters.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .visual_budget_audit import audit_visual_multiseed_worker_pairs


_FORMAT = "simple-adp-native-actual-training-forward-mac-sum-v1"
_STAGE_FORMAT = "simple-adp-actual-training-forward-partial-macs-v1"
_COMPONENTS = (
    "observed_linear_conv_training_forward_macs",
    "analytical_mha_training_forward_macs",
    "recorded_training_forward_batches",
    "recorded_training_samples",
)


def _verified_job_work(row: Mapping[str, Any]) -> dict[str, object] | None:
    value = row.get("training_forward_compute")
    if value is None:
        return None
    if not isinstance(value, Mapping) or value.get("format") != _FORMAT:
        raise ValueError("unknown original signed visual training-forward summary format")
    if value.get("complete") is not True:
        return None  # Missing older stages cannot be silently counted as zero.
    if value.get("full_training_flops") is not None:
        raise ValueError("partial training forward accounting cannot claim full training FLOPs")
    budget = row.get("training_budget")
    stages = row.get("trained_forward_mac_trace")
    if not isinstance(budget, Mapping) or budget.get("complete") is not True:
        return None
    if row.get("trained_forward_mac_trace_complete") is not True:
        return None
    if not isinstance(stages, list) or not stages:
        raise ValueError("signed native stage operator-work evidence is missing")
    if type(budget.get("optimizer_steps")) is not int or budget["optimizer_steps"] <= 0:
        raise ValueError("native operator work requires real positive optimizer-step count")
    if type(budget.get("trained_stages")) is not int or budget["trained_stages"] != len(stages):
        raise ValueError("signed native training-stage count does not match source traces")
    counts: dict[str, int] = {}
    for key in _COMPONENTS:
        number = value.get(key)
        if type(number) is not int or number < 0:
            raise ValueError(f"invalid visual training-forward counter {key}")
        counts[key] = number
        subtotal = 0
        for stage in stages:
            if not isinstance(stage, Mapping) or stage.get("format") != _STAGE_FORMAT:
                raise ValueError("invalid signed stage forward-work format")
            if stage.get("complete") is not True:
                return None
            fragment = stage.get(key)
            if type(fragment) is not int or fragment < 0:
                raise ValueError(f"invalid signed stage MAC fragment {key}")
            subtotal += fragment
        if subtotal != number:
            raise ValueError(f"stage work does not close with aggregate {key}")
    if counts["recorded_training_forward_batches"] != budget["optimizer_steps"]:
        raise ValueError("recorded actual forward count must match optimizer work")
    if counts["recorded_training_samples"] < counts["recorded_training_forward_batches"]:
        raise ValueError("recorded training sample count smaller than forward batch count")
    combined = (
        counts["observed_linear_conv_training_forward_macs"]
        + counts["analytical_mha_training_forward_macs"]
    )
    if value.get("partial_training_forward_macs") != combined or combined <= 0:
        raise ValueError("partial native training-forward operator work does not close")
    return {
        **counts,
        "partial_training_forward_macs": combined,
        "partial_macs_per_forward_sample": (
            combined / counts["recorded_training_samples"]
        ),
    }


def audit_visual_training_forward_work_pairs(
    rows: Iterable[Mapping[str, Any]], *, expected_adapters: Iterable[str],
    expected_seeds: Iterable[int],
) -> dict[str, object]:
    """Compare measured forwards within strictly signed fixed/growth pairs.

    Output is descriptive partial-operator arithmetic, not an inference
    about model superiority or whole training compute parity.
    """
    records = tuple(rows)
    adapters = tuple(expected_adapters)
    seeds = tuple(expected_seeds)
    identity = audit_visual_multiseed_worker_pairs(
        records, expected_adapters=adapters, expected_seeds=seeds,
    )
    for group in identity["per_seed_budget_audits"]:
        if any(
            item["identity_status"] != "paired-input-identity"
            for item in group["paired_budget_evidence"]
        ):
            raise ValueError("training forward work comparison requires matching signed input identity")

    jobs = {(r["adapter"], r["method"], r["seed"]): r for r in records}
    comparisons: list[dict[str, object]] = []
    measured = 0
    missing = 0
    for seed in seeds:
        for adapter in adapters:
            fixed = jobs[(adapter, "MLP-Fixed", seed)]
            growth = jobs[(adapter, "ADP-W-RR", seed)]
            a = _verified_job_work(fixed)
            b = _verified_job_work(growth)
            if a is None or b is None:
                missing += 1
                comparisons.append({
                    "adapter": adapter, "seed": seed,
                    "status": "unknown-or-incomplete-forward-work",
                    "fixed": a, "growth": b,
                    "growth_to_fixed_partial_forward_mac_ratio": None,
                    "same_observed_optimizer_steps": None,
                    "full_training_flops_matched": None,
                })
                continue
            measured += 1
            equal_steps = (
                a["recorded_training_forward_batches"]
                == b["recorded_training_forward_batches"]
            )
            comparisons.append({
                "adapter": adapter, "seed": seed,
                "status": "measured-partial-training-forwards",
                "fixed": a, "growth": b,
                "same_observed_optimizer_steps": equal_steps,
                "growth_to_fixed_partial_forward_mac_ratio": (
                    b["partial_training_forward_macs"]
                    / a["partial_training_forward_macs"]
                ),
                "growth_to_fixed_partial_forward_mac_per_sample_ratio": (
                    b["partial_macs_per_forward_sample"]
                    / a["partial_macs_per_forward_sample"]
                ),
                "full_training_flops_matched": None,
            })
    return {
        "format": "simple-adp-signed-paired-actual-training-forward-work-v1",
        "dataset_id": next(iter({r["dataset_id"] for r in records})),
        "seeds": list(seeds),
        "expected_pairs": len(seeds) * len(adapters),
        "measured_pairs": measured,
        "unknown_pairs": missing,
        "pairs": comparisons,
        "training_work_scope": (
            "executed main forward nn.Linear/Conv2d MACs plus "
            "shape-analytical dense nn.MultiheadAttention"
        ),
        "backward_flops": None,
        "optimizer_flops": None,
        "full_training_flops_matched": None,
        "scientific_performance_conclusion": None,
    }
