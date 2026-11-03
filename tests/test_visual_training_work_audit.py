"""Strict paired real-training-forward work, never fabricated whole training FLOPs."""
from __future__ import annotations

import copy

import pytest

from simple_adp.visual_training_work_audit import audit_visual_training_forward_work_pairs


def _stage(kind, macs, batches, samples):
    return {
        "format": "simple-adp-actual-training-forward-partial-macs-v1",
        "stage": kind, "complete": True,
        "observed_linear_conv_training_forward_macs": macs,
        "analytical_mha_training_forward_macs": 0,
        "recorded_training_forward_batches": batches,
        "recorded_training_samples": samples,
        "partial_training_forward_macs": macs,
    }


def _work(stages):
    return {
        "format": "simple-adp-native-actual-training-forward-mac-sum-v1",
        "complete": True, "full_training_flops": None,
        "observed_linear_conv_training_forward_macs": sum(
            x["observed_linear_conv_training_forward_macs"] for x in stages
        ),
        "analytical_mha_training_forward_macs": 0,
        "recorded_training_forward_batches": sum(
            x["recorded_training_forward_batches"] for x in stages
        ),
        "recorded_training_samples": sum(
            x["recorded_training_samples"] for x in stages
        ),
        "partial_training_forward_macs": sum(x["partial_training_forward_macs"] for x in stages),
    }


def _rows():
    rows = []
    for seed in (11, 22):
        for adapter in ("vision_inner", "vision_adp"):
            for method in ("MLP-Fixed", "ADP-W-RR"):
                if method == "MLP-Fixed":
                    trace = [_stage("fixed", 1000, 4, 16)]
                else:
                    trace = [_stage("initial", 600, 2, 8), _stage("width", 700, 2, 8)]
                row = {
                    "adapter": adapter, "method": method, "seed": seed,
                    "status": "completed", "dataset_id": "official-fashion",
                    "split_sha256": "same-split-across-all-seeds",
                    "architecture_id": "V02" if adapter == "vision_inner" else "V01",
                    "growth_site": "adapter:block1" if adapter == "vision_inner" else "head",
                    "training_budget": {
                        "complete": True, "optimizer_steps": 4,
                        "trained_stages": len(trace),
                    } if adapter == "vision_inner" else None,
                    "job_id": f"{seed}/{adapter}/{method}",
                }
                if adapter == "vision_inner":
                    row["training_forward_compute"] = _work(trace)
                    row["trained_forward_mac_trace"] = trace
                    row["trained_forward_mac_trace_complete"] = True
                rows.append(row)
    return rows


def _audit(rows):
    return audit_visual_training_forward_work_pairs(
        rows, expected_adapters=["vision_inner", "vision_adp"],
        expected_seeds=[11, 22],
    )


def test_known_partial_train_forward_work_paired_by_real_seed_not_legacy():
    result = _audit(_rows())
    assert result["expected_pairs"] == 4
    assert result["measured_pairs"] == 2
    assert result["unknown_pairs"] == 2
    assert result["seeds"] == [11, 22]
    assert result["full_training_flops_matched"] is None
    assert result["scientific_performance_conclusion"] is None
    for row in result["pairs"]:
        if row["adapter"] == "vision_adp":
            assert row["status"] == "unknown-or-incomplete-forward-work"
            assert row["growth_to_fixed_partial_forward_mac_ratio"] is None
            continue
        assert row["status"] == "measured-partial-training-forwards"
        assert row["same_observed_optimizer_steps"] is True
        assert row["fixed"]["recorded_training_samples"] == 16
        assert row["fixed"]["partial_training_forward_macs"] == 1000
        assert row["growth"]["partial_training_forward_macs"] == 1300
        assert row["growth_to_fixed_partial_forward_mac_ratio"] == pytest.approx(1.3)
        assert row["full_training_flops_matched"] is None


@pytest.mark.parametrize("corruption", [
    "wrong-mac-component", "wrong-stage-sum", "fake-full-flops",
    "fake-optimizer-steps", "missing-stages", "cross-seed-dataset",
    "cross-seed-split", "bad-native-format",
])
def test_corrupt_work_or_cross_seed_sources_fail_closed(corruption):
    rows = _rows()
    target = next(r for r in rows if r["adapter"] == "vision_inner" and r["seed"] == 22)
    if corruption == "wrong-mac-component":
        target["training_forward_compute"]["partial_training_forward_macs"] += 1
    elif corruption == "wrong-stage-sum":
        target["trained_forward_mac_trace"][0]["observed_linear_conv_training_forward_macs"] += 1
    elif corruption == "fake-full-flops":
        target["training_forward_compute"]["full_training_flops"] = 999
    elif corruption == "fake-optimizer-steps":
        target["training_budget"]["optimizer_steps"] = 5
    elif corruption == "missing-stages":
        target["trained_forward_mac_trace"].pop()
    elif corruption == "cross-seed-dataset":
        for r in rows:
            if r["seed"] == 22:
                r["dataset_id"] = "unrelated-source"
    elif corruption == "cross-seed-split":
        for r in rows:
            if r["seed"] == 22:
                r["split_sha256"] = "another-split"
    else:
        target["training_forward_compute"]["format"] = "invented"
    with pytest.raises(ValueError):
        _audit(rows)


def test_unknown_from_unmeasured_training_does_not_become_zero_cost():
    rows = _rows()
    target = next(r for r in rows if r["adapter"] == "vision_inner" and r["seed"] == 11)
    target.pop("training_forward_compute")
    result = _audit(rows)
    assert result["unknown_pairs"] == 3
    assert result["measured_pairs"] == 1
    rows = _rows()
    target = next(r for r in rows if r["adapter"] == "vision_inner" and r["seed"] == 11)
    target["trained_forward_mac_trace_complete"] = False
    result = _audit(rows)
    assert result["unknown_pairs"] == 3
    assert result["measured_pairs"] == 1


def test_paired_observer_never_mutates_worker_result_records():
    source = _rows()
    original = copy.deepcopy(source)
    _audit(source)
    assert source == original
