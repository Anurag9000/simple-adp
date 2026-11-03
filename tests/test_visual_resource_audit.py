"""Descriptive resource comparisons must fail closed on provenance or unknown data."""
from __future__ import annotations

import copy

import pytest

from simple_adp.visual_resource_audit import summarize_isolated_visual_resource_pairs


def rows(seeds=(7, 9), adapters=("vision_inner",)):
    result = []
    for seed in seeds:
        for adapter in adapters:
            for method, offset in (("MLP-Fixed", 0), ("ADP-W-RR", 40)):
                result.append({
                    "adapter": adapter, "method": method, "seed": seed,
                    "status": "completed",
                    "dataset_id": "real-fashion-ids",
                    "split_sha256": "same-raw-split",
                    "architecture_id": "V02", "growth_site": "adapter:block1",
                    "training_budget": {"complete": True, "optimizer_steps": 6},
                    "fit_process_resources": {
                        "format": "simple-adp-visual-fit-process-resources-v1",
                        "pid": 3000 + 100 * seed + offset,
                        "device": "cpu", "cuda_pytorch_allocator": None,
                        "rss_current_before_bytes": 500,
                        "rss_current_after_bytes": 600 + offset,
                        "lifetime_peak_rss_after_bytes": 700 + offset,
                        "exact_fit_window_peak_rss_bytes": None,
                        "new_lifetime_peak_rss_during_fit_bytes": 700 + offset,
                    },
                })
    return result


def audit(records, **kwargs):
    return summarize_isolated_visual_resource_pairs(
        records, expected_adapters=("vision_inner",),
        expected_seeds=(7, 9),
        worker_process_scope=kwargs.get("scope", "one-process-per-signed-run"),
    )


def test_pair_process_hwm_and_endpoint_deltas_have_distinct_meanings():
    out = audit(rows())
    assert out["expected_pairs"] == 2
    assert out["independent_seed_count"] == 2
    assert out["actual_optimizer_step_pairs_same"] == 2
    assert out["unknown_step_pairs"] == 0
    assert out["full_training_flops_matched"] is None
    assert out["scientific_performance_conclusion"] is None
    assert out["uncertainty_intervals_computed"] is False
    for pair in out["pairs"]:
        assert pair["growth_minus_fixed_lifetime_highwater_bytes"] == 40
        assert pair["growth_minus_fixed_fit_endpoint_delta_bytes"] == 40
        assert pair["exact_fit_peak_comparison_available"] is False


def test_shared_process_matrix_cannot_be_memory_compared_as_isolated():
    with pytest.raises(ValueError, match="independently executed"):
        audit(rows(), scope="shared-driver-process")


@pytest.mark.parametrize("mutate", [
    lambda values: values.pop(),
    lambda values: values.append(copy.deepcopy(values[0])),
    lambda values: values[0].update({"dataset_id": "different"}),
    lambda values: values[0].update({"seed": 123}),
    lambda values: values[0].update({"split_sha256": "different-split"}),
    lambda values: values[0].update({"growth_site": "wrong-site"}),
])
def test_missing_duplicate_or_cross_identity_pairs_cannot_gain_memory_claims(mutate):
    records = rows()
    mutate(records)
    with pytest.raises(ValueError):
        audit(records)


@pytest.mark.parametrize("mutate", [
    lambda x: x.pop("fit_process_resources"),
    lambda x: x["fit_process_resources"].update({"exact_fit_window_peak_rss_bytes": 111}),
    lambda x: x["fit_process_resources"].update({"rss_current_before_bytes": None}),
    lambda x: x["fit_process_resources"].update({"rss_current_after_bytes": 0}),
    lambda x: x["fit_process_resources"].update({"lifetime_peak_rss_after_bytes": 1}),
    lambda x: x["fit_process_resources"].update({"device": "cuda:0"}),
    lambda x: x["fit_process_resources"].update({"cuda_pytorch_allocator": {"allocated_bytes": 50}}),
])
def test_invalid_or_unavailable_os_resource_metrics_are_never_guessed(mutate):
    records = rows()
    mutate(records[0])
    with pytest.raises(ValueError):
        audit(records)


def test_source_budget_unknown_remains_unknown_with_real_memory_data():
    records = rows(seeds=(7,), adapters=("vision_adp",))
    for row in records:
        row["training_budget"] = None
    out = summarize_isolated_visual_resource_pairs(
        records, expected_adapters=("vision_adp",),
        expected_seeds=(7,), worker_process_scope="one-process-per-signed-run",
    )
    assert out["unknown_step_pairs"] == 1
    assert out["actual_optimizer_step_pairs_same"] == 0
    assert out["expected_pairs"] == 1
    assert out["pairs"][0]["scientific_performance_conclusion"] is None
