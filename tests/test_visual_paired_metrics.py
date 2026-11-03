"""Strict descriptive visual fixed/RR metrics on signed same-split seed pairs."""
from __future__ import annotations

import copy
import math

import pytest

from simple_adp.visual_paired_metrics import (
    METRIC_DIRECTIONS, summarize_visual_paired_metrics,
)


def metrics(accuracy: float, log_loss: float) -> dict[str, float]:
    return {
        "accuracy": accuracy,
        "balanced_accuracy": accuracy,
        "macro_f1": accuracy * 0.8,
        "log_loss": log_loss,
        "multiclass_brier": accuracy / 2,
        "top_label_ece_10": accuracy / 3,
    }


def make_rows():
    rows = []
    for adapter in ("vision_inner", "vision_v03"):
        for seed in (101, 202):
            for method in ("MLP-Fixed", "ADP-W-RR"):
                changed = method == "ADP-W-RR"
                seed_effect = 0.05 if seed == 202 else 0.0
                rows.append({
                    "adapter": adapter, "method": method, "status": "completed",
                    "job_id": f"{adapter}/{seed}/{method}",
                    "seed": seed, "dataset_id": "official-dataset",
                    "split_sha256": "official-identical-split",
                    "architecture_id": "V02" if adapter == "vision_inner" else "V03",
                    "growth_site": "adapter:block1" if adapter == "vision_inner" else "ffn:block0",
                    "candidate_stages_executed": 2 if changed else 1,
                    "training_budget": {
                        "complete": True, "optimizer_steps": 6 if adapter == "vision_inner" else (8 if changed else 6),
                    },
                    "validation_metrics": metrics(
                        0.6 + seed_effect + (0.1 if changed else 0.0),
                        1.3 - (0.2 if changed else 0.0),
                    ),
                    "test_metrics": metrics(
                        0.5 + seed_effect + (0.1 if changed else 0.0),
                        1.4 - (0.2 if changed else 0.0),
                    ),
                })
    return rows


def evaluate(rows):
    return summarize_visual_paired_metrics(
        rows, expected_adapters=["vision_inner", "vision_v03"],
        expected_seeds=[101, 202],
    )


def test_descriptive_pairs_retain_metric_directions_and_seed_level_observations():
    report = evaluate(make_rows())
    assert report["paired_worker_count"] == 8
    assert report["actual_run_seeds"] == [101, 202]
    assert report["adapter_count"] == 2
    assert report["full_training_flops_matched"] is None
    assert report["scientific_performance_conclusion"] is None
    assert report["confidence_intervals_computed"] is False
    assert report["multiple_comparisons_adjusted"] is False
    assert [a["adapter"] for a in report["adapters"]] == ["vision_inner", "vision_v03"]
    first, second = report["adapters"]
    assert first["observed_optimizer_step_pair_status"] == "all-equal-optimizer-steps"
    assert second["observed_optimizer_step_pair_status"] == "unequal-optimizer-steps"
    val = first["metrics_by_split"]["validation_metrics"]
    assert list(val) == list(METRIC_DIRECTIONS)
    acc = val["accuracy"]
    assert acc["direction"] == "max"
    assert acc["wins"] == 2 and acc["ties"] == acc["losses"] == 0
    assert acc["mean_improvement"] == pytest.approx(0.1)
    assert acc["confidence_interval"] is None and acc["p_value"] is None
    assert [x["seed"] for x in acc["paired_seed_observations"]] == [101, 202]
    loss = first["metrics_by_split"]["test_metrics"]["log_loss"]
    assert loss["direction"] == "min"
    assert loss["mean_improvement"] == pytest.approx(0.2)
    # Calibration scores were deliberately made worse despite accuracy gains.
    brier = first["metrics_by_split"]["test_metrics"]["multiclass_brier"]
    assert brier["mean_improvement"] < 0
    assert brier["wins"] == 0 and brier["losses"] == 2
    assert report["adapters"][0]["metrics_by_split"]["test_metrics"]["accuracy"][
        "mean_improvement"
    ] == pytest.approx(0.1)


def test_pair_analysis_does_not_mutate_source_signed_rows():
    rows = make_rows()
    prior = copy.deepcopy(rows)
    evaluate(rows)
    assert rows == prior


@pytest.mark.parametrize("corruption", [
    lambda a: a.pop(),
    lambda a: a.append(copy.deepcopy(a[0])),
    lambda a: a[0].update(status="failed"),
    lambda a: a[0].update(dataset_id="different-dataset"),
    lambda a: a[0].update(split_sha256="different-split"),
    lambda a: a[0].update(growth_site="wrong-site"),
    lambda a: a[0].update(job_id=""),
    lambda a: a[0].update(job_id=a[1]["job_id"]),
    lambda a: a[0]["test_metrics"].update(accuracy=float("nan")),
    lambda a: a[0]["test_metrics"].update(accuracy=True),
    lambda a: a[0]["test_metrics"].update(accuracy=-0.05),
    lambda a: a[0]["validation_metrics"].update(log_loss=-1),
    lambda a: a[0]["test_metrics"].pop("multiclass_brier"),
])
def test_malformed_or_incomplete_paired_metric_sources_fail_closed(corruption):
    rows = make_rows()
    corruption(rows)
    with pytest.raises(ValueError):
        evaluate(rows)


def test_cross_seed_architecture_renaming_is_not_pooled():
    rows = make_rows()
    rows[4]["architecture_id"] = "changed architecture"
    rows[5]["architecture_id"] = "changed architecture"
    with pytest.raises(ValueError, match="different architecture/sites"):
        evaluate(rows)


def test_missing_native_budget_is_unknown_not_equal_and_still_descriptive():
    rows = make_rows()
    for row in rows:
        if row["adapter"] == "vision_inner":
            row["training_budget"] = None
    report = evaluate(rows)
    first = report["adapters"][0]
    assert first["observed_optimizer_step_pair_status"] == "unknown-optimizer-steps"
    assert first["metrics_by_split"]["test_metrics"]["accuracy"]["mean_improvement"] == pytest.approx(0.1)


def test_single_seed_has_no_sample_std_and_no_statistical_claim():
    rows = [r for r in make_rows() if r["seed"] == 101]
    report = summarize_visual_paired_metrics(
        rows, expected_adapters=["vision_inner", "vision_v03"], expected_seeds=[101],
    )
    metric = report["adapters"][0]["metrics_by_split"]["test_metrics"]["accuracy"]
    assert metric["seed_count"] == 1
    assert metric["sample_std_improvement"] is None
    assert metric["confidence_interval"] is None


def test_multiple_seeds_have_std_but_no_confidence_interval():
    rows = make_rows()
    rows[-1]["test_metrics"]["accuracy"] += 0.04
    report = evaluate(rows)
    metric = report["adapters"][1]["metrics_by_split"]["test_metrics"]["accuracy"]
    assert metric["sample_std_improvement"] > 0
    assert metric["confidence_interval"] is None
