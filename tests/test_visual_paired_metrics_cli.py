"""End-to-end post-run analyzer works on complete signed-summary-shaped JSON."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest


ADAPTERS = (
    "vision_adp", "vision_inner", "vision_v03", "vision_v04", "vision_v05",
    "vision_v06a", "vision_v06b", "vision_v07_token", "vision_v07_channel",
    "vision_v07_both", "vision_v08_early", "vision_v08_middle",
    "vision_v08_late", "vision_v08_schedule", "vision_v09_bank0",
    "vision_v09_bank1", "vision_v09_bank2", "vision_v10", "vision_v11",
)
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "visual_paired_metrics_analysis.py"


def data_records():
    metric = {
        "accuracy": 0.5, "balanced_accuracy": 0.5, "macro_f1": 0.4,
        "log_loss": 1.0, "multiclass_brier": 0.4, "top_label_ece_10": 0.1,
    }
    rows = []
    for i, adapter in enumerate(ADAPTERS):
        for method in ("MLP-Fixed", "ADP-W-RR"):
            rows.append({
                "adapter": adapter,
                "method": method,
                "status": "completed",
                "job_id": f"{adapter}:{method}",
                "dataset_id": "official-dataset-fingerprint",
                "split_sha256": "signed-source-split",
                "seed": 401,
                "architecture_id": f"V{i:02d}",
                "growth_site": f"site:{i}",
                "training_budget": (
                    None if adapter == "vision_adp" else
                    {"complete": True, "optimizer_steps": 6}
                ),
                "validation_metrics": copy.deepcopy(metric),
                "test_metrics": {
                    **metric, "accuracy": 0.55 if method == "ADP-W-RR" else 0.5,
                },
            })
    return {
        "status": "completed", "failed_jobs": 0,
        "selected_worker_jobs": len(rows), "completed_jobs": len(rows),
        "scientific_scope": "one-seed tiny official-image smoke; not a benchmark",
        "format": "simple-adp-official-fashion-visual-controls-v1",
        "dataset": "fashion_mnist", "rows": rows,
    }


def invoke(path):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--matrix", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=45, check=False,
    )


def test_cli_complete_19_adapter_single_seed_matrix_emits_only_descriptive_data(tmp_path):
    source = tmp_path / "original.json"
    matrix = data_records()
    source.write_text(json.dumps(matrix), encoding="utf-8")
    result = invoke(source)
    assert result.returncode == 0, result.stderr
    out = tmp_path / "original_paired_metrics.json"
    assert out.is_file()
    analysis = json.loads(out.read_text(encoding="utf-8"))
    assert analysis["adapter_count"] == 19
    assert analysis["paired_worker_count"] == 38
    assert len(analysis["adapters"]) == 19
    assert analysis["scientific_performance_conclusion"] is None
    assert analysis["confidence_intervals_computed"] is False
    assert analysis["source_scientific_scope"] == matrix["scientific_scope"]
    assert analysis["adapters"][0]["observed_optimizer_step_pair_status"] == "unknown-optimizer-steps"
    assert analysis["adapters"][1]["observed_optimizer_step_pair_status"] == "all-equal-optimizer-steps"
    assert analysis["adapters"][1]["metrics_by_split"]["test_metrics"]["accuracy"][
        "mean_improvement"
    ] == pytest.approx(0.05)
    assert json.loads(source.read_text(encoding="utf-8")) == matrix


@pytest.mark.parametrize("mutate", [
    lambda obj: obj.update(status="incomplete"),
    lambda obj: obj.update(failed_jobs=1),
    lambda obj: obj["rows"].pop(),
    lambda obj: obj.update(model_seed_count=2),
    lambda obj: obj.update(selected_worker_jobs=1),
    lambda obj: obj.update(scientific_scope=""),
    lambda obj: obj["rows"][0].update(job_id=""),
    lambda obj: obj["rows"][0]["test_metrics"].update(log_loss=float("nan")),
])
def test_cli_rejects_partial_or_unsigned_19_adapter_input_without_writing_output(
    tmp_path, mutate,
):
    obj = data_records()
    mutate(obj)
    source = tmp_path / "matrix.json"
    source.write_text(json.dumps(obj), encoding="utf-8")
    completed = invoke(source)
    assert completed.returncode != 0
    assert not (tmp_path / "matrix_paired_metrics.json").exists()
