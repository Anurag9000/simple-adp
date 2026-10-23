import json

import pytest

from simple_adp.runner.datasets import DatasetTaskRef
from simple_adp.runner.plan import RunIdentity, RunSpec
from simple_adp.runner.route import ModelRoute
from simple_adp.runner.screening import build_screening_reports


@pytest.mark.parametrize(
    "family,task_kind,metric,direction,shallow,deep,expected_gain",
    [
        ("grouped_ranking", "ranking", "ndcg@10", "max", 0.5, 0.8, 0.3),
        ("forecasting", "regression", "rmse", "min", 1.2, 0.9, 0.3),
    ],
)
def test_screening_uses_new_task_primary_metric_and_declared_direction(
    tmp_path, family, task_kind, metric, direction, shallow, deep, expected_gain,
):
    specs = []
    for depth, score in ((1, shallow), (2, deep)):
        label = f"d{depth}"
        identity = RunIdentity("fixture", "suite", "task", 0, 0, 0, 1, label, 1, "split", "signature")
        result_dir = tmp_path / label
        result_dir.mkdir()
        (result_dir / "result.json").write_text(json.dumps({
            "status": "completed", "screening_only": True, "task_kind": task_kind,
            "task_family": family, "primary_metric": metric,
            "metric_directions": {metric: direction},
            "validation_metrics": {metric: score},
            "validation_selection_score": score,
        }), encoding="utf-8")
        specs.append(RunSpec(
            identity, DatasetTaskRef("fixture", "suite", "task"), "simple_adp",
            ModelRoute("MLP-Fixed", fixed_grid=((4,) * depth,), label=label,
                       screening_only=True, screening_seed=1, screening_architecture=(depth, 4)),
            str(tmp_path / "split.npz"), str(result_dir), {}, ("unused",),
        ))
    output = tmp_path / "out"
    build_screening_reports(
        specs, output, classification_ceiling=1.0, regression_ceiling_r2=0.999,
        min_complexity_gain=0.02, min_hard_fold_fraction=0.8, screen_seed_count=1,
    )
    fold = json.loads((output / "screening-folds.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert fold["validation_complexity_gain"] == pytest.approx(expected_gain)
    assert fold["architectures"][0]["validation_score_name"] == metric
    assert fold["architectures"][0]["validation_score_direction"] == direction
