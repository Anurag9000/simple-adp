import numpy as np
import pytest

from simple_adp.advanced_tasks import (
    EventDetectionTask,
    ForecastingTask,
    GroupedRankingTask,
    SequenceLabelingTask,
    SupervisedAnomalyTask,
    SupervisedPreferenceTask,
)
from simple_adp.config import SearchSettings
from simple_adp.runner.datasets import PreparedTaskData
from simple_adp.runner.models import SimpleADPModelAdapter
from simple_adp.runner.route import ModelRoute
from simple_adp.runner.plan import SplitSpec, _sha_indices


def _split(train, valid, test):
    return SplitSpec(0, 0, 0, tuple(train), tuple(valid), tuple(test), _sha_indices(train, valid, test))


def _run(task, y, *, context=None, train=None, valid=None, test=None, family="toy"):
    n = len(y)
    x = np.random.default_rng(17).normal(size=(n, 5)).astype(np.float32)
    train = list(range(0, n - 4)) if train is None else train
    valid = list(range(n - 4, n - 2)) if valid is None else valid
    test = list(range(n - 2, n)) if test is None else test
    data = PreparedTaskData(
        dataset_id="fixture", name="fixture", target_name="target", X=x, y=y,
        feature_schema=None, content_sha256="fixture-hash", dataset_metadata={},
        task_kind=family, task_adapter=task, context=context,
        task_family=family, task_variant="smoke", split_protocol="fixed-fixture",
    )
    result = SimpleADPModelAdapter().fit_predict(
        ModelRoute("MLP-Fixed", fixed_grid=((4,),)), data=data,
        split=_split(train, valid, test), seed=9,
        settings=SearchSettings(init_width=4, init_depth=1, max_epochs=1,
                                batch_size=len(train), max_width=4, max_depth=1,
                                max_neurons=64, patience_es=2, device="cpu"),
    )
    assert result["history_entries"] >= 1
    assert result["task_family"] == family
    assert result["test_metrics"]
    return result


def test_one_epoch_full_training_partition_cpu_smokes_cover_advanced_task_families():
    x = np.random.default_rng(18).normal(size=(12, 5)).astype(np.float32)
    _run(ForecastingTask(), np.linspace(0, 1, 12, dtype=np.float32), family="forecasting")
    sequence_y = np.tile(np.array([[0, 1, 0, 1]], dtype=np.int64), (12, 1))
    _run(SequenceLabelingTask(), sequence_y, family="sequence_labeling")
    _run(EventDetectionTask(), np.array([0, 1] * 6), family="event_detection")
    groups = np.repeat(np.arange(4), 6)
    ratings = np.tile(np.array([5, 4, 3, 2, 5, 1], dtype=np.float32), 4)
    ranking = _run(
        GroupedRankingTask(), ratings, context={"group_id": groups},
        train=list(range(0, 12)), valid=list(range(12, 18)), test=list(range(18, 24)),
        family="ranking",
    )
    assert ranking["test_metrics"]["ndcg@10"] is not None
    labels = np.array([0, 1] * 6)
    _run(SupervisedAnomalyTask(), labels, family="anomaly")
    _run(SupervisedPreferenceTask(), np.where(labels == 1, 5, 2), family="preference")


def test_every_advanced_task_family_gets_the_same_complete_model_route_matrix(tmp_path):
    from simple_adp.advanced_tasks import EventDetectionTask
    from simple_adp.runner.datasets import DatasetTaskRef
    from simple_adp.runner.config import RunnerSettings
    from simple_adp.runner.models import SimpleADPModelAdapter
    from simple_adp.runner.plan import SuitePlan
    from simple_adp.runner.scheduler import SuiteCoordinator
    from simple_adp.registry import MODEL_REGISTRY

    rows = 12
    features = np.arange(rows * 5, dtype=np.float32).reshape(rows, 5)
    group_ids = np.repeat(np.arange(6), 2)
    task_specs = {
        "forecasting": (ForecastingTask(), np.linspace(0, 1, rows, dtype=np.float32), None),
        "sequence": (SequenceLabelingTask(), np.tile([[0, 1], [1, 0]], (6, 1)), None),
        "events": (EventDetectionTask(), np.tile([0, 1], 6), None),
        "ranking": (GroupedRankingTask(), np.tile([5, 2], 6), {"group_id": group_ids}),
        "anomaly": (SupervisedAnomalyTask(), np.tile([0, 1], 6), None),
        "preference": (SupervisedPreferenceTask(), np.tile([2, 5], 6), None),
    }

    class FixtureProvider:
        name = "task_fixtures"

        @property
        def suites(self):
            return tuple(task_specs)

        def list_tasks(self, suite):
            if suite not in task_specs:
                raise ValueError(suite)
            return [DatasetTaskRef(self.name, suite, suite)]

        def load_task(self, ref):
            task, target, context = task_specs[ref.suite]
            return PreparedTaskData(
                dataset_id=ref.suite, name=ref.suite, target_name="target",
                X=features, y=np.asarray(target), feature_schema=None,
                content_sha256=f"fixture-{ref.suite}", dataset_metadata={},
                task_kind="regression" if ref.suite == "forecasting" else "classification",
                task_adapter=task, context=context, task_family=ref.suite,
                task_variant=ref.suite, split_protocol="fixed-fixture",
            )

        def list_splits(self, ref, data, settings):
            train, validation, test = tuple(range(8)), tuple(range(8, 10)), tuple(range(10, 12))
            return [_split(train, validation, test)]

    settings = RunnerSettings(
        cpu_only=True, run_hardness_screen=False, model_seeds=(0,),
        generic_profiles=("faithful", "corrected"), tasks=("all",),
        scheduler="pressure_aware",
        include_vanilla=True, width_only_width=1, width_only_depths=(1,),
        width_to_depth_width=1, width_to_depth_depth=1,
        depth_only_widths=(1,), fold_limit=1, dry_run=True,
        results_dir=str(tmp_path / "results"), run_root=str(tmp_path / "runs"),
    )
    plan = SuitePlan.build(
        settings=settings, results_dir=tmp_path / "results", run_root=tmp_path / "runs",
        dataset_providers=(FixtureProvider(),),
        dataset_suites=tuple(("task_fixtures", suite) for suite in task_specs),
        model_adapters=(SimpleADPModelAdapter(),),
    )
    routes_by_task = {
        suite: {run.route.route_label for run in plan.runs if run.task_ref.task_id == suite}
        for suite in task_specs
    }
    expected = next(iter(routes_by_task.values()))
    assert len(expected) > len(MODEL_REGISTRY)
    assert all(routes == expected for routes in routes_by_task.values())
    for route_set in routes_by_task.values():
        assert {route.split("@", 1)[0] for route in route_set} >= set(MODEL_REGISTRY)
        assert len({route for route in route_set if route.startswith("vanilla-d")}) == 50
        assert any("@faithful@" in route for route in route_set)
        assert any("@corrected@" in route for route in route_set)

    coordinator = SuiteCoordinator(
        plan, settings, run_root=tmp_path / "runs", results_dir=tmp_path / "results",
        repository_root=tmp_path,
    )
    summary = coordinator.run()
    dry_run_jobs = __import__("json").loads((tmp_path / "runs" / "dry_run_jobs.json").read_text())
    assert summary["dry_run"] is True
    assert summary["job_count"] == len(plan.runs) == len(dry_run_jobs)
    assert {job["task_ref"]["task_id"] for job in dry_run_jobs} == set(task_specs)
    assert plan.manifest["runner_settings"]["scheduler"] == "pressure_aware"
