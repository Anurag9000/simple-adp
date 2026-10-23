import numpy as np
import pytest
import torch
import copy

from simple_adp.advanced_tasks import (
    EventDetectionTask,
    ForecastingTask,
    GroupedRankingTask,
    SequenceLabelingTask,
    SupervisedAnomalyTask,
    SupervisedPreferenceTask,
)
from simple_adp.task_preparation import (
    chronological_split,
    beat_centered_windows,
    event_window_labels,
    grouped_chronological_split,
    sequence_windows,
    stratified_split,
    temporal_windows,
)
from simple_adp.config import SearchSettings
from simple_adp.models import TabularMLP
from simple_adp.training import StageTrainer


def test_forecast_windows_have_strict_future_targets_and_no_entity_crossing():
    values = np.arange(30, dtype=np.float32)
    entities = np.repeat([0, 1], 15)
    x, y, endpoints = temporal_windows(values, lookback=4, horizon=2, entity_ids=entities)
    assert x.shape == (20, 4)
    assert np.all(y[:, 0] > x[:, -1])
    assert not any(endpoints[i] == 14 and endpoints[i + 1] == 15 for i in range(len(endpoints) - 1))
    task = ForecastingTask()
    task.fit_targets(y)
    assert task.output_dim == 1


def test_default_forecast_stride_is_selected_per_series_not_after_concatenation():
    values = np.arange(14_000, dtype=np.float32)
    entity_ids = np.repeat([0, 1], 7_000)
    x, _y, _positions = temporal_windows(values, lookback=2, entity_ids=entity_ids)
    assert len(x) == 2 * (7_000 - 2)


def test_sequence_windows_ignore_incomplete_tail_and_keep_subject_ids():
    x = np.arange(22, dtype=np.float32).reshape(11, 2)
    y = np.arange(11) % 2
    window_x, window_y, owners = sequence_windows(x, y, window_size=4, entity_ids=[0] * 5 + [1] * 6)
    assert window_x.shape == (2, 8)
    assert window_y.tolist() == [[0, 1, 0, 1], [1, 0, 1, 0]]
    assert owners.tolist() == [0, 1]
    task = SequenceLabelingTask()
    task.fit_targets(window_y)
    logits = torch.randn(2, task.output_dim)
    target = task.encode_targets(window_y, device=torch.device("cpu"))
    assert task.loss(logits, target).ndim == 0
    assert task.predictions(logits).shape == window_y.shape


def test_sequence_boundaries_bridge_only_adjacent_source_windows():
    task = SequenceLabelingTask()
    truth = np.asarray([[0, 0, 0, 0], [1, 1, 1, 1]])
    task.fit_targets(truth)
    metrics = task.evaluation_metrics(
        truth, truth,
        context={"subject_id": [3, 3], "experiment_id": [2, 2], "window_start": [0, 4]},
    )
    assert metrics["boundary_precision"] == metrics["boundary_recall"] == metrics["boundary_f1"] == 1.0
    gap_metrics = task.evaluation_metrics(
        truth, truth,
        context={"subject_id": [3, 3], "experiment_id": [2, 2], "window_start": [0, 8]},
    )
    assert gap_metrics["boundary_precision"] == gap_metrics["boundary_recall"] == gap_metrics["boundary_f1"] == 0.0


def test_grouped_ranking_loss_and_metrics_are_context_aware():
    task = GroupedRankingTask()
    y = np.array([5, 1, 4, 2], dtype=np.float32)
    task.fit_targets(y)
    targets = task.encode_targets(y, device=torch.device("cpu"))
    context = {"group_id": torch.tensor([10, 10, 20, 20])}
    scores = torch.tensor([[2.0], [0.0], [0.0], [2.0]], requires_grad=True)
    loss = task.loss(scores, targets, context)
    assert loss.item() > 0
    loss.backward()
    metrics = task.evaluation_metrics(y, scores.detach().numpy(), context={"group_id": [10, 10, 20, 20]})
    assert metrics["ndcg@10"] < 1
    assert task.validation_score(scores.detach(), targets, context) == pytest.approx(loss.item())
    assert task.validation_score(scores.detach(), targets, context) != pytest.approx(1 - metrics["ndcg@10"])


def test_event_level_precision_uses_one_to_one_alarm_interval_matching():
    task = EventDetectionTask()
    task.fit_targets(np.array([0, 1]))
    metrics = task.evaluation_metrics(
        targets=np.array([1, 1, 1, 1]),
        predictions=np.array([1, 1, 1, 1]),
        context={
            "event_id": np.array([10, 10, 20, 20]),
            "series_id": np.array([0, 0, 0, 0]),
            "window_index": np.array([0, 1, 2, 3]),
        },
    )
    assert metrics["event_precision"] == pytest.approx(1.0)
    assert metrics["event_recall"] == pytest.approx(0.5)
    assert metrics["event_f1"] == pytest.approx(2 / 3)

    split_alarm_metrics = task.evaluation_metrics(
        targets=np.array([1, 1, 1]),
        predictions=np.array([1, 0, 1]),
        context={"event_id": np.array([7, 7, 7]), "series_id": np.zeros(3), "window_index": np.arange(3)},
    )
    assert split_alarm_metrics["event_precision"] == pytest.approx(0.5)
    assert split_alarm_metrics["event_recall"] == pytest.approx(1.0)


def test_anomaly_weight_is_train_derived_and_preference_labels_are_rating_based():
    anomaly = SupervisedAnomalyTask()
    anomaly.fit_targets(np.array([0, 0, 0, 1]))
    assert anomaly.positive_weight_ == 3
    preference = SupervisedPreferenceTask(relevant_threshold=4)
    ratings = np.array([1, 2, 4, 5])
    preference.fit_targets(ratings)
    encoded = preference.encode_targets(ratings, device=torch.device("cpu"))
    assert encoded[:, 0].tolist() == [0, 0, 1, 1]
    assert preference.evaluation_metrics(ratings, [0, 0, 1, 1])["accuracy"] == 1
    state = preference.state_dict()
    clone = SupervisedPreferenceTask(relevant_threshold=4)
    clone.load_state_dict(state)
    assert clone.positive_weight_ == 1


def test_time_and_group_splits_and_event_intervals():
    train, valid, test = chronological_split(np.arange(20), train_fraction=0.6, validation_fraction=0.2)
    assert train.max() < valid.min() < test.min()
    grouped = grouped_chronological_split([0] * 10 + [1] * 10, list(range(10)) * 2)
    assert [len(part) for part in grouped] == [16, 2, 2]
    bounds, labels = event_window_labels(np.arange(8), [(3, 4)], window_size=3, stride=2)
    assert bounds.tolist() == [[0, 3], [2, 5], [4, 7]]
    assert labels.tolist() == [0, 1, 1]
    partitions = stratified_split(np.repeat([0, 1], 30), seed=3)
    assert all(set(np.unique(np.repeat([0, 1], 30)[part])) == {0, 1} for part in partitions)


def test_aami_beat_windows_group_symbols_and_drop_boundary_beats():
    signal = np.arange(1000, dtype=np.float32)
    windows, labels, positions = beat_centered_windows(signal, [20, 300, 700, 990], ["N", "A", "V", "F"])
    assert windows.shape == (2, 360)
    assert labels.tolist() == ["S", "V"]
    assert positions.tolist() == [300, 700]


def test_grouped_batch_plan_is_saved_and_resumed_without_splitting_users():
    class Interrupted(Exception):
        pass

    task = GroupedRankingTask()
    labels = np.array([5, 1, 4, 2, 5, 3], dtype=np.float32)
    task.fit_targets(labels)
    target = task.encode_targets(labels, device=torch.device("cpu"))
    context = {"group_id": torch.tensor([0, 0, 1, 1, 2, 2])}
    x = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    settings = SearchSettings(init_width=3, init_depth=1, max_epochs=2, batch_size=2,
                              max_width=3, max_depth=1, max_neurons=32,
                              patience_es=5, seed=29, device="cpu")
    captured = {}

    def interrupt_after_first_batch(state):
        if state.get("stage_status") == "active" and state.get("batch_cursor") == 1:
            captured["state"] = copy.deepcopy(state)
            raise Interrupted

    model = TabularMLP(2, [3], 1)
    with pytest.raises(Interrupted):
        StageTrainer(task, settings, checkpoint_callback=interrupt_after_first_batch).train_stage(
            model, x, target, x, target, train_context=context, valid_context=context,
        )
    state = captured["state"]
    assert state["batch_cursor"] == 1
    assert all(len(set(context["group_id"][batch].tolist())) == 1 for batch in state["batch_plan"])
    resumed_model = TabularMLP(2, [3], 1)
    result = StageTrainer(task, settings).train_stage(
        resumed_model, x, target, x, target, resume_state=state,
        train_context=context, valid_context=context,
    )
    assert result.batches >= 2
