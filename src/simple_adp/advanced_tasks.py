"""Task adapters for fixed-window temporal, grouped-ranking, and anomaly problems.

All adapters keep the estimator a fully connected MLP. Temporal structure is
represented by fixed-size rows and optional aligned context arrays.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from .tasks import ClassificationTask, RegressionTask, TaskOutputSpec


class ForecastingTask(RegressionTask):
    """Fixed-horizon numeric forecasting; one output is produced per horizon."""

    @property
    def output_spec(self) -> TaskOutputSpec:
        return TaskOutputSpec(self.output_dim, "fixed_horizon_forecast")


class SequenceLabelingTask(ClassificationTask):
    """Per-step labels emitted as a flattened (steps × classes) MLP output."""

    def __init__(self) -> None:
        super().__init__()
        self.sequence_length_: int | None = None

    def fit_targets(self, targets: Any) -> None:
        values = np.asarray(targets)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("sequence labels must have non-empty shape (examples, steps)")
        self.sequence_length_ = int(values.shape[1])
        super().fit_targets(values.reshape(-1))

    @property
    def output_dim(self) -> int:
        if self.sequence_length_ is None:
            raise RuntimeError("fit_targets must be called before output_dim")
        return self.sequence_length_ * len(self.classes_)

    @property
    def output_spec(self) -> TaskOutputSpec:
        return TaskOutputSpec(self.output_dim, f"per_step_classification:{self.sequence_length_}")

    @property
    def primary_metric(self) -> str:
        return "macro_f1"

    @property
    def metric_directions(self) -> dict[str, str]:
        return {**super().metric_directions, "boundary_precision": "max",
                "boundary_recall": "max", "boundary_f1": "max"}

    def encode_targets(self, targets: Any, *, device: torch.device) -> torch.Tensor:
        values = np.asarray(targets)
        if values.ndim != 2 or values.shape[1] != self.sequence_length_:
            raise ValueError("sequence target length differs from fitted sequence length")
        return super().encode_targets(values.reshape(-1), device=device).reshape(values.shape)

    def loss(self, outputs, targets, context=None):
        if outputs.ndim != 2 or outputs.shape[1] != self.output_dim:
            raise ValueError("sequence logits do not match (steps × classes) output specification")
        logits = outputs.reshape(-1, self.sequence_length_, len(self.classes_))
        return nn.functional.cross_entropy(logits.reshape(-1, len(self.classes_)), targets.reshape(-1))

    def validation_score(self, outputs, targets, context=None) -> float:
        return float(self.loss(outputs, targets).detach().cpu())

    def predictions(self, outputs) -> np.ndarray:
        logits = outputs.reshape(-1, self.sequence_length_, len(self.classes_))
        indexes = logits.argmax(dim=-1).detach().cpu().numpy()
        return np.asarray(self.classes_, dtype=object)[indexes]

    def predict_proba(self, outputs) -> np.ndarray:
        logits = outputs.reshape(-1, self.sequence_length_, len(self.classes_))
        return torch.softmax(logits, dim=-1).detach().cpu().numpy()

    def evaluation_metrics(self, targets, predictions, probabilities=None, context=None):
        probs = None
        if probabilities is not None:
            probs = np.asarray(probabilities).reshape(-1, len(self.classes_))
        target_array = np.asarray(targets)
        prediction_array = np.asarray(predictions)
        if target_array.ndim != 2 or prediction_array.shape != target_array.shape:
            raise ValueError("sequence predictions and targets must have matching (examples, steps) shapes")
        truth = self._encode_flat(target_array)
        predicted = self._encode_flat(prediction_array)
        if truth.size == 0:
            raise ValueError("sequence evaluation requires at least one labeled step")
        class_count = len(self.classes_)
        support = np.bincount(truth, minlength=class_count)
        predicted_count = np.bincount(predicted, minlength=class_count)
        true_positive = np.bincount(truth[truth == predicted], minlength=class_count)
        recall_per_class = np.divide(true_positive, support, out=np.zeros(class_count, dtype=np.float64), where=support > 0)
        denominator = 2 * true_positive + (predicted_count - true_positive) + (support - true_positive)
        f1_per_class = np.divide(2 * true_positive, denominator, out=np.zeros(class_count, dtype=np.float64), where=denominator > 0)
        active = (support + predicted_count) > 0
        metrics = {
            "accuracy": float(np.mean(truth == predicted)),
            "balanced_accuracy": float(np.mean(recall_per_class[support > 0])),
            "macro_f1": float(np.mean(f1_per_class[active])),
        }
        if probs is not None:
            if probs.shape != (len(truth), class_count) or not np.isfinite(probs).all():
                raise ValueError("sequence probabilities do not match flattened labels and classes")
            probs = probs.astype(np.float64)
            probs = probs / probs.sum(axis=1, keepdims=True)
            probs = np.clip(probs, np.finfo(np.float64).eps, 1 - np.finfo(np.float64).eps)
            metrics["log_loss"] = float(-np.mean(np.log(probs[np.arange(len(truth)), truth])))
        context_rows = context or {}
        has_source_positions = all(key in context_rows for key in ("subject_id", "experiment_id", "window_start"))
        if has_source_positions:
            subjects = np.asarray(context_rows["subject_id"]).reshape(-1)
            experiments = np.asarray(context_rows["experiment_id"]).reshape(-1)
            starts = np.asarray(context_rows["window_start"], dtype=np.int64).reshape(-1)
            if any(len(values) != len(target_array) for values in (subjects, experiments, starts)):
                raise ValueError("sequence source-position context must align with sequence examples")
            true_boundaries: set[tuple[Any, Any, int]] = set()
            predicted_boundaries: set[tuple[Any, Any, int]] = set()
            for row in range(len(target_array)):
                owner = (subjects[row].item() if hasattr(subjects[row], "item") else subjects[row],
                         experiments[row].item() if hasattr(experiments[row], "item") else experiments[row])
                true_boundaries.update((*owner, int(starts[row] + index + 1)) for index in np.flatnonzero(target_array[row, 1:] != target_array[row, :-1]))
                predicted_boundaries.update((*owner, int(starts[row] + index + 1)) for index in np.flatnonzero(prediction_array[row, 1:] != prediction_array[row, :-1]))
            # Count transitions exactly between windows only when both blocks
            # are adjacent in the same experiment. A missing/unlabeled block
            # creates a source-position gap and must not be bridged.
            order = sorted(range(len(starts)), key=lambda row: (subjects[row], experiments[row], starts[row]))
            for previous, current in zip(order, order[1:]):
                same_owner = subjects[previous] == subjects[current] and experiments[previous] == experiments[current]
                if not same_owner or starts[current] != starts[previous] + target_array.shape[1]:
                    continue
                owner = (subjects[current].item() if hasattr(subjects[current], "item") else subjects[current],
                         experiments[current].item() if hasattr(experiments[current], "item") else experiments[current])
                boundary_position = int(starts[current])
                if target_array[previous, -1] != target_array[current, 0]:
                    true_boundaries.add((*owner, boundary_position))
                if prediction_array[previous, -1] != prediction_array[current, 0]:
                    predicted_boundaries.add((*owner, boundary_position))
        else:
            true_boundaries = set()
            predicted_boundaries = set()
            for row in range(len(target_array)):
                true_boundaries.update((row, int(index + 1)) for index in np.flatnonzero(target_array[row, 1:] != target_array[row, :-1]))
                predicted_boundaries.update((row, int(index + 1)) for index in np.flatnonzero(prediction_array[row, 1:] != prediction_array[row, :-1]))
        hits = len(true_boundaries & predicted_boundaries)
        precision = hits / len(predicted_boundaries) if predicted_boundaries else 0.0
        recall = hits / len(true_boundaries) if true_boundaries else 0.0
        metrics.update({
            "boundary_precision": precision,
            "boundary_recall": recall,
            "boundary_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        })
        return metrics

    def _encode_flat(self, values: np.ndarray) -> np.ndarray:
        try:
            return np.asarray([self._class_to_index[self._key(value)] for value in values.reshape(-1)], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f"sequence class {exc.args[0]!r} was absent from training data") from exc

    def state_dict(self) -> dict[str, object]:
        state = super().state_dict()
        state["sequence_length"] = self.sequence_length_
        return state

    def load_state_dict(self, state: dict[str, object]) -> None:
        super().load_state_dict(state)
        length = state.get("sequence_length")
        self.sequence_length_ = int(length) if length is not None else None


class EventDetectionTask(ClassificationTask):
    """Supervised classification of labeled events or event-centered windows."""

    @property
    def output_spec(self) -> TaskOutputSpec:
        return TaskOutputSpec(self.output_dim, "supervised_event_classification")

    @property
    def primary_metric(self) -> str:
        return "event_f1" if self.output_dim == 2 else "macro_f1"

    @property
    def metric_directions(self) -> dict[str, str]:
        return {"accuracy": "max", "balanced_accuracy": "max", "macro_f1": "max",
                "log_loss": "min", "event_precision": "max", "event_recall": "max", "event_f1": "max"}

    def evaluation_metrics(self, targets, predictions, probabilities=None, context=None):
        metrics = super().evaluation_metrics(targets, predictions, probabilities, context)
        raw = context or {}
        event_ids = raw.get("event_id")
        if event_ids is None:
            return metrics
        event_ids = np.asarray(event_ids).reshape(-1)
        truth = np.asarray(targets).reshape(-1)
        predicted = np.asarray(predictions).reshape(-1)
        if event_ids.shape != truth.shape or predicted.shape != truth.shape:
            raise ValueError("event_id context must align with event predictions")
        try:
            valid_events = event_ids >= 0
        except TypeError as exc:
            raise ValueError("event_id values must be numeric, with negative values reserved for non-events") from exc
        true_ids = set(event_ids[valid_events].tolist())
        # Predicted event alarms are contiguous positive windows within a
        # source series; a series change or a gap in window indexes starts a
        # new alarm. Match alarms to true intervals one-to-one so precision
        # cannot exceed one when a single alarm spans multiple intervals.
        series = np.asarray(raw.get("series_id", np.zeros(len(truth), dtype=np.int64))).reshape(-1)
        window_index = np.asarray(raw.get("window_index", np.arange(len(truth)))).reshape(-1)
        if series.shape != truth.shape or window_index.shape != truth.shape:
            raise ValueError("series_id and window_index context must align with event predictions")
        order = np.lexsort((window_index, series))
        predicted_alarms: list[list[int]] = []
        active_alarm: list[int] = []
        previous_series = None
        previous_window = None
        for index in order:
            current_series = series[index]
            current_window = window_index[index]
            if (current_series != previous_series or
                    (previous_window is not None and current_window != previous_window + 1) or
                    predicted[index] != 1):
                if active_alarm:
                    predicted_alarms.append(active_alarm)
                    active_alarm = []
            if predicted[index] == 1:
                active_alarm.append(int(index))
            previous_series = current_series
            previous_window = current_window
        if active_alarm:
            predicted_alarms.append(active_alarm)

        unmatched = set(true_ids)
        hits = 0
        for alarm in predicted_alarms:
            overlapping = sorted(set(event_ids[alarm][event_ids[alarm] >= 0].tolist()))
            match = next((event_id for event_id in overlapping if event_id in unmatched), None)
            if match is not None:
                unmatched.remove(match)
                hits += 1
        precision = hits / len(predicted_alarms) if predicted_alarms else 0.0
        recall = hits / len(true_ids) if true_ids else 0.0
        metrics.update({"event_precision": precision, "event_recall": recall,
                        "event_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0})
        return metrics


class GroupedRankingTask(RegressionTask):
    """Pairwise logistic ranking with whole-group validation metrics.

    Targets are graded relevance values. A positive is relevance >= 4 and a
    negative is relevance <= 3. For each positive, one negative is selected
    deterministically by positive row position modulo the number of negatives.
    """

    def __init__(self, relevant_threshold: float = 4.0) -> None:
        super().__init__()
        self.relevant_threshold = float(relevant_threshold)

    @property
    def output_dim(self) -> int:
        return 1

    @property
    def output_spec(self) -> TaskOutputSpec:
        return TaskOutputSpec(1, "grouped_pairwise_ranking_score")

    @property
    def batch_group_key(self) -> str:
        return "group_id"

    @property
    def primary_metric(self) -> str:
        return "ndcg@10"

    @property
    def metric_directions(self) -> dict[str, str]:
        return {"ndcg@5": "max", "ndcg@10": "max", "recall@5": "max", "recall@10": "max"}

    def fit_targets(self, targets: Any) -> None:
        values = np.asarray(targets, dtype=np.float32)
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("ranking relevance targets must be a finite non-empty nonnegative vector")
        self._output_dim = 1

    def encode_targets(self, targets: Any, *, device: torch.device) -> torch.Tensor:
        values = np.asarray(targets, dtype=np.float32).reshape(-1, 1)
        if len(values) == 0 or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("ranking relevance targets must be finite, non-empty, and nonnegative")
        return torch.as_tensor(values, dtype=torch.float32, device=device)

    @staticmethod
    def _groups(context: dict[str, torch.Tensor] | None, count: int, device) -> torch.Tensor:
        if context is None or "group_id" not in context:
            raise ValueError("grouped ranking requires context['group_id']")
        groups = torch.as_tensor(context["group_id"], device=device, dtype=torch.long).reshape(-1)
        if groups.numel() != count:
            raise ValueError("ranking group IDs must align with candidate rows")
        return groups

    def loss(self, outputs, targets, context=None):
        scores = outputs.reshape(-1)
        relevance = targets.reshape(-1)
        groups = self._groups(context, len(scores), scores.device)
        losses = []
        for group in torch.unique(groups, sorted=True):
            indexes = torch.nonzero(groups == group, as_tuple=False).reshape(-1)
            positive = indexes[relevance[indexes] >= self.relevant_threshold]
            negative = indexes[relevance[indexes] < self.relevant_threshold]
            if positive.numel() == 0 or negative.numel() == 0:
                continue
            selection = torch.arange(positive.numel(), device=scores.device) % negative.numel()
            losses.append(nn.functional.softplus(scores[negative[selection]] - scores[positive]).mean())
        if not losses:
            raise ValueError("ranking batch must contain at least one group with positive and negative candidates")
        return torch.stack(losses).mean()

    def validation_score(self, outputs, targets, context=None) -> float:
        # Architecture selection follows the task's differentiable objective.
        # Ranking metrics are useful for reporting, but substituting NDCG here
        # would make the search optimize a different quantity than the plan
        # specifies (pairwise validation loss).
        with torch.no_grad():
            return float(self.loss(outputs, targets, context=context).detach().cpu())

    def predictions(self, outputs) -> np.ndarray:
        return outputs.detach().cpu().numpy().reshape(-1)

    def evaluation_metrics(self, targets, predictions, probabilities=None, context=None):
        relevance = np.asarray(targets, dtype=np.float64).reshape(-1)
        scores = np.asarray(predictions, dtype=np.float64).reshape(-1)
        if relevance.shape != scores.shape or not np.isfinite(relevance).all() or not np.isfinite(scores).all():
            raise ValueError("ranking targets and scores must be aligned finite vectors")
        raw_context = context or {}
        group_values = raw_context.get("group_id")
        if group_values is None:
            raise ValueError("grouped ranking evaluation requires context['group_id']")
        groups = np.asarray(group_values).reshape(-1)
        if groups.shape != relevance.shape:
            raise ValueError("ranking group IDs must align with candidate rows")
        ndcg: dict[int, list[float]] = {5: [], 10: []}
        recall: dict[int, list[float]] = {5: [], 10: []}
        for group in np.unique(groups):
            idx = np.flatnonzero(groups == group)
            order = idx[np.argsort(-scores[idx], kind="stable")]
            ideal = idx[np.argsort(-relevance[idx], kind="stable")]
            gains = np.exp2(relevance) - 1.0
            for k in (5, 10):
                discounts = np.log2(np.arange(2, min(k, len(idx)) + 2))
                dcg = np.sum(gains[order[:k]] / discounts) if len(discounts) else 0.0
                idcg = np.sum(gains[ideal[:k]] / discounts) if len(discounts) else 0.0
                ndcg[k].append(float(dcg / idcg) if idcg > 0 else 0.0)
                positives = int(np.sum(relevance[idx] >= self.relevant_threshold))
                hits = int(np.sum(relevance[order[:k]] >= self.relevant_threshold))
                recall[k].append(float(hits / positives) if positives else 0.0)
        if not ndcg[10]:
            raise ValueError("ranking evaluation contains no groups")
        return {
            "ndcg@5": float(np.mean(ndcg[5])), "ndcg@10": float(np.mean(ndcg[10])),
            "recall@5": float(np.mean(recall[5])), "recall@10": float(np.mean(recall[10])),
        }

    def state_dict(self) -> dict[str, object]:
        return {"type": type(self).__qualname__, "relevant_threshold": self.relevant_threshold}

    def load_state_dict(self, state: dict[str, object]) -> None:
        super().load_state_dict(state)
        threshold = float(state.get("relevant_threshold", self.relevant_threshold))
        if threshold != self.relevant_threshold:
            raise ValueError("checkpoint ranking relevance threshold differs")


class SupervisedAnomalyTask(ClassificationTask):
    """Binary anomaly classifier with training-derived positive weighting."""

    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold_ = float(threshold)
        self.positive_weight_: float | None = None

    @property
    def primary_metric(self) -> str:
        return "pr_auc"

    @property
    def metric_directions(self) -> dict[str, str]:
        return {"accuracy": "max", "precision": "max", "recall": "max", "f1": "max", "pr_auc": "max", "roc_auc": "max"}

    def fit_targets(self, targets: Any) -> None:
        values = np.asarray(targets)
        if values.ndim != 1 or set(np.unique(values).tolist()) != {0, 1}:
            raise ValueError("supervised anomaly targets must contain both binary labels 0 and 1")
        self.classes_ = [0, 1]
        self._class_to_index = {self._key(0): 0, self._key(1): 1}
        positives = int(np.sum(values == 1))
        negatives = int(np.sum(values == 0))
        self.positive_weight_ = negatives / positives

    @property
    def output_dim(self) -> int:
        if self.positive_weight_ is None:
            raise RuntimeError("fit_targets must be called before output_dim")
        return 1

    @property
    def output_spec(self) -> TaskOutputSpec:
        return TaskOutputSpec(1, "binary_anomaly_logit")

    def encode_targets(self, targets: Any, *, device: torch.device) -> torch.Tensor:
        values = np.asarray(targets, dtype=np.float32).reshape(-1, 1)
        if not np.isin(values, [0.0, 1.0]).all():
            raise ValueError("anomaly targets must be binary 0/1")
        return torch.as_tensor(values, dtype=torch.float32, device=device)

    def loss(self, outputs, targets, context=None):
        if self.positive_weight_ is None:
            raise RuntimeError("fit_targets must run before anomaly loss")
        weight = outputs.new_tensor([self.positive_weight_])
        return nn.functional.binary_cross_entropy_with_logits(outputs.reshape(-1, 1), targets.reshape(-1, 1), pos_weight=weight)

    def validation_score(self, outputs, targets, context=None) -> float:
        return float(self.loss(outputs, targets).detach().cpu())

    def predict_proba(self, outputs) -> np.ndarray:
        positive = torch.sigmoid(outputs.reshape(-1)).detach().cpu().numpy()
        return np.column_stack((1.0 - positive, positive))

    def predictions(self, outputs) -> np.ndarray:
        probs = self.predict_proba(outputs)[:, 1]
        return (probs >= self.threshold_).astype(np.int64)

    @staticmethod
    def select_threshold(targets: Any, positive_probabilities: Any) -> float:
        truth = np.asarray(targets, dtype=np.int64).reshape(-1)
        probabilities = np.asarray(positive_probabilities, dtype=np.float64).reshape(-1)
        if truth.shape != probabilities.shape or not np.isfinite(probabilities).all():
            raise ValueError("threshold selection inputs must be aligned and finite")
        candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probabilities)))
        best = (-1.0, 0.5)
        for threshold in candidates:
            pred = probabilities >= threshold
            tp = np.sum((truth == 1) & pred)
            fp = np.sum((truth == 0) & pred)
            fn = np.sum((truth == 1) & ~pred)
            f1 = float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0
            if f1 > best[0] or (f1 == best[0] and threshold > best[1]):
                best = (f1, float(threshold))
        return best[1]

    @staticmethod
    def _auc(truth: np.ndarray, scores: np.ndarray) -> float | None:
        positives = truth == 1
        negatives = truth == 0
        if not positives.any() or not negatives.any():
            return None
        order = np.argsort(scores, kind="stable")
        sorted_scores = scores[order]
        ranks = np.empty(len(scores), dtype=np.float64)
        start = 0
        while start < len(scores):
            end = start + 1
            while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
                end += 1
            ranks[order[start:end]] = (start + 1 + end) / 2.0
            start = end
        return float((ranks[positives].sum() - positives.sum() * (positives.sum() + 1) / 2)
                     / (positives.sum() * negatives.sum()))

    def evaluation_metrics(self, targets, predictions, probabilities=None, context=None):
        truth = np.asarray(targets, dtype=np.int64).reshape(-1)
        pred = np.asarray(predictions, dtype=np.int64).reshape(-1)
        if truth.shape != pred.shape or len(truth) == 0:
            raise ValueError("anomaly predictions and labels must align and be non-empty")
        positive = np.asarray(probabilities)[:, 1] if probabilities is not None else pred.astype(float)
        tp = int(np.sum((truth == 1) & (pred == 1)))
        fp = int(np.sum((truth == 0) & (pred == 1)))
        fn = int(np.sum((truth == 1) & (pred == 0)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        positives = int(np.sum(truth == 1))
        average_precision = None
        if positives:
            # Average precision at score thresholds, grouping ties so input
            # ordering among equal scores cannot change the metric.
            order = np.argsort(-positive, kind="stable")
            sorted_scores, sorted_truth = positive[order], truth[order]
            cumulative_true = 0
            previous_recall = 0.0
            average_precision = 0.0
            cursor = 0
            while cursor < len(sorted_truth):
                end = cursor + 1
                while end < len(sorted_truth) and sorted_scores[end] == sorted_scores[cursor]:
                    end += 1
                cumulative_true += int(np.sum(sorted_truth[cursor:end] == 1))
                recall_at = cumulative_true / positives
                precision_at = cumulative_true / end
                average_precision += (recall_at - previous_recall) * precision_at
                previous_recall = recall_at
                cursor = end
        return {
            "accuracy": float(np.mean(truth == pred)), "precision": float(precision),
            "recall": float(recall), "f1": float(f1), "pr_auc": average_precision,
            "roc_auc": self._auc(truth, positive),
        }

    def state_dict(self) -> dict[str, object]:
        return {"type": type(self).__qualname__, "threshold": self.threshold_, "positive_weight": self.positive_weight_}

    def load_state_dict(self, state: dict[str, object]) -> None:
        super().load_state_dict({"type": type(self).__qualname__, "classes": [0, 1]})
        self.threshold_ = float(state.get("threshold", 0.5))
        weight = state.get("positive_weight")
        self.positive_weight_ = float(weight) if weight is not None else None


class SupervisedPreferenceTask(SupervisedAnomalyTask):
    """Binary rating-derived preference task, explicitly distinct from clicks."""

    def __init__(self, relevant_threshold: float = 4.0, threshold: float = 0.5) -> None:
        super().__init__(threshold=threshold)
        if not np.isfinite(relevant_threshold):
            raise ValueError("relevant_threshold must be finite")
        self.relevant_threshold = float(relevant_threshold)

    def fit_targets(self, targets: Any) -> None:
        ratings = np.asarray(targets, dtype=np.float32).reshape(-1)
        if not np.isfinite(ratings).all():
            raise ValueError("preference ratings must be finite")
        binary = (ratings >= self.relevant_threshold).astype(np.int64)
        super().fit_targets(binary)

    def encode_targets(self, targets: Any, *, device: torch.device) -> torch.Tensor:
        ratings = np.asarray(targets, dtype=np.float32).reshape(-1)
        binary = (ratings >= self.relevant_threshold).astype(np.float32)
        return super().encode_targets(binary, device=device)

    @staticmethod
    def _binarize(targets: Any, threshold: float) -> np.ndarray:
        ratings = np.asarray(targets, dtype=np.float32).reshape(-1)
        if not np.isfinite(ratings).all():
            raise ValueError("preference ratings must be finite")
        return (ratings >= threshold).astype(np.int64)

    def select_threshold(self, targets: Any, positive_probabilities: Any) -> float:
        binary = self._binarize(targets, self.relevant_threshold)
        return SupervisedAnomalyTask.select_threshold(binary, positive_probabilities)

    def evaluation_metrics(self, targets, predictions, probabilities=None, context=None):
        binary = self._binarize(targets, self.relevant_threshold)
        return super().evaluation_metrics(binary, predictions, probabilities, context)

    def predictions(self, outputs: torch.Tensor) -> np.ndarray:
        return super().predictions(outputs)

    @property
    def output_spec(self) -> TaskOutputSpec:
        return TaskOutputSpec(1, f"rating_derived_preference_threshold_{self.relevant_threshold:g}")

    def state_dict(self) -> dict[str, object]:
        return {**super().state_dict(), "relevant_threshold": self.relevant_threshold}

    def load_state_dict(self, state: dict[str, object]) -> None:
        super().load_state_dict(state)
        threshold = float(state.get("relevant_threshold", self.relevant_threshold))
        if threshold != self.relevant_threshold:
            raise ValueError("checkpoint preference threshold differs")
