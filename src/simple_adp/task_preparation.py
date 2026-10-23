"""Deterministic fixed-window framing and leakage-safe task split utilities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


_AAMI_BEATS = {
    "N": frozenset(("N", "L", "R", "e", "j")),
    "S": frozenset(("A", "a", "J", "S")),
    "V": frozenset(("V", "E")),
    "F": frozenset(("F",)),
    "Q": frozenset(("/", "f", "Q")),
}


def aami_beat_class(symbol: str) -> str:
    """Map a WFDB beat annotation symbol to the documented five AAMI groups."""
    for group, symbols in _AAMI_BEATS.items():
        if symbol in symbols:
            return group
    raise ValueError(f"unsupported beat annotation symbol {symbol!r}")


def beat_centered_windows(
    signal: Any,
    beat_positions: Any,
    beat_symbols: Sequence[str],
    *,
    samples: int = 360,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract fixed beat-centered windows, dropping only out-of-range beats."""
    values = np.asarray(signal, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    positions = np.asarray(beat_positions, dtype=np.int64).reshape(-1)
    symbols = np.asarray(beat_symbols, dtype=object).reshape(-1)
    if values.ndim != 2 or len(positions) != len(symbols) or samples <= 0:
        raise ValueError("signal must be 2D and beat positions/symbols must align")
    left = samples // 2
    right = samples - left
    windows, labels, retained = [], [], []
    for position, symbol in zip(positions, symbols):
        start, end = int(position) - left, int(position) + right
        if start < 0 or end > len(values):
            continue
        windows.append(values[start:end].reshape(-1))
        labels.append(aami_beat_class(str(symbol)))
        retained.append(int(position))
    if not windows:
        raise ValueError("no complete beat-centered windows fit inside the signal")
    return np.stack(windows), np.asarray(labels), np.asarray(retained, dtype=np.int64)


def _series_ids(entity_ids: Any | None, size: int) -> np.ndarray:
    if entity_ids is None:
        return np.zeros(size, dtype=np.int64)
    ids = np.asarray(entity_ids).reshape(-1)
    if len(ids) != size:
        raise ValueError("entity_ids must align with observations")
    return ids


def temporal_windows(
    values: Any,
    *,
    lookback: int = 128,
    horizon: int = 1,
    stride: int | None = None,
    entity_ids: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return flattened lag features, future targets, and source-row end indexes.

    Windows never cross entities. The returned source-row positions identify
    each target observation, so chronological partitions can be assigned by
    target time without allowing a training target past its split boundary.
    """
    series = np.asarray(values, dtype=np.float32)
    if series.ndim == 1:
        series = series[:, None]
    if series.ndim != 2 or np.isinf(series).any():
        raise ValueError("values must be a (observations, channels) array without infinities")
    if lookback <= 0 or horizon <= 0:
        raise ValueError("lookback and horizon must be positive")
    if stride is not None and stride <= 0:
        raise ValueError("stride must be positive")
    entities = _series_ids(entity_ids, len(series))
    entity_indexes = [(entity, np.flatnonzero(entities == entity)) for entity in np.unique(entities)]
    estimated_rows = sum(
        max(0, (len(indexes) - lookback - horizon) // (
            int(stride) if stride is not None else (1 if len(indexes) < 10_000 else 24)
        ) + 1)
        for _entity, indexes in entity_indexes
    )
    features = np.empty((estimated_rows, lookback * series.shape[1]), dtype=np.float32)
    targets = np.empty((estimated_rows, series.shape[1]), dtype=np.float32)
    endpoints = np.empty(estimated_rows, dtype=np.int64)
    row = 0
    for _entity, indexes in entity_indexes:
        entity_stride = int(stride) if stride is not None else (1 if len(indexes) < 10_000 else 24)
        for start in range(0, len(indexes) - lookback - horizon + 1, entity_stride):
            window = indexes[start : start + lookback]
            future = indexes[start + lookback + horizon - 1]
            if not np.isfinite(series[future]).all():
                continue
            features[row] = series[window].reshape(-1)
            targets[row] = series[future]
            endpoints[row] = int(future)
            row += 1
    if row == 0:
        raise ValueError("series are too short for the requested lookback and horizon")
    return features[:row], targets[:row], endpoints[:row]


def chronological_split(
    row_times: Any,
    *,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.1,
    entity_ids: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split observations by time within each entity, preserving row order."""
    times = np.asarray(row_times).reshape(-1)
    entities = _series_ids(entity_ids, len(times))
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise ValueError("split fractions must be positive and leave a non-empty test partition")
    parts: list[list[int]] = [[], [], []]
    for entity in np.unique(entities):
        indexes = np.flatnonzero(entities == entity)
        indexes = indexes[np.argsort(times[indexes], kind="stable")]
        n_train = int(np.floor(len(indexes) * train_fraction))
        n_valid = int(np.floor(len(indexes) * validation_fraction))
        if min(n_train, n_valid, len(indexes) - n_train - n_valid) < 1:
            raise ValueError(f"entity {entity!r} has too few observations for three-way split")
        parts[0].extend(indexes[:n_train].tolist())
        parts[1].extend(indexes[n_train : n_train + n_valid].tolist())
        parts[2].extend(indexes[n_train + n_valid :].tolist())
    return tuple(np.asarray(part, dtype=np.int64) for part in parts)  # type: ignore[return-value]


def sequence_windows(
    features: Any,
    labels: Any,
    *,
    window_size: int = 128,
    stride: int | None = None,
    entity_ids: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create non-overlapping per-step sequence examples within each subject."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("features and per-step labels must be aligned 2D/1D arrays")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    stride = window_size if stride is None else int(stride)
    if stride <= 0:
        raise ValueError("stride must be positive")
    entities = _series_ids(entity_ids, len(x))
    windows: list[np.ndarray] = []
    label_windows: list[np.ndarray] = []
    owners: list[Any] = []
    for entity in np.unique(entities):
        indexes = np.flatnonzero(entities == entity)
        for start in range(0, len(indexes) - window_size + 1, stride):
            selected = indexes[start : start + window_size]
            windows.append(x[selected].reshape(-1))
            label_windows.append(y[selected])
            owners.append(entity)
    if not windows:
        raise ValueError("no complete sequence windows were available")
    return np.stack(windows), np.stack(label_windows), np.asarray(owners)


def grouped_chronological_split(
    group_ids: Any,
    timestamps: Any,
    *,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split every ranking group chronologically and keep group IDs intact."""
    groups = np.asarray(group_ids).reshape(-1)
    times = np.asarray(timestamps).reshape(-1)
    if groups.shape != times.shape:
        raise ValueError("group IDs and timestamps must align")
    return chronological_split(times, train_fraction=train_fraction,
                               validation_fraction=validation_fraction, entity_ids=groups)


def stratified_split(
    labels: Any,
    *,
    seed: int = 42,
    fractions: Sequence[float] = (0.6, 0.2, 0.2),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seeded class-stratified index split with deterministic per-class rounding."""
    y = np.asarray(labels).reshape(-1)
    ratios = np.asarray(fractions, dtype=np.float64)
    if y.size == 0 or len(ratios) != 3 or np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1):
        raise ValueError("stratified split needs labels and three positive fractions summing to one")
    rng = np.random.default_rng(seed)
    partitions: list[list[int]] = [[], [], []]
    for label in np.unique(y):
        indexes = np.flatnonzero(y == label)
        if len(indexes) < 3:
            raise ValueError(f"class {label!r} needs at least three rows for train/validation/test")
        rng.shuffle(indexes)
        counts = np.floor(ratios * len(indexes)).astype(int)
        counts[: len(indexes) - int(counts.sum())] += 1
        # Keep all three partitions populated for each class.
        for partition in range(3):
            if counts[partition] == 0:
                donor = int(np.argmax(counts))
                counts[donor] -= 1
                counts[partition] += 1
        cursor = 0
        for partition, count in enumerate(counts):
            partitions[partition].extend(indexes[cursor : cursor + count].tolist())
            cursor += int(count)
    return tuple(np.asarray(sorted(part), dtype=np.int64) for part in partitions)  # type: ignore[return-value]


def event_window_labels(
    timestamps: Any,
    event_intervals: Sequence[tuple[Any, Any]],
    *,
    window_size: int,
    stride: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return contiguous window bounds and anomaly labels from inclusive intervals."""
    times = np.asarray(timestamps).reshape(-1)
    stride = window_size if stride is None else int(stride)
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    bounds, labels = [], []
    for start in range(0, len(times) - window_size + 1, stride):
        end = start + window_size
        window_start, window_end = times[start], times[end - 1]
        anomalous = any(window_start <= event_end and window_end >= event_start
                        for event_start, event_end in event_intervals)
        bounds.append((start, end))
        labels.append(int(anomalous))
    return np.asarray(bounds, dtype=np.int64), np.asarray(labels, dtype=np.int64)
