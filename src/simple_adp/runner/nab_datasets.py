"""Pinned Numenta Anomaly Benchmark source and file-isolated event windows."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..advanced_tasks import EventDetectionTask
from ..task_preparation import event_window_labels
from .datasets import DatasetTaskRef, PreparedTaskData
from .plan import SplitSpec, _sha_indices


class NABProvider:
    """NAB v1.1 data, labeled intervals, and deterministic file-level splits."""

    name = "nab"
    commit = "0dcd73007a349ca0f4128c4c9b18133ce00d9296"
    source_url = f"https://github.com/numenta/NAB/archive/{commit}.zip"
    archive_name = f"nab-{commit}.zip"
    window_size = 128
    stride = 24
    split_seed = 42

    @property
    def suites(self) -> tuple[str, ...]:
        return ("v1.1",)

    def __init__(self, cache_dir: str | Path | None = None, *, offline: bool = False) -> None:
        configured = cache_dir or os.environ.get("SIMPLE_ADP_DATA_CACHE")
        self.cache_dir = Path(configured).expanduser() if configured else Path.home() / ".cache/simple-adp/datasets"
        self.offline = offline

    def list_tasks(self, suite: str) -> list[DatasetTaskRef]:
        if suite != "v1.1":
            raise ValueError("unknown NAB suite; supported suite is 'v1.1'")
        return [DatasetTaskRef(self.name, suite, "nab-v1.1-window-anomaly", dataset_id="nab-v1.1")]

    @staticmethod
    def _sha(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(4 * 1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def _archive(self) -> Path:
        path = self.cache_dir / self.archive_name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file():
            if self.offline:
                raise FileNotFoundError(f"NAB v1.1 archive is not cached: {path}")
            temporary = path.with_suffix(path.suffix + ".part")
            try:
                request = urllib.request.Request(self.source_url, headers={"User-Agent": "simple-adp/0.1"})
                with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                    while chunk := response.read(4 * 1024 * 1024):
                        output.write(chunk)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        return path

    @staticmethod
    def _members(bundle: zipfile.ZipFile) -> tuple[str, str]:
        csvs = [name for name in bundle.namelist() if "/data/" in name and name.lower().endswith(".csv")]
        labels = [name for name in bundle.namelist() if name.endswith("/labels/combined_labels.json")]
        if not csvs or len(labels) != 1:
            raise ValueError("pinned NAB archive must contain source CSV files and one combined_labels.json")
        return sorted(csvs), labels[0]

    def load_task(self, ref: DatasetTaskRef) -> PreparedTaskData:
        if ref.provider != self.name or ref.suite != "v1.1" or ref.task_id != "nab-v1.1-window-anomaly":
            raise ValueError("unsupported NAB task reference")
        archive = self._archive()
        archive_hash = self._sha(archive)
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        series_ids: list[np.ndarray] = []
        window_indexes: list[np.ndarray] = []
        event_ids: list[np.ndarray] = []
        source_files: list[str] = []
        with zipfile.ZipFile(archive) as bundle:
            csv_members, labels_member = self._members(bundle)
            labels = json.loads(bundle.read(labels_member).decode("utf-8"))
            label_by_path = {key.replace("\\", "/"): spans for key, spans in labels.items()}
            for series_number, member in enumerate(csv_members):
                relative = member.split("/data/", 1)[1]
                key = relative
                if key not in label_by_path:
                    raise ValueError(f"NAB labels omit source file {key!r}")
                frame = pd.read_csv(bundle.open(member))
                if frame.shape[1] < 2:
                    raise ValueError(f"NAB source file has no metric column: {relative}")
                times = pd.to_datetime(frame.iloc[:, 0], utc=True, errors="raise")
                values = pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy(dtype=np.float32)
                if len(values) < self.window_size or not np.isfinite(values).all():
                    continue
                # Pandas preserves source resolution (seconds/us/ns), so its
                # raw integer representation is not guaranteed to be ns.
                timestamps = np.asarray([int(value.value) for value in times], dtype=np.int64)
                intervals: list[tuple[int, int]] = []
                for span in label_by_path[key]:
                    if len(span) != 2:
                        raise ValueError(f"malformed NAB anomaly interval in {key!r}")
                    start, end = pd.to_datetime(span[0].strip(), utc=True), pd.to_datetime(span[1].strip(), utc=True)
                    intervals.append((int(start.value), int(end.value)))
                bounds, targets = event_window_labels(timestamps, intervals, window_size=self.window_size, stride=self.stride)
                file_windows: list[np.ndarray] = []
                file_event_ids = np.full(len(bounds), -1, dtype=np.int64)
                for index, (start, end) in enumerate(bounds):
                    file_windows.append(values[start:end])
                    left, right = timestamps[start], timestamps[end - 1]
                    for event_index, (event_start, event_end) in enumerate(intervals):
                        if left <= event_end and right >= event_start:
                            file_event_ids[index] = series_number * 1_000_000 + event_index
                            break
                xs.append(np.stack(file_windows))
                ys.append(targets.astype(np.int64))
                series_ids.append(np.full(len(bounds), series_number, dtype=np.int64))
                window_indexes.append(np.arange(len(bounds), dtype=np.int64))
                event_ids.append(file_event_ids)
                source_files.append(relative)
        if not xs or len(np.unique(np.concatenate(ys))) != 2:
            raise ValueError("NAB preparation requires at least one normal and one anomalous window")
        x, y = np.concatenate(xs), np.concatenate(ys)
        data = PreparedTaskData(
            dataset_id="nab-v1.1", name="Numenta Anomaly Benchmark v1.1",
            target_name="window_overlaps_published_anomaly_interval", X=x, y=y,
            feature_schema=["numeric"] * self.window_size,
            content_sha256=hashlib.sha256(f"{archive_hash}|window={self.window_size}|stride={self.stride}".encode()).hexdigest(),
            dataset_metadata={
                "source_url": self.source_url, "repository_commit": self.commit,
                "archive_sha256": archive_hash, "version": "NAB v1.1",
                "license": "NAB repository code is MIT; included time-series data are attributed to their original providers in data/README.md and retain source-specific terms",
                "attribution": "Numenta Anomaly Benchmark and the original sources identified in the pinned data/README.md",
                "files": source_files, "window_size": self.window_size, "stride": self.stride,
                "label_policy": "positive if a window overlaps any inclusive published anomaly interval",
            },
            task_kind="classification", task_adapter=EventDetectionTask(),
            context={"series_id": np.concatenate(series_ids),
                     "window_index": np.concatenate(window_indexes),
                     "event_id": np.concatenate(event_ids)},
            task_family="event_detection", task_variant="nab_interval_labeled_windows",
            split_protocol="seeded_file_level_60_20_20",
        )
        return data

    def list_splits(self, ref: DatasetTaskRef, data: PreparedTaskData, settings: Any) -> list[SplitSpec]:
        file_ids = np.asarray(data.context["series_id"], dtype=np.int64)
        file_labels = np.asarray([int(np.any(np.asarray(data.y)[file_ids == file] == 1)) for file in np.unique(file_ids)])
        files = np.unique(file_ids)
        seed = int(getattr(settings, "seed", self.split_seed))
        rng = np.random.default_rng(seed)
        partitions: list[list[int]] = [[], [], []]
        # Stratify files by whether the published labels give that file at
        # least one positive window; the split unit remains the whole file.
        for label in (0, 1):
            selected = files[file_labels == label].copy()
            rng.shuffle(selected)
            n = len(selected)
            counts = np.floor(np.asarray([0.6, 0.2, 0.2]) * n).astype(int)
            counts[: n - int(counts.sum())] += 1
            for part in range(3):
                if n >= 3 and counts[part] == 0:
                    donor = int(np.argmax(counts)); counts[donor] -= 1; counts[part] += 1
            cursor = 0
            for part, count in enumerate(counts):
                partitions[part].extend(int(value) for value in selected[cursor:cursor + int(count)])
                cursor += int(count)
        rows = [np.flatnonzero(np.isin(file_ids, selected)).astype(np.int64) for selected in partitions]
        if any(len(part) == 0 for part in rows) or any(len(np.unique(np.asarray(data.y)[part])) < 2 for part in rows):
            raise ValueError("NAB file-level split must give each partition normal and anomalous windows")
        train, validation, test = (tuple(part.tolist()) for part in rows)
        return [SplitSpec(0, 0, 0, train, validation, test, _sha_indices(train, validation, test))]
