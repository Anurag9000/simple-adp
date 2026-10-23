"""Raw sensor providers for supervised per-step tabular MLP tasks."""

from __future__ import annotations

import hashlib
import os
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from ..advanced_tasks import SequenceLabelingTask
from .datasets import DatasetTaskRef, PreparedTaskData
from .plan import SplitSpec, _sha_indices


class UCIHAPTProvider:
    """HAPT raw accelerometer/gyroscope signal windows with per-sample labels."""

    name = "uci_sequence"
    source_url = "https://archive.ics.uci.edu/static/public/341/smartphone+based+recognition+of+human+activities+and+postural+transitions.zip"
    archive_name = "uci-hapt.zip"
    window_size = 128

    @property
    def suites(self) -> tuple[str, ...]:
        return ("hapt",)

    def __init__(self, cache_dir: str | Path | None = None, *, offline: bool = False) -> None:
        configured = cache_dir or os.environ.get("SIMPLE_ADP_DATA_CACHE")
        self.cache_dir = Path(configured).expanduser() if configured else Path.home() / ".cache/simple-adp/datasets"
        self.offline = offline

    def list_tasks(self, suite: str) -> list[DatasetTaskRef]:
        if suite != "hapt":
            raise ValueError("unknown UCI sequence suite; supported suite is 'hapt'")
        return [DatasetTaskRef(self.name, suite, "uci-hapt-per-step-activity", dataset_id="uci-341")]

    def _archive(self) -> Path:
        path = self.cache_dir / self.archive_name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file():
            if self.offline:
                raise FileNotFoundError(f"UCI HAPT archive is not cached for offline use: {path}")
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
    def _read_vector(bundle: zipfile.ZipFile, suffix: str) -> np.ndarray:
        members = [name for name in bundle.namelist() if name.lower().endswith(suffix.lower())]
        if len(members) != 1:
            raise ValueError(f"HAPT archive must contain exactly one {suffix} file")
        with bundle.open(members[0]) as stream:
            return np.loadtxt(stream, dtype=np.int64).reshape(-1)

    def load_task(self, ref: DatasetTaskRef) -> PreparedTaskData:
        if ref.provider != self.name or ref.suite != "hapt" or ref.task_id != "uci-hapt-per-step-activity":
            raise ValueError("unsupported UCI HAPT task reference")
        archive = self._archive()
        digest = hashlib.sha256()
        with archive.open("rb") as stream:
            while chunk := stream.read(4 * 1024 * 1024):
                digest.update(chunk)
        archive_hash = digest.hexdigest()
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.namelist()
            acc_members: dict[tuple[int, int], str] = {}
            gyro_members: dict[tuple[int, int], str] = {}
            for member in members:
                name = Path(member).name.lower()
                match = re.fullmatch(r"(acc|gyro)_exp(\d+)_user(\d+)\.txt", name)
                if match:
                    target = acc_members if match.group(1) == "acc" else gyro_members
                    key = (int(match.group(2)), int(match.group(3)))
                    if key in target:
                        raise ValueError(f"duplicate HAPT sensor file for experiment/user {key}")
                    target[key] = member
            if not acc_members or set(acc_members) != set(gyro_members):
                raise ValueError("HAPT archive must contain paired acceleration and gyroscope files")
            label_members = [member for member in members if Path(member).name.lower() == "labels.txt"]
            if len(label_members) != 1:
                raise ValueError("HAPT archive must contain exactly one RawData/labels.txt")
            with bundle.open(label_members[0]) as stream:
                label_rows = np.loadtxt(stream, dtype=np.int64, ndmin=2)
            if label_rows.ndim != 2 or label_rows.shape[1] != 5:
                raise ValueError("HAPT labels.txt must have columns experiment,user,activity,start,end")
            train_ids = self._read_vector(bundle, "Train/subject_id_train.txt")
            test_ids = self._read_vector(bundle, "Test/subject_id_test.txt")
            train_subjects = set(int(value) for value in train_ids)
            test_subjects = set(int(value) for value in test_ids)
            if not train_subjects or not test_subjects or train_subjects & test_subjects:
                raise ValueError("HAPT source train/test subject lists must be nonempty and disjoint")
            if train_subjects | test_subjects != {key[1] for key in acc_members}:
                raise ValueError("HAPT source subject lists do not cover exactly the raw sensor subjects")
            windows: list[np.ndarray] = []
            labels: list[np.ndarray] = []
            subjects: list[int] = []
            experiments: list[int] = []
            window_starts: list[int] = []
            for (experiment, subject), acc_member in sorted(acc_members.items()):
                with bundle.open(acc_member) as stream:
                    acceleration = np.loadtxt(stream, dtype=np.float32, ndmin=2)
                with bundle.open(gyro_members[(experiment, subject)]) as stream:
                    gyroscope = np.loadtxt(stream, dtype=np.float32, ndmin=2)
                if acceleration.shape != gyroscope.shape or acceleration.ndim != 2 or acceleration.shape[1] != 3:
                    raise ValueError(f"HAPT sensor channels are malformed for experiment/user {(experiment, subject)}")
                signal = np.concatenate((acceleration, gyroscope), axis=1)
                step_labels = np.full(len(signal), -1, dtype=np.int64)
                for exp_id, user_id, activity, start, end in label_rows:
                    if int(exp_id) == experiment and int(user_id) == subject:
                        start, end = int(start) - 1, int(end)
                        if start < 0 or end > len(signal) or start >= end:
                            raise ValueError("HAPT activity interval exceeds its raw signal")
                        step_labels[start:end] = int(activity)
                for start in range(0, len(signal) - self.window_size + 1, self.window_size):
                    end = start + self.window_size
                    label_window = step_labels[start:end]
                    if np.any(label_window < 0):
                        continue
                    windows.append(signal[start:end].reshape(-1))
                    labels.append(label_window)
                    subjects.append(subject)
                    experiments.append(experiment)
                    window_starts.append(start)
        if not windows:
            raise ValueError("HAPT parsing produced no fully labeled 128-sample windows")
        x = np.stack(windows).astype(np.float32, copy=False)
        y = np.stack(labels).astype(np.int64, copy=False)
        return PreparedTaskData(
            dataset_id="uci-341", name="UCI HAPT Raw Activity Signals",
            target_name="activity_label_per_sample", X=x, y=y,
            feature_schema=["numeric"] * x.shape[1],
            content_sha256=hashlib.sha256(f"{archive_hash}|window=128|stride=128".encode()).hexdigest(),
            dataset_metadata={
                "source_url": self.source_url, "archive_sha256": archive_hash,
                "uci_dataset_id": 341, "version": "UCI repository dataset 341",
                "license": "CC BY 4.0", "attribution": "UCI Machine Learning Repository",
                "train_subject_ids": sorted(train_subjects), "test_subject_ids": sorted(test_subjects),
                "window_size": self.window_size, "stride": self.window_size,
                "unlabeled_windows": "discarded; only complete labeled windows are used",
            },
            task_kind="classification", task_adapter=SequenceLabelingTask(),
            context={"subject_id": np.asarray(subjects, dtype=np.int64),
                     "experiment_id": np.asarray(experiments, dtype=np.int64),
                     "window_start": np.asarray(window_starts, dtype=np.int64)},
            task_family="sequence_labeling", task_variant="hapt_per_step_activity",
            split_protocol="official_subject_train_test_with_subject_validation_from_train",
        )

    def list_splits(self, ref: DatasetTaskRef, data: PreparedTaskData, settings: Any) -> list[SplitSpec]:
        train_subjects = set(data.dataset_metadata["train_subject_ids"])
        test_subjects = set(data.dataset_metadata["test_subject_ids"])
        ordered = sorted(train_subjects)
        validation_count = max(1, int(round(0.2 * len(ordered))))
        validation_subjects = set(ordered[-validation_count:])
        actual_train_subjects = train_subjects - validation_subjects
        row_subjects = np.asarray(data.context["subject_id"], dtype=np.int64)
        train = np.flatnonzero(np.isin(row_subjects, sorted(actual_train_subjects)))
        validation = np.flatnonzero(np.isin(row_subjects, sorted(validation_subjects)))
        test = np.flatnonzero(np.isin(row_subjects, sorted(test_subjects)))
        if not len(train) or not len(validation) or not len(test):
            raise ValueError("HAPT subject split yielded an empty train, validation or test partition")
        train_tuple, valid_tuple, test_tuple = tuple(train.tolist()), tuple(validation.tolist()), tuple(test.tolist())
        return [SplitSpec(0, 0, 0, train_tuple, valid_tuple, test_tuple,
                          _sha_indices(train_tuple, valid_tuple, test_tuple))]
