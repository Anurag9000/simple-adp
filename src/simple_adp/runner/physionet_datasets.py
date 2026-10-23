"""Open PhysioNet ECG beat classification with record-level isolation."""

from __future__ import annotations

import hashlib
import importlib
import os
from pathlib import Path
from typing import Any

import numpy as np

from ..advanced_tasks import EventDetectionTask
from ..task_preparation import aami_beat_class, beat_centered_windows
from .datasets import DatasetTaskRef, PreparedTaskData
from .plan import SplitSpec, _sha_indices


_MITDB_RECORDS = tuple(str(value) for value in (
    100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112,
    113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 200,
    201, 202, 203, 205, 207, 208, 209, 210, 212, 213, 214, 215,
    217, 219, 220, 221, 222, 223, 228, 230, 231, 232, 233, 234,
))
_LTDB_RECORDS = ("14046", "14134", "14149", "14157", "14172", "14184", "15814")


class PhysioNetECGProvider:
    """MIT-BIH Arrhythmia and Long-Term ECG datasets, pinned at v1.0.0."""

    name = "physionet_ecg"
    record_window = 360
    target_frequency = 360
    _database_info = {
        "mitdb": {
            "title": "MIT-BIH Arrhythmia Database", "version": "1.0.0",
            "records": _MITDB_RECORDS, "sampling_frequency": 360,
            "doi": "10.13026/C2F305", "source_url": "https://physionet.org/content/mitdb/1.0.0/",
        },
        "ltdb": {
            "title": "MIT-BIH Long-Term ECG Database", "version": "1.0.0",
            "records": _LTDB_RECORDS, "sampling_frequency": None,
            "doi": "10.13026/C2KS3F", "source_url": "https://physionet.org/content/ltdb/1.0.0/",
        },
    }

    def __init__(self, cache_dir: str | Path | None = None, *, offline: bool = False,
                 wfdb_module: Any | None = None) -> None:
        configured = cache_dir or os.environ.get("SIMPLE_ADP_DATA_CACHE")
        self.cache_dir = Path(configured).expanduser() if configured else Path.home() / ".cache/simple-adp/datasets"
        self.offline = offline
        self._wfdb = wfdb_module
        # Test seam; production keeps the complete versioned PhysioNet lists.
        self.record_overrides: dict[str, tuple[str, ...]] = {}

    @property
    def suites(self) -> tuple[str, ...]:
        return tuple(self._database_info)

    def list_tasks(self, suite: str) -> list[DatasetTaskRef]:
        if suite not in self._database_info:
            raise ValueError(f"unknown PhysioNet ECG suite {suite!r}; supported: {', '.join(self.suites)}")
        info = self._database_info[suite]
        return [DatasetTaskRef(self.name, suite, f"{suite}-aami-beat-classification",
                               dataset_id=f"physionet-{suite}-{info['version']}")]

    def _module(self):
        if self._wfdb is None:
            try:
                self._wfdb = importlib.import_module("wfdb")
            except ImportError as exc:
                raise RuntimeError("PhysioNet ECG loading requires the optional 'physionet' extra (wfdb)") from exc
        return self._wfdb

    def _records(self, suite: str) -> tuple[str, ...]:
        return self.record_overrides.get(suite, self._database_info[suite]["records"])

    def _database_dir(self, suite: str) -> Path:
        info = self._database_info[suite]
        directory = self.cache_dir / f"{suite}-{info['version']}"
        directory.mkdir(parents=True, exist_ok=True)
        records = self._records(suite)
        missing = [record for record in records if not (directory / f"{record}.hea").is_file()]
        if missing:
            if self.offline:
                raise FileNotFoundError(f"PhysioNet {suite} records are not fully cached: {missing[:5]}")
            self._module().dl_database(suite, dl_dir=str(directory), records=list(records))
        return directory

    @staticmethod
    def _source_digest(directory: Path, records: tuple[str, ...]) -> str:
        digest = hashlib.sha256()
        found = 0
        for record in records:
            for extension in ("hea", "dat", "atr"):
                path = directory / f"{record}.{extension}"
                if path.is_file():
                    found += 1
                    digest.update(path.name.encode())
                    with path.open("rb") as stream:
                        while chunk := stream.read(4 * 1024 * 1024):
                            digest.update(chunk)
        if found == 0:
            raise FileNotFoundError(f"no waveform/annotation files found in {directory}")
        return digest.hexdigest()

    @staticmethod
    def _resample(wfdb: Any, signal: np.ndarray, source_hz: float, target_hz: int) -> np.ndarray:
        values = np.asarray(signal, dtype=np.float32)
        if values.ndim == 2:
            values = values[:, 0]
        if source_hz == target_hz:
            return values
        resampled, _ = wfdb.processing.resample_sig(values, float(source_hz), float(target_hz))
        return np.asarray(resampled, dtype=np.float32)

    def load_task(self, ref: DatasetTaskRef) -> PreparedTaskData:
        if ref.provider != self.name or ref.suite not in self._database_info:
            raise ValueError("unsupported PhysioNet ECG task reference")
        if ref.task_id != f"{ref.suite}-aami-beat-classification":
            raise ValueError("unsupported PhysioNet ECG task ID")
        wfdb = self._module()
        info = self._database_info[ref.suite]
        records = self._records(ref.suite)
        directory = self._database_dir(ref.suite)
        source_digest = self._source_digest(directory, records)
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        record_ids: list[np.ndarray] = []
        beat_positions_context: list[np.ndarray] = []
        patient_codes = {record: index for index, record in enumerate(records)}
        if ref.suite == "mitdb" and "201" in patient_codes and "202" in patient_codes:
            patient_codes["202"] = patient_codes["201"]
        original_frequencies: dict[str, float] = {}
        for record in records:
            signal, fields = wfdb.rdsamp(str(directory / record))
            source_hz = float(fields.get("fs") or info["sampling_frequency"] or 0)
            if source_hz <= 0:
                raise ValueError(f"PhysioNet record {record} has no valid sampling frequency")
            original_frequencies[record] = source_hz
            annotations = wfdb.rdann(str(directory / record), extension="atr")
            samples = np.asarray(annotations.sample, dtype=np.int64)
            symbols = [str(symbol) for symbol in annotations.symbol]
            signal_360 = self._resample(wfdb, signal, source_hz, self.target_frequency)
            scaled_samples = np.rint(samples.astype(np.float64) * self.target_frequency / source_hz).astype(np.int64)
            beat_samples: list[int] = []
            beat_symbols: list[str] = []
            for sample, symbol in zip(scaled_samples, symbols):
                try:
                    aami_beat_class(symbol)
                except ValueError:
                    # Rhythm/noise/marker annotations are not beat targets.
                    continue
                beat_samples.append(int(sample))
                beat_symbols.append(symbol)
            try:
                x_record, y_record, centers = beat_centered_windows(
                    signal_360, beat_samples, beat_symbols, samples=self.record_window,
                )
            except ValueError:
                continue
            xs.append(x_record)
            ys.append(y_record.astype(str))
            record_ids.append(np.full(len(centers), int(record), dtype=np.int64))
            beat_positions_context.append(centers)
        if not xs:
            raise ValueError(f"PhysioNet {ref.suite} produced no complete labeled beat windows")
        x, y = np.concatenate(xs), np.concatenate(ys)
        numeric_records = np.concatenate(record_ids)
        group_codes = np.asarray([patient_codes[str(record)] for record in numeric_records], dtype=np.int64)
        positions = np.concatenate(beat_positions_context)
        return PreparedTaskData(
            dataset_id=f"physionet-{ref.suite}-{info['version']}", name=str(info["title"]),
            target_name="aami_beat_group", X=x, y=y,
            feature_schema=["numeric"] * self.record_window,
            content_sha256=hashlib.sha256(f"{source_digest}|window={self.record_window}|hz={self.target_frequency}".encode()).hexdigest(),
            dataset_metadata={
                "source_url": info["source_url"], "version": info["version"],
                "doi": info["doi"], "source_files_sha256": source_digest,
                "license": "Open Data Commons Attribution License v1.0",
                "attribution": "PhysioNet; cite the database-specific source publication and DOI",
                "source_records": list(records), "source_sampling_frequencies_hz": original_frequencies,
                "resampled_frequency_hz": self.target_frequency, "window_samples": self.record_window,
                "window_policy": "180 samples before and 180 samples from each annotated beat center",
                "aami_groups": ["N", "S", "V", "F", "Q"],
            },
            task_kind="classification", task_adapter=EventDetectionTask(),
            context={"record_id": numeric_records, "patient_group_id": group_codes,
                     "beat_sample_360hz": positions},
            task_family="event_detection", task_variant=f"{ref.suite}_aami_beat_classification",
            split_protocol="seeded_record_patient_group_60_20_20_with_201_202_paired",
        )

    def list_splits(self, ref: DatasetTaskRef, data: PreparedTaskData, settings: Any) -> list[SplitSpec]:
        groups = np.asarray(data.context["patient_group_id"], dtype=np.int64)
        labels = np.asarray(data.y).astype(str)
        unique_groups = np.unique(groups)
        if len(unique_groups) < 3:
            raise ValueError("PhysioNet record-level split requires at least three patient groups")
        rng = np.random.default_rng(int(getattr(settings, "seed", 42)))
        # Build a training partition that contains every observed AAMI class;
        # validation and test then partition only whole held-out record groups.
        group_order = unique_groups.copy()
        rng.shuffle(group_order)
        train_groups: list[int] = []
        remaining_classes = set(np.unique(labels).tolist())
        group_rows = {int(group): np.flatnonzero(groups == group) for group in unique_groups}
        group_classes = {int(group): set(labels[group_rows[int(group)]].tolist()) for group in unique_groups}
        while remaining_classes:
            candidates = [int(group) for group in group_order if int(group) not in train_groups]
            if not candidates:
                raise ValueError("cannot construct a training record partition covering all AAMI classes")
            chosen = max(candidates, key=lambda group: (len(group_classes[group] & remaining_classes),
                                                        len(group_rows[group]), -int(np.flatnonzero(group_order == group)[0])))
            gain = group_classes[chosen] & remaining_classes
            if not gain:
                raise ValueError("at least one AAMI class occurs only in groups already assigned outside training")
            train_groups.append(chosen)
            remaining_classes -= gain
        target_train = int(np.ceil(0.6 * len(groups)))
        for group in group_order:
            group = int(group)
            if len(np.concatenate([group_rows[value] for value in train_groups])) >= target_train:
                break
            if group not in train_groups:
                train_groups.append(group)
        heldout = [int(group) for group in group_order if int(group) not in train_groups]
        if len(heldout) < 2:
            raise ValueError("PhysioNet split needs at least two held-out record groups")
        validation_groups: list[int] = []
        validation_rows = 0
        target_validation = max(1, int(round(0.5 * sum(len(group_rows[group]) for group in heldout))))
        for group in heldout:
            if validation_rows < target_validation and len(validation_groups) < len(heldout) - 1:
                validation_groups.append(group)
                validation_rows += len(group_rows[group])
        test_groups = [group for group in heldout if group not in validation_groups]
        train = np.flatnonzero(np.isin(groups, train_groups))
        validation = np.flatnonzero(np.isin(groups, validation_groups))
        test = np.flatnonzero(np.isin(groups, test_groups))
        if min(len(train), len(validation), len(test)) == 0:
            raise ValueError("PhysioNet record-level split produced an empty partition")
        train_classes = set(labels[train].tolist())
        if not set(labels).issubset(train_classes):
            raise ValueError("PhysioNet training records do not cover all classes present in the dataset")
        train_tuple, valid_tuple, test_tuple = tuple(train.tolist()), tuple(validation.tolist()), tuple(test.tolist())
        return [SplitSpec(0, 0, 0, train_tuple, valid_tuple, test_tuple,
                          _sha_indices(train_tuple, valid_tuple, test_tuple))]
