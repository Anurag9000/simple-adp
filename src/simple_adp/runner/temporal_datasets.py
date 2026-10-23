"""Time-series public provider implementations with fixed-window MLP rows."""

from __future__ import annotations

import hashlib
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..advanced_tasks import ForecastingTask
from ..task_preparation import chronological_split, temporal_windows
from .datasets import DatasetTaskRef, PreparedTaskData
from .plan import SplitSpec, _sha_indices


class UCIElectricityProvider:
    """UCI Electricity Load Diagrams as 370-client fixed-window forecasting."""

    name = "uci_temporal"
    source_url = "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
    archive_name = "electricityloaddiagrams20112014.zip"
    lookback = 128
    horizon = 1

    @property
    def suites(self) -> tuple[str, ...]:
        return ("electricity",)

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(4 * 1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def __init__(self, cache_dir: str | Path | None = None, *, offline: bool = False) -> None:
        configured = cache_dir or os.environ.get("SIMPLE_ADP_DATA_CACHE")
        self.cache_dir = Path(configured).expanduser() if configured else Path.home() / ".cache/simple-adp/datasets"
        self.offline = offline

    def list_tasks(self, suite: str) -> list[DatasetTaskRef]:
        if suite != "electricity":
            raise ValueError("unknown UCI temporal suite; supported suite is 'electricity'")
        return [DatasetTaskRef(self.name, suite, "uci-electricity-load-h1", dataset_id="uci-321")]

    def _archive(self) -> Path:
        path = self.cache_dir / self.archive_name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file():
            if self.offline:
                raise FileNotFoundError(f"UCI Electricity archive is not cached for offline use: {path}")
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
    def _data_member(bundle: zipfile.ZipFile) -> str:
        matches = [name for name in bundle.namelist() if Path(name).name.lower() == "ld2011_2014.txt"]
        if len(matches) != 1:
            raise ValueError("UCI Electricity archive must contain exactly one LD2011_2014.txt file")
        return matches[0]

    def load_task(self, ref: DatasetTaskRef) -> PreparedTaskData:
        if ref.provider != self.name or ref.suite != "electricity" or ref.task_id != "uci-electricity-load-h1":
            raise ValueError("unsupported UCI Electricity task reference")
        archive = self._archive()
        archive_digest = self._file_sha256(archive)
        with zipfile.ZipFile(archive) as bundle:
            with bundle.open(self._data_member(bundle)) as raw:
                frame = pd.read_csv(raw, sep=";", decimal=",", index_col=0)
        if frame.empty or frame.shape[1] < 1:
            raise ValueError("UCI Electricity source has no consumption columns")
        frame.columns = [str(column) for column in frame.columns]
        values = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        stride = 1 if len(frame) < 10_000 else 24
        x, y, target_positions = temporal_windows(
            values, lookback=self.lookback, horizon=self.horizon, stride=stride,
        )
        feature_names = [f"lag_{lag}_meter_{meter}" for lag in range(self.lookback) for meter in frame.columns]
        digest = hashlib.sha256(f"{archive_digest}|lookback={self.lookback}|horizon={self.horizon}|stride={stride}".encode()).hexdigest()
        return PreparedTaskData(
            dataset_id="uci-321", name="UCI Electricity Load Diagrams 2011-2014",
            target_name="next_interval_meter_consumption", X=x, y=y,
            feature_schema=["numeric"] * len(feature_names), content_sha256=digest,
            dataset_metadata={
                "source_url": self.source_url, "archive_sha256": archive_digest,
                "uci_dataset_id": 321, "version": "UCI repository dataset 321",
                "license": "CC BY 4.0", "attribution": "UCI Machine Learning Repository",
                "raw_observations": len(frame), "meters": int(frame.shape[1]),
                "lookback": self.lookback, "horizon": self.horizon,
                "stride": stride,
            },
            task_kind="regression", task_adapter=ForecastingTask(),
            context={"target_position": target_positions},
            task_family="forecasting", task_variant="electricity_next_interval",
            split_protocol="chronological_target_position_70_10_20",
        )

    def list_splits(self, ref: DatasetTaskRef, data: PreparedTaskData, settings: Any) -> list[SplitSpec]:
        train, validation, test = chronological_split(
            data.context["target_position"], train_fraction=0.7, validation_fraction=0.1,
        )
        train_tuple, valid_tuple, test_tuple = tuple(train.tolist()), tuple(validation.tolist()), tuple(test.tolist())
        return [SplitSpec(0, 0, 0, train_tuple, valid_tuple, test_tuple,
                          _sha_indices(train_tuple, valid_tuple, test_tuple))]
