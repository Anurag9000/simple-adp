"""Monash TSF forecasting provider with an explicit, versioned artifact list."""

from __future__ import annotations

import hashlib
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from ..advanced_tasks import ForecastingTask
from ..task_preparation import chronological_split, temporal_windows
from .datasets import DatasetTaskRef, PreparedTaskData
from .plan import SplitSpec, _sha_indices


# Curated allowlist: this Monash variation is explicitly CC BY 4.0 at Zenodo;
# its five long series remain tractable with the documented long-series stride.
_ARTIFACTS = {
    "aus-electricity-demand": {
        "record_id": "4659727",
        "title": "Australian Electricity Demand Dataset",
        "archive": "australian_electricity_demand_dataset.zip",
        "zenodo_version": "1",
        "supported_horizons": (1, 7, 28),
        "frequency": "half_hourly",
    },
}


def parse_tsf(text: str) -> tuple[list[np.ndarray], dict[str, str | int | bool]]:
    """Parse the Monash/sktime TSF text format without lossy dataframe casts."""
    attributes: list[tuple[str, str]] = []
    metadata: dict[str, str | int | bool] = {}
    series: list[np.ndarray] = []
    in_data = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not in_data:
            if line.lower() == "@data":
                in_data = True
                continue
            if line.lower().startswith("@attribute "):
                fields = line.split()
                if len(fields) != 3 or fields[2] not in {"numeric", "string", "date"}:
                    raise ValueError(f"invalid TSF attribute declaration: {line!r}")
                attributes.append((fields[1], fields[2]))
            elif line.startswith("@"):
                key, _, value = line[1:].partition(" ")
                if not value:
                    raise ValueError(f"invalid TSF metadata declaration: {line!r}")
                normalized = value.strip()
                if key == "horizon":
                    metadata[key] = int(normalized)
                elif key in {"missing", "equallength"}:
                    if normalized.lower() not in {"true", "false"}:
                        raise ValueError(f"invalid TSF boolean metadata: {line!r}")
                    metadata[key] = normalized.lower() == "true"
                else:
                    metadata[key] = normalized
            else:
                raise ValueError("TSF series data appeared before @data")
            continue
        fields = line.split(":")
        if len(fields) != len(attributes) + 1:
            raise ValueError("TSF row attribute count differs from its declarations")
        values = [np.nan if value.strip() == "?" else float(value) for value in fields[-1].split(",")]
        if not values or not np.isfinite(np.asarray(values, dtype=np.float64)).any():
            raise ValueError("TSF series must contain at least one observed numeric value")
        series.append(np.asarray(values, dtype=np.float32))
    if not in_data or not series:
        raise ValueError("TSF file must contain @data and at least one series")
    return series, metadata


class MonashForecastingProvider:
    """Fixed-lookback Australian electricity forecasting across five states."""

    name = "monash"
    lookback = 128
    stride: int | None = None

    def __init__(self, cache_dir: str | Path | None = None, *, offline: bool = False) -> None:
        configured = cache_dir or os.environ.get("SIMPLE_ADP_DATA_CACHE")
        self.cache_dir = Path(configured).expanduser() if configured else Path.home() / ".cache/simple-adp/datasets"
        self.offline = offline

    @property
    def suites(self) -> tuple[str, ...]:
        return tuple(_ARTIFACTS)

    @staticmethod
    def _sha(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(4 * 1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def list_tasks(self, suite: str) -> list[DatasetTaskRef]:
        if suite not in _ARTIFACTS:
            raise ValueError(f"unknown Monash suite {suite!r}; supported: {', '.join(self.suites)}")
        artifact = _ARTIFACTS[suite]
        return [DatasetTaskRef(self.name, suite, f"monash-{suite}-h{horizon}",
                               dataset_id=f"zenodo-{artifact['record_id']}",
                               metadata={"horizon": horizon})
                for horizon in artifact["supported_horizons"]]

    def _archive(self, suite: str) -> Path:
        artifact = _ARTIFACTS[suite]
        path = self.cache_dir / f"monash-{suite}.zip"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file():
            if self.offline:
                raise FileNotFoundError(f"Monash archive is not cached for offline use: {path}")
            url = f"https://zenodo.org/api/records/{artifact['record_id']}/files/{artifact['archive']}/content"
            temporary = path.with_suffix(path.suffix + ".part")
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "simple-adp/0.1"})
                with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as output:
                    while chunk := response.read(4 * 1024 * 1024):
                        output.write(chunk)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        return path

    def load_task(self, ref: DatasetTaskRef) -> PreparedTaskData:
        if ref.provider != self.name or ref.suite not in _ARTIFACTS:
            raise ValueError("unsupported Monash task reference")
        artifact = _ARTIFACTS[ref.suite]
        try:
            horizon = int(ref.metadata["horizon"])
        except (TypeError, KeyError, ValueError) as exc:
            raise ValueError("Monash task reference must include a supported forecast horizon") from exc
        if ref.task_id != f"monash-{ref.suite}-h{horizon}" or horizon not in artifact["supported_horizons"]:
            raise ValueError("unsupported Monash task ID or horizon")
        archive = self._archive(ref.suite)
        archive_hash = self._sha(archive)
        with zipfile.ZipFile(archive) as bundle:
            members = [name for name in bundle.namelist() if name.lower().endswith(".tsf")]
            if len(members) != 1:
                raise ValueError("Monash archive must contain exactly one TSF file")
            text = bundle.read(members[0]).decode("cp1252")
        series, source_metadata = parse_tsf(text)
        declared_horizon = int(source_metadata.get("horizon", -1))
        frequency = str(source_metadata.get("frequency", "unknown"))
        if (declared_horizon != -1 and horizon > declared_horizon) or frequency.lower() != str(artifact["frequency"]):
            raise ValueError("Monash TSF metadata does not match the pinned catalog record")
        values = np.concatenate(series)
        entities = np.concatenate([np.full(len(values_i), index, dtype=np.int64)
                                   for index, values_i in enumerate(series)])
        x, y, target_positions = temporal_windows(values, lookback=self.lookback,
                                                  horizon=horizon, stride=self.stride,
                                                  entity_ids=entities)
        y = y[:, 0]
        example_entities = np.concatenate([np.full(max(0, (len(values_i) - self.lookback - horizon) //
                                                       (1 if len(values_i) < 10_000 else 24) + 1), index,
                                                   dtype=np.int64)
                                           for index, values_i in enumerate(series)
                                           if len(values_i) >= self.lookback + horizon])
        if len(example_entities) != len(x):
            # Windows whose target is missing are intentionally omitted by
            # temporal_windows; construct entity IDs against returned global
            # target source positions to keep context exactly row-aligned.
            example_entities = entities[target_positions]
        digest = hashlib.sha256(f"{archive_hash}|{self.lookback}|{horizon}|stride=1_below_10000_else_24".encode()).hexdigest()
        return PreparedTaskData(
            dataset_id=f"zenodo-{artifact['record_id']}",
            name=str(artifact["title"]), target_name=f"forecast_{horizon}_steps_ahead",
            X=x, y=y, feature_schema=["numeric"] * self.lookback,
            content_sha256=digest,
            dataset_metadata={
                "source_url": f"https://zenodo.org/records/{artifact['record_id']}",
                "archive_sha256": archive_hash, "zenodo_record": artifact["record_id"],
                "archive_file": artifact["archive"], "artifact_version": artifact["zenodo_version"],
                "frequency": frequency, "official_forecast_horizon": declared_horizon if declared_horizon >= 0 else None,
                "license": "CC BY 4.0",
                "attribution": "Monash Australian Electricity Demand Dataset; O'Hara-Wild et al. tsibbledata; cite Zenodo record 4659727",
                "series_count": len(series), "lookback": self.lookback, "horizon": horizon,
                "stride": "1 for series shorter than 10,000 observations; otherwise 24",
            },
            task_kind="regression", task_adapter=ForecastingTask(),
            context={"entity_id": example_entities, "target_position": target_positions},
            task_family="forecasting", task_variant=f"aus_electricity_demand_h{horizon}",
            split_protocol="chronological_per_series_target_position_70_10_20",
        )

    def list_splits(self, ref: DatasetTaskRef, data: PreparedTaskData, settings: Any) -> list[SplitSpec]:
        train, validation, test = chronological_split(
            data.context["target_position"], train_fraction=0.7,
            validation_fraction=0.1, entity_ids=data.context["entity_id"],
        )
        train_tuple, valid_tuple, test_tuple = tuple(train.tolist()), tuple(validation.tolist()), tuple(test.tolist())
        return [SplitSpec(0, 0, 0, train_tuple, valid_tuple, test_tuple,
                          _sha_indices(train_tuple, valid_tuple, test_tuple))]
