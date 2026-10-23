"""Built-in public MovieLens provider for rating, preference and ranking tasks.

The provider downloads only the two public GroupLens archives, caches them
outside the checkout, and records the archive digest and terms in run metadata.
Other curated sources are represented by separately registered providers so
their data-specific parsing and split rules cannot be silently guessed here.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..advanced_tasks import GroupedRankingTask, SupervisedPreferenceTask
from ..tasks import RegressionTask
from .datasets import DatasetTaskRef, PreparedTaskData
from .plan import SplitSpec, _sha_indices


_MOVIELENS = {
    "ml-100k": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
        "archive": "ml-100k.zip",
        "member": "ml-100k/u.data",
        "delimiter": "\t",
        "header": None,
        "license": "GroupLens MovieLens 100K README terms; research/education use",
    },
    "ml-1m": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "archive": "ml-1m.zip",
        "member": "ml-1m/ratings.dat",
        "delimiter": "::",
        "header": None,
        "license": "GroupLens MovieLens 1M README terms; non-commercial research use",
    },
}


class MovieLensProvider:
    """Provider for explicit ratings and rating-derived supervised tasks."""

    name = "movielens"

    @property
    def suites(self) -> tuple[str, ...]:
        return tuple(_MOVIELENS)

    def __init__(self, cache_dir: str | Path | None = None, *, offline: bool = False) -> None:
        configured = cache_dir or os.environ.get("SIMPLE_ADP_DATA_CACHE")
        self.cache_dir = Path(configured).expanduser() if configured else Path.home() / ".cache/simple-adp/datasets"
        self.offline = offline

    def list_tasks(self, suite: str) -> list[DatasetTaskRef]:
        if suite not in _MOVIELENS:
            raise ValueError(f"unknown MovieLens suite {suite!r}; choose from {', '.join(_MOVIELENS)}")
        dataset = suite
        return [
            DatasetTaskRef(self.name, suite, f"{dataset}-{variant}", dataset_id=dataset,
                           metadata={"task_variant": variant})
            for variant in ("rating", "preference", "ranking")
        ]

    def _archive(self, dataset: str) -> tuple[Path, str]:
        spec = _MOVIELENS[dataset]
        path = self.cache_dir / spec["archive"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file():
            if self.offline:
                raise FileNotFoundError(f"MovieLens archive is not cached for offline use: {path}")
            temporary = path.with_suffix(path.suffix + ".part")
            try:
                request = urllib.request.Request(spec["url"], headers={"User-Agent": "simple-adp/0.1"})
                with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return path, digest

    def load_task(self, ref: DatasetTaskRef) -> PreparedTaskData:
        if ref.provider != self.name or ref.suite not in _MOVIELENS:
            raise ValueError("task reference does not belong to a supported MovieLens suite")
        variant = (ref.metadata or {}).get("task_variant")
        if variant not in {"rating", "preference", "ranking"}:
            raise ValueError(f"unsupported MovieLens task variant {variant!r}")
        spec = _MOVIELENS[ref.suite]
        archive, archive_hash = self._archive(ref.suite)
        with zipfile.ZipFile(archive) as bundle, bundle.open(spec["member"]) as raw:
            if ref.suite == "ml-100k":
                frame = pd.read_csv(raw, sep="\t", header=None, names=["user_id", "item_id", "rating", "timestamp"])
            else:
                frame = pd.read_csv(raw, sep="::", engine="python", header=None,
                                    names=["user_id", "item_id", "rating", "timestamp"])
        if frame.empty or frame.isna().any().any():
            raise ValueError(f"MovieLens source {ref.suite} is empty or has missing values")
        frame["user_id"] = frame["user_id"].astype(str)
        frame["item_id"] = frame["item_id"].astype(str)
        frame["rating"] = frame["rating"].astype(np.float32)
        frame["timestamp"] = frame["timestamp"].astype(np.int64)
        # Stable IDs are prepared as integer context for the group-preserving sampler.
        user_codes, _ = pd.factorize(frame["user_id"], sort=True)
        x = frame[["user_id", "item_id"]].copy().reset_index(drop=True)
        y = frame["rating"].to_numpy(dtype=np.float32)
        if variant == "rating":
            task_adapter = RegressionTask()
            task_kind, family = "regression", "recommendation_rating"
        elif variant == "preference":
            task_adapter = SupervisedPreferenceTask(relevant_threshold=4.0)
            task_kind, family = "classification", "recommendation_preference"
        else:
            task_adapter = GroupedRankingTask(relevant_threshold=4.0)
            task_kind, family = "ranking", "grouped_ranking"
        digest = hashlib.sha256()
        digest.update(archive_hash.encode())
        digest.update(variant.encode())
        return PreparedTaskData(
            dataset_id=ref.suite,
            name=f"MovieLens {ref.suite.removeprefix('ml-')}",
            target_name="rating" if variant == "rating" else "rating-derived relevance",
            X=x,
            y=y,
            feature_schema={"user_id": "categorical", "item_id": "categorical"},
            content_sha256=digest.hexdigest(),
            dataset_metadata={
                "source_url": spec["url"], "archive_sha256": archive_hash,
                "license_and_terms": spec["license"], "attribution": "GroupLens Research, University of Minnesota",
                "raw_row_count": len(frame), "variant": variant,
            },
            task_kind=task_kind,
            task_adapter=task_adapter,
            context={"group_id": user_codes.astype(np.int64), "timestamp": frame["timestamp"].to_numpy()},
            task_family=family,
            task_variant=variant,
            split_protocol="per_user_chronological_80_10_10",
        )

    def list_splits(self, ref: DatasetTaskRef, data: PreparedTaskData, settings: Any) -> list[SplitSpec]:
        groups = np.asarray(data.context["group_id"], dtype=np.int64)
        timestamps = np.asarray(data.context["timestamp"], dtype=np.int64)
        train: list[int] = []
        validation: list[int] = []
        test: list[int] = []
        for group in np.unique(groups):
            indexes = np.flatnonzero(groups == group)
            indexes = indexes[np.argsort(timestamps[indexes], kind="stable")]
            if len(indexes) < 10:
                raise ValueError(f"MovieLens user {group} has fewer than 10 interactions; cannot make 80/10/10 split")
            n_train = int(np.floor(0.8 * len(indexes)))
            n_valid = int(np.floor(0.1 * len(indexes)))
            train.extend(indexes[:n_train].tolist())
            validation.extend(indexes[n_train:n_train + n_valid].tolist())
            test.extend(indexes[n_train + n_valid:].tolist())
        train_tuple, validation_tuple, test_tuple = tuple(train), tuple(validation), tuple(test)
        split = SplitSpec(0, 0, 0, train_tuple, validation_tuple, test_tuple,
                          _sha_indices(train_tuple, validation_tuple, test_tuple))
        return [split]
