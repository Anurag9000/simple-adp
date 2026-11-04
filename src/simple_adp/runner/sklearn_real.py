"""Opt-in, immutable, download-free real data for whole-worker ADP pairing.

Never registered in the legacy provider inventory unless the explicit
SIMPLE_ADP_ENABLE_SKLEARN_REAL=1 opt-in is active. Uses scikit-learn packaged
Fisher Iris and Diabetes data, NOT test fixtures or synthetic examples.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .datasets import DatasetTaskRef, PreparedTaskData
from .plan import SplitSpec, _sha_indices


class SklearnRealDatasetProvider:
    name = "sklearn_real"
    suites = ("iris", "diabetes")
    _TASK_IDS = {"iris": "iris-150", "diabetes": "diabetes-442"}

    def list_tasks(self, suite: str) -> list[DatasetTaskRef]:
        if suite not in self.suites:
            raise ValueError(f"unsupported packaged sklearn real dataset suite: {suite!r}")
        return [DatasetTaskRef(self.name, suite, self._TASK_IDS[suite])]

    def load_task(self, ref: DatasetTaskRef) -> PreparedTaskData:
        if (ref.provider != self.name or ref.suite not in self.suites
                or ref.task_id != self._TASK_IDS[ref.suite]):
            raise ValueError("invalid sklearn real dataset task reference")
        import sklearn
        from sklearn.datasets import load_diabetes, load_iris

        dataset = load_iris() if ref.suite == "iris" else load_diabetes()
        X = np.ascontiguousarray(dataset.data, dtype=np.float32)
        y = np.ascontiguousarray(
            dataset.target, dtype=np.int64 if ref.suite == "iris" else np.float32,
        )
        source = (
            "sklearn-iris-packed-real-data-v1"
            if ref.suite == "iris" else "sklearn-diabetes-packed-real-data-v1"
        )
        digest = hashlib.sha256(source.encode("ascii"))
        digest.update(X.tobytes(order="C"))
        digest.update(y.tobytes(order="C"))
        is_classification = ref.suite == "iris"
        return PreparedTaskData(
            dataset_id=f"sklearn-{ref.task_id}",
            name=f"Real scikit-learn {ref.suite}",
            target_name="species" if is_classification else "disease progression",
            X=X, y=y, feature_schema=None,
            content_sha256=digest.hexdigest(),
            dataset_metadata={
                "origin": "sklearn.datasets.load_iris" if is_classification
                          else "sklearn.datasets.load_diabetes",
                "sklearn_version": sklearn.__version__,
                "samples": int(len(X)), "features": int(X.shape[1]),
                "outer_split_seed": 731, "inner_split_seed": 732,
                "classification_stratified": is_classification,
                "downloaded": False,
            },
            task_kind="classification" if is_classification else "regression",
            task_adapter=None,
            task_family="tabular",
            task_variant=f"{ref.suite}_real_worker_pair",
            split_protocol="fixed_seed_731_732_60_20_20",
        )

    def list_splits(self, ref: DatasetTaskRef, data: PreparedTaskData, settings):
        if (ref.provider != self.name or ref.suite not in self.suites
                or ref.task_id != self._TASK_IDS[ref.suite]):
            raise ValueError("invalid sklearn real split source")
        from sklearn.model_selection import train_test_split

        indices = np.arange(len(data.y), dtype=np.int64)
        stratify = data.y if ref.suite == "iris" else None
        development, test = train_test_split(
            indices, test_size=0.2, random_state=731,
            stratify=stratify,
        )
        train, validation = train_test_split(
            development, test_size=0.25, random_state=732,
            stratify=data.y[development] if ref.suite == "iris" else None,
        )
        parts = tuple(tuple(int(v) for v in array)
                      for array in (train, validation, test))
        if sum(len(part) for part in parts) != len(indices) or (
            len(set().union(*(set(part) for part in parts))) != len(indices)
        ):
            raise RuntimeError("real sklearn dataset partition has overlap or missing rows")
        return [SplitSpec(0, 0, 0, *parts, _sha_indices(*parts))]
