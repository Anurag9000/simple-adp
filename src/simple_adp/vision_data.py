"""Opt-in deterministic real-image dataset inputs for V01.

Official test partitions are never used in architecture search. torchvision is
optional and is imported only when its named datasets are requested.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _raw_nchw(images: Any) -> np.ndarray:
    data = np.asarray(images)
    if data.ndim == 3:
        data = data[:, None, :, :]
    elif data.ndim == 4 and data.shape[1] in {1, 3}:
        pass
    elif data.ndim == 4 and data.shape[-1] in {1, 3}:
        data = np.moveaxis(data, -1, 1)
    else:
        raise ValueError("images must have [N,H,W], [N,C,H,W], or [N,H,W,C] shape")
    if data.dtype != np.uint8 or not data.size or min(data.shape) <= 0:
        raise ValueError("image arrays must be nonempty uint8 pixels")
    return np.ascontiguousarray(data)


def _bytes_digest(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _split_classwise(labels: np.ndarray, rng: np.random.Generator, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    train, valid = [], []
    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        if len(indices) < 2:
            raise ValueError("each training class needs at least two samples for nonempty validation")
        shuffled = rng.permutation(indices)
        n_valid = min(len(indices) - 1, max(1, round(len(indices) * fraction)))
        valid.extend(shuffled[:n_valid].tolist())
        train.extend(shuffled[n_valid:].tolist())
    return np.array(sorted(train), dtype=np.int64), np.array(sorted(valid), dtype=np.int64)


def _balanced_limit(indices: np.ndarray, labels: np.ndarray, limit: int | None, rng: np.random.Generator) -> np.ndarray:
    if limit is None or limit >= len(indices):
        return indices
    if limit <= 0:
        raise ValueError("sample limit must be positive")
    classes = np.unique(labels[indices])
    if limit < len(classes):
        raise ValueError("sample limit must include at least one example per class")
    buckets = {int(c): rng.permutation(indices[labels[indices] == c]).tolist() for c in classes}
    selected: list[int] = []
    while len(selected) < limit:
        progressed = False
        for label in classes:
            bucket = buckets[int(label)]
            if bucket and len(selected) < limit:
                selected.append(int(bucket.pop()))
                progressed = True
        if not progressed:
            break
    return np.asarray(sorted(selected), dtype=np.int64)


@dataclass(frozen=True)
class ImagePartitions:
    """Raw images plus immutable source-split indices and materialization API."""

    train_images: np.ndarray
    train_labels: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray
    fit_indices: np.ndarray
    validation_indices: np.ndarray
    evaluation_indices: np.ndarray
    manifest: dict[str, Any]

    def materialize(self, partition: str) -> tuple[np.ndarray, np.ndarray]:
        """Return float32 [N,C,H,W] in [0,1], with official test kept isolated."""
        if partition == "train":
            x, y, idx = self.train_images, self.train_labels, self.fit_indices
        elif partition == "validation":
            x, y, idx = self.train_images, self.train_labels, self.validation_indices
        elif partition == "test":
            x, y, idx = self.test_images, self.test_labels, self.evaluation_indices
        else:
            raise ValueError("partition must be train, validation, or test")
        return np.asarray(x[idx], dtype=np.float32) / 255.0, np.asarray(y[idx], dtype=np.int64)


def make_image_partitions(
    source_train_images: Any,
    source_train_labels: Any,
    source_test_images: Any,
    source_test_labels: Any,
    *,
    source: str,
    revision: str,
    seed: int = 0,
    validation_fraction: float = 0.2,
    train_limit: int | None = None,
    validation_limit: int | None = None,
    test_limit: int | None = None,
) -> ImagePartitions:
    """Stratify architecture-validation inside official training data only.

    Limits are applied after original group splits, with deterministic
    class-balanced sampling. All original source indices enter the manifest.
    """
    if not source.strip() or not revision.strip():
        raise ValueError("dataset source and revision identifiers are required")
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between zero and one half")
    train_x, test_x = _raw_nchw(source_train_images), _raw_nchw(source_test_images)
    train_y, test_y = np.asarray(source_train_labels), np.asarray(source_test_labels)
    if train_y.ndim != 1 or test_y.ndim != 1 or len(train_x) != len(train_y) or len(test_x) != len(test_y):
        raise ValueError("image/target partitions must have equal nonempty row counts")
    if train_x.shape[1:] != test_x.shape[1:]:
        raise ValueError("source train and test image geometries must match")
    if not np.issubdtype(train_y.dtype, np.integer) or not np.issubdtype(test_y.dtype, np.integer):
        raise ValueError("labels must be integer class identifiers")
    if set(test_y.tolist()) - set(train_y.tolist()):
        raise ValueError("official test contains an unobserved training class")
    rng = np.random.default_rng(seed)
    fit, valid = _split_classwise(train_y, rng, validation_fraction)
    evaluation = np.arange(len(test_y), dtype=np.int64)
    fit = _balanced_limit(fit, train_y, train_limit, rng)
    valid = _balanced_limit(valid, train_y, validation_limit, rng)
    evaluation = _balanced_limit(evaluation, test_y, test_limit, rng)
    assert not (set(fit.tolist()) & set(valid.tolist()))
    manifest = {
        "source": source, "revision": revision,
        "seed": int(seed), "validation_fraction": float(validation_fraction),
        "train_raw_sha256": _bytes_digest(train_x), "test_raw_sha256": _bytes_digest(test_x),
        "train_target_sha256": _bytes_digest(np.ascontiguousarray(train_y)),
        "test_target_sha256": _bytes_digest(np.ascontiguousarray(test_y)),
        "normalization": "float32 raw_uint8/255 (no fitted statistics)",
        "augmentation": "none",
        "source_train_indices": fit.tolist(),
        "source_validation_indices": valid.tolist(),
        "source_test_indices": evaluation.tolist(),
        "label_map": sorted(set(int(x) for x in train_y)),
        "image_shape": list(train_x.shape[1:]),
    }
    return ImagePartitions(train_x, train_y, test_x, test_y, fit, valid, evaluation, manifest)


def load_torchvision_partitions(
    dataset: str, root: str | Path, *, download: bool = False, seed: int = 0,
    validation_fraction: float = 0.2, train_limit: int | None = None,
    validation_limit: int | None = None, test_limit: int | None = None,
) -> ImagePartitions:
    """Load CIFAR-10, CIFAR-100, or Fashion-MNIST with official test isolation.

    Download defaults to disabled; no implicit network traffic is initiated.
    torchvision must be installed by the caller for real-image experiments.
    """
    if dataset not in {"cifar10", "cifar100", "fashion_mnist"}:
        raise ValueError("dataset must be cifar10, cifar100, or fashion_mnist")
    try:
        from torchvision import datasets
    except ImportError as exc:
        raise RuntimeError("real vision datasets require optional torchvision") from exc
    catalog = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100,
               "fashion_mnist": datasets.FashionMNIST}
    loader = catalog[dataset]
    train = loader(root=str(root), train=True, download=download)
    test = loader(root=str(root), train=False, download=download)
    train_labels = getattr(train, "targets")
    test_labels = getattr(test, "targets")
    return make_image_partitions(
        train.data, train_labels, test.data, test_labels,
        source="torchvision:" + dataset, revision="torchvision-local-dataset",
        seed=seed, validation_fraction=validation_fraction,
        train_limit=train_limit, validation_limit=validation_limit, test_limit=test_limit,
    )
