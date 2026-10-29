"""Deterministic vision partition and provenance contract tests (offline)."""
import numpy as np
import pytest

from simple_adp.vision_data import make_image_partitions, load_torchvision_partitions


def _data():
    rng = np.random.default_rng(19)
    train = rng.integers(0, 255, size=(30, 8, 9, 3), dtype=np.uint8)
    labels = np.array([0, 1, 2] * 10, dtype=np.int64)
    test = rng.integers(0, 255, size=(15, 8, 9, 3), dtype=np.uint8)
    test_labels = np.array([0, 1, 2] * 5, dtype=np.int64)
    return train, labels, test, test_labels


def test_stratified_splits_and_manifest_are_exactly_reproducible():
    args = _data()
    a = make_image_partitions(*args, source="toy-image", revision="v1", seed=77,
                              train_limit=12, validation_limit=6, test_limit=9)
    b = make_image_partitions(*args, source="toy-image", revision="v1", seed=77,
                              train_limit=12, validation_limit=6, test_limit=9)
    assert a.manifest == b.manifest
    assert len(a.fit_indices) == 12 and len(a.validation_indices) == 6
    assert len(a.evaluation_indices) == 9
    assert set(a.fit_indices.tolist()).isdisjoint(a.validation_indices.tolist())
    assert set(a.manifest["label_map"]) == {0, 1, 2}
    x, y = a.materialize("train")
    assert x.shape == (12, 3, 8, 9) and x.dtype == np.float32
    assert y.shape == (12,) and 0 <= x.min() <= x.max() <= 1
    for part in ("train", "validation", "test"):
        _, labels = a.materialize(part)
        assert set(labels.tolist()) == {0, 1, 2}


def test_no_test_labels_enter_architecture_split():
    train, labels, test, test_labels = _data()
    a = make_image_partitions(train, labels, test, test_labels,
                              source="x", revision="r", seed=4)
    assert max(a.fit_indices) < len(train)
    assert max(a.validation_indices) < len(train)
    assert max(a.evaluation_indices) < len(test)
    assert len(a.fit_indices) + len(a.validation_indices) == len(train)


def test_reject_unsupported_and_damaged_data(tmp_path):
    train, labels, test, test_labels = _data()
    with pytest.raises(ValueError, match="fraction"):
        make_image_partitions(train, labels, test, test_labels,
                              source="x", revision="r", validation_fraction=0.7)
    with pytest.raises(ValueError, match="integer"):
        make_image_partitions(train, labels.astype(float), test, test_labels,
                              source="x", revision="r")
    with pytest.raises(ValueError, match="class"):
        make_image_partitions(train, labels, test, np.full(len(test_labels), 5),
                              source="x", revision="r")
    with pytest.raises(ValueError, match="sample limit"):
        make_image_partitions(train, labels, test, test_labels,
                              source="x", revision="r", train_limit=2)
    with pytest.raises(ValueError, match="dataset must be"):
        load_torchvision_partitions("nonexistent", tmp_path)
