import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from simple_adp.benchmark import OPENML_SUITES, _sha_indices, make_fold_split


@pytest.mark.parametrize(
    ("suite_key", "labels", "expected_stratify"),
    [
        ("cc18", np.asarray([0, 1] * 5 + [1, 0]), True),
        ("ctr23", np.asarray([0.0, 1.0] * 5 + [1.0, 0.0]), False),
    ],
)
def test_openml_inner_validation_split_preserves_outer_test_boundary(
    suite_key, labels, expected_stratify, monkeypatch,
):
    outer_train = np.arange(10, dtype=np.int64)
    outer_test = np.asarray([10, 11], dtype=np.int64)
    observed = {}

    class FakeOpenMLTask:
        def get_train_test_split_indices(self, *, repeat, fold, sample):
            observed["outer_split"] = (repeat, fold, sample)
            return outer_train.copy(), outer_test.copy()

    def fake_train_test_split(rows, *, test_size, random_state, shuffle, stratify):
        observed.update({
            "rows": np.asarray(rows).copy(),
            "test_size": test_size,
            "random_state": random_state,
            "shuffle": shuffle,
            "stratify": None if stratify is None else np.asarray(stratify).copy(),
        })
        return np.asarray([0, 1, 2, 3, 4, 5, 6, 7]), np.asarray([8, 9])

    sklearn = ModuleType("sklearn")
    model_selection = ModuleType("sklearn.model_selection")
    model_selection.train_test_split = fake_train_test_split
    sklearn.model_selection = model_selection
    monkeypatch.setitem(sys.modules, "sklearn", sklearn)
    monkeypatch.setitem(sys.modules, "sklearn.model_selection", model_selection)

    result = make_fold_split(
        FakeOpenMLTask(),
        pd.Series(labels),
        OPENML_SUITES[suite_key],
        repeat=2,
        fold=3,
        sample=4,
        validation_fraction=0.2,
        split_seed=7919,
    )

    assert observed["outer_split"] == (2, 3, 4)
    np.testing.assert_array_equal(observed["rows"], outer_train)
    assert observed["test_size"] == 0.2
    assert observed["random_state"] == 7919
    assert observed["shuffle"] is True
    if expected_stratify:
        np.testing.assert_array_equal(observed["stratify"], labels[:10])
    else:
        assert observed["stratify"] is None
    np.testing.assert_array_equal(result.train_indices, np.arange(8))
    np.testing.assert_array_equal(result.validation_indices, np.asarray([8, 9]))
    np.testing.assert_array_equal(result.test_indices, outer_test)
    assert set(result.train_indices).isdisjoint(set(result.validation_indices))
    assert set(result.train_indices).isdisjoint(set(result.test_indices))
    assert set(result.validation_indices).isdisjoint(set(result.test_indices))
    assert result.split_sha256 == _sha_indices(
        result.train_indices, result.validation_indices, result.test_indices,
    )
