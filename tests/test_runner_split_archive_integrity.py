"""Boundary-safe split archives while preserving historical run-ID hashes."""
from __future__ import annotations

import concurrent.futures
import numpy as np
import pytest

from simple_adp.runner.plan import SplitSpec, _sha_indices, _sha_indices_v2


def _colliding_legacy_splits():
    first = ((0, 1), (2,), (3,))
    second = ((0,), (1, 2), (3,))
    return (
        SplitSpec(0, 0, 0, *first, _sha_indices(*first)),
        SplitSpec(0, 0, 0, *second, _sha_indices(*second)),
    )


def test_legacy_split_hash_collides_without_boundary_encoding():
    a, b = _colliding_legacy_splits()
    assert a.split_sha256 == b.split_sha256
    assert _sha_indices_v2(a.train_indices, a.validation_indices, a.test_indices) != (
        _sha_indices_v2(b.train_indices, b.validation_indices, b.test_indices)
    )


def test_new_archive_records_v2_partition_digest_and_roundtrips(tmp_path):
    a, _ = _colliding_legacy_splits()
    path = tmp_path / "split.npz"
    a.save(path)
    with np.load(path, allow_pickle=False) as archive:
        assert "split_partition_sha256_v2" in archive.files
    inode = path.stat().st_ino
    a.save(path)
    assert path.stat().st_ino == inode
    assert SplitSpec.load(path, repeat=0, fold=0, sample=0) == a


def test_legacy_hash_collision_cannot_reuse_incorrect_existing_split(tmp_path):
    a, b = _colliding_legacy_splits()
    path = tmp_path / "split.npz"
    a.save(path)
    with pytest.raises(ValueError, match="different partition indices"):
        b.save(path)
    assert SplitSpec.load(path, repeat=0, fold=0, sample=0) == a


def test_v2_digest_detects_partition_shift_with_identical_legacy_hash(tmp_path):
    a, b = _colliding_legacy_splits()
    path = tmp_path / "split.npz"
    a.save(path)
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name].copy() for name in archive.files}
    payload["train"] = np.asarray(b.train_indices, dtype=np.int64)
    payload["validation"] = np.asarray(b.validation_indices, dtype=np.int64)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    with pytest.raises(ValueError, match="partition-boundary"):
        SplitSpec.load(path, repeat=0, fold=0, sample=0)


def test_legacy_archive_remains_readable_but_mismatched_reuse_fails(tmp_path):
    a, b = _colliding_legacy_splits()
    path = tmp_path / "legacy.npz"
    with path.open("wb") as stream:
        np.savez_compressed(
            stream, train=a.train_indices, validation=a.validation_indices,
            test=a.test_indices, split_sha256=a.split_sha256,
        )
    assert SplitSpec.load(path, repeat=0, fold=0, sample=0) == a
    a.save(path)
    with pytest.raises(ValueError, match="different partition indices"):
        b.save(path)


def test_corrupt_split_archive_is_not_overwritten(tmp_path):
    a, _ = _colliding_legacy_splits()
    path = tmp_path / "broken.npz"
    path.write_bytes(b"bad archive")
    with pytest.raises(ValueError, match="integrity"):
        a.save(path)
    assert path.read_bytes() == b"bad archive"


def test_two_competing_boundary_colliding_splits_cannot_both_publish(tmp_path):
    a, b = _colliding_legacy_splits()
    path = tmp_path / "split.npz"

    def attempt(spec):
        try:
            spec.save(path)
            return "published"
        except ValueError:
            return "collision"

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(attempt, (a, b)))
    assert sorted(outcomes) == ["collision", "published"]
    assert SplitSpec.load(path, repeat=0, fold=0, sample=0) in (a, b)
    assert not list(tmp_path.glob(".split-*.tmp"))


def test_duplicate_row_indices_are_rejected_before_publication(tmp_path):
    repeated = ((0, 0), (1,), (2,))
    split = SplitSpec(0, 0, 0, *repeated, _sha_indices(*repeated))
    with pytest.raises(ValueError, match="disjoint"):
        split.save(tmp_path / "split.npz")
    assert not (tmp_path / "split.npz").exists()
