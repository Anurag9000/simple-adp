"""Original/variance child specs must be immutable and race-safe."""
import concurrent.futures
import json

import pytest

from simple_adp.runner.spec_integrity import publish_immutable_run_spec


def _document(criterion="loss", seed=7):
    return json.dumps({
        "identity": {"run_id": "job-A", "seed": seed},
        "worker_settings": {"expansion_criterion": criterion},
        "result_dir": "/results/job-A",
    }, sort_keys=True, indent=2) + "\n"


def test_spec_can_be_published_once_and_reused_without_replacing(tmp_path):
    path = tmp_path / "jobs" / "run_spec.json"
    text = _document()
    publish_immutable_run_spec(path, text)
    original_inode = path.stat().st_ino
    publish_immutable_run_spec(path, text)
    assert path.read_text() == text
    assert path.stat().st_ino == original_inode
    assert not list(path.parent.glob(".run-spec-*.tmp"))


@pytest.mark.parametrize("modified", [_document("variance_stagnation"), _document(seed=8)])
def test_existing_spec_refuses_changed_trigger_or_seed(tmp_path, modified):
    path = tmp_path / "run_spec.json"
    publish_immutable_run_spec(path, _document())
    with pytest.raises(ValueError, match="differs from planned science"):
        publish_immutable_run_spec(path, modified)
    assert path.read_text() == _document()


def test_corrupt_existing_spec_never_overwritten(tmp_path):
    path = tmp_path / "run_spec.json"
    path.write_text("{bad json")
    with pytest.raises(ValueError, match="unreadable or corrupt"):
        publish_immutable_run_spec(path, _document())
    assert path.read_text() == "{bad json"


def test_symlink_target_is_not_followed(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text(_document())
    path = tmp_path / "run_spec.json"
    try:
        path.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="symlink"):
        publish_immutable_run_spec(path, _document())
    assert outside.read_text() == _document()


def test_two_competing_specs_can_never_both_publish(tmp_path):
    path = tmp_path / "jobs" / "run_spec.json"
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda s: _attempt(path, s),
                                 (_document(), _document("variance_stagnation"))))
    assert sorted(outcomes) == ["collision", "published"]
    assert path.read_text() in (_document(), _document("variance_stagnation"))
    assert not list(path.parent.glob(".run-spec-*.tmp"))


def _attempt(path, document):
    try:
        publish_immutable_run_spec(path, document)
        return "published"
    except ValueError as exc:
        assert "differs from planned science" in str(exc)
        return "collision"


def test_existing_spec_refuses_even_bool_integer_alias(tmp_path):
    path = tmp_path / "run_spec.json"
    original = json.dumps({"worker_settings": {"cpu_only": False}})
    altered = json.dumps({"worker_settings": {"cpu_only": 0}})
    publish_immutable_run_spec(path, original)
    with pytest.raises(ValueError, match="differs from planned science"):
        publish_immutable_run_spec(path, altered)
    assert path.read_text() == original
