import json

import numpy as np
import pytest

from simple_adp import SearchSettings, TabularADP
from simple_adp.generic_artifacts import GenericSearchArtifactStore
from simple_adp.tasks import RegressionTask


def _inputs():
    rng = np.random.default_rng(1207)
    x = rng.normal(size=(14, 3)).astype(np.float32)
    y = (x[:, 0] * x[:, 1] - 0.4 * x[:, 2]).astype(np.float32)
    return x[:10], y[:10], x[10:], y[10:]


def _settings(**overrides):
    values = dict(
        init_width=2,
        init_depth=1,
        ex_k_width=1,
        max_width=4,
        max_depth=3,
        max_neurons=64,
        max_epochs=1,
        patience_es=1,
        patience_width_exp=2,
        patience_depth_exp=2,
        max_candidates=2,
        batch_size=5,
        seed=29,
    )
    values.update(overrides)
    return SearchSettings(**values)


def test_generic_search_writes_candidate_method_task_and_replay_artifacts(tmp_path):
    x, y, x_valid, y_valid = _inputs()
    checkpoint = tmp_path / "estimator.pt"
    estimator = TabularADP("ADP-WD-All-D1", task=RegressionTask(), settings=_settings())
    estimator.fit(
        x, y, validation_data=(x_valid, y_valid), checkpoint_path=checkpoint,
    )

    root = tmp_path / "estimator.pt.search"
    manifest = json.loads((root / "manifest.json").read_text())
    method = json.loads((root / "method_state.json").read_text())
    task = json.loads((root / "task_state.json").read_text())
    replay = json.loads((root / "replay_manifest.json").read_text())
    candidates = sorted((root / "candidates").glob("candidate_*/candidate_state.json"))

    assert manifest["signature"]
    assert method["status"] == "completed"
    assert task["status"] == "completed"
    assert replay["status"] == "completed"
    assert len(candidates) == estimator.search_state_["candidate_count"] + 1
    for state_path in candidates:
        state = json.loads(state_path.read_text())
        candidate_dir = state_path.parent
        assert state["status"] == "completed"
        assert state["signature"] == manifest["signature"]
        assert (candidate_dir / "checkpoint_best.pt").is_file()
        assert (candidate_dir / "checkpoint_last.pt").is_file()
        assert (candidate_dir / "search_context.pt").is_file()

    # A completed candidate hierarchy remains independently loadable after the
    # estimator's exact-resume checkpoint is removed.
    checkpoint.unlink()
    recovered = TabularADP("ADP-WD-All-D1", task=RegressionTask(), settings=_settings())
    recovered.fit(x, y, validation_data=(x_valid, y_valid), checkpoint_path=checkpoint)
    assert recovered.history_ == estimator.history_
    np.testing.assert_array_equal(recovered.predict(x_valid), estimator.predict(x_valid))


def test_generic_search_resumes_from_last_completed_candidate_artifact(tmp_path, monkeypatch):
    x, y, x_valid, y_valid = _inputs()
    checkpoint = tmp_path / "interrupted.pt"
    settings = _settings()
    original = GenericSearchArtifactStore.record_progress

    def interrupt_after_first_width_candidate(self, **kwargs):
        if kwargs["context"].get("candidate_count") == 1:
            raise RuntimeError("simulated stop after durable candidate")
        return original(self, **kwargs)

    monkeypatch.setattr(GenericSearchArtifactStore, "record_progress", interrupt_after_first_width_candidate)
    interrupted = TabularADP("ADP-WD-All-D1", task=RegressionTask(), settings=settings)
    with pytest.raises(RuntimeError, match="durable candidate"):
        interrupted.fit(x, y, validation_data=(x_valid, y_valid), checkpoint_path=checkpoint)

    candidate = tmp_path / "interrupted.pt.search" / "candidates" / "candidate_000001"
    assert (candidate / "candidate_state.json").is_file()
    checkpoint.unlink()
    monkeypatch.setattr(GenericSearchArtifactStore, "record_progress", original)

    resumed = TabularADP("ADP-WD-All-D1", task=RegressionTask(), settings=settings)
    resumed.fit(x, y, validation_data=(x_valid, y_valid), checkpoint_path=checkpoint)
    uninterrupted = TabularADP("ADP-WD-All-D1", task=RegressionTask(), settings=settings)
    uninterrupted.fit(x, y, validation_data=(x_valid, y_valid))

    assert resumed.history_ == uninterrupted.history_
    assert resumed.search_state_ == uninterrupted.search_state_
    np.testing.assert_array_equal(resumed.predict(x_valid), uninterrupted.predict(x_valid))


def test_generic_artifact_directory_rejects_changed_data_signature(tmp_path):
    x, y, x_valid, y_valid = _inputs()
    artifact_dir = tmp_path / "search-artifacts"
    settings = _settings()
    TabularADP("ADP-WD-All-D1", task=RegressionTask(), settings=settings).fit(
        x, y, validation_data=(x_valid, y_valid), candidate_artifact_dir=artifact_dir,
    )
    changed_y = y.copy()
    changed_y[0] += 1.0
    with pytest.raises(ValueError, match="artifact signature does not match"):
        TabularADP("ADP-WD-All-D1", task=RegressionTask(), settings=settings).fit(
            x, changed_y, validation_data=(x_valid, y_valid), candidate_artifact_dir=artifact_dir,
        )
    assert (artifact_dir / "manifest_mismatch.json").is_file()


def test_generic_candidate_recovery_rejects_checkpoint_path_escape(tmp_path):
    x, y, x_valid, y_valid = _inputs()
    artifact_dir = tmp_path / "search-artifacts"
    settings = _settings()
    TabularADP("ADP-WD-All-D1", task=RegressionTask(), settings=settings).fit(
        x, y, validation_data=(x_valid, y_valid), candidate_artifact_dir=artifact_dir,
    )
    state_path = sorted((artifact_dir / "candidates").glob("candidate_*/candidate_state.json"))[-1]
    state = json.loads(state_path.read_text())
    state["context_checkpoint"] = "../../outside.pt"
    state_path.write_text(json.dumps(state))
    task_path = artifact_dir / "task_state.json"
    task = json.loads(task_path.read_text())
    task["status"] = "active"
    task_path.write_text(json.dumps(task))
    signature = json.loads((artifact_dir / "manifest.json").read_text())["signature"]

    store = GenericSearchArtifactStore(artifact_dir, signature=signature)
    with pytest.raises(ValueError, match="must be a filename"):
        store.load_resume()


def test_generic_completed_task_recovery_rejects_escaping_symlink(tmp_path):
    x, y, x_valid, y_valid = _inputs()
    artifact_dir = tmp_path / "search-artifacts"
    settings = _settings()
    TabularADP("ADP-WD-All-D1", task=RegressionTask(), settings=settings).fit(
        x, y, validation_data=(x_valid, y_valid), candidate_artifact_dir=artifact_dir,
    )
    outside = tmp_path / "outside.pt"
    outside.write_bytes((artifact_dir / "task_best.pt").read_bytes())
    link = artifact_dir / "outside-link.pt"
    link.symlink_to(outside)
    task_path = artifact_dir / "task_state.json"
    task = json.loads(task_path.read_text())
    task["best_checkpoint"] = link.name
    task_path.write_text(json.dumps(task))
    signature = json.loads((artifact_dir / "manifest.json").read_text())["signature"]

    store = GenericSearchArtifactStore(artifact_dir, signature=signature)
    with pytest.raises(ValueError, match="escapes its directory"):
        store.load_resume()


def test_generic_artifact_schema_mismatch_is_rejected(tmp_path):
    x, y, x_valid, y_valid = _inputs()
    artifact_dir = tmp_path / "search-artifacts"
    TabularADP("ADP-WD-All-D1", task=RegressionTask(), settings=_settings()).fit(
        x, y, validation_data=(x_valid, y_valid), candidate_artifact_dir=artifact_dir,
    )
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema"] = 1
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="artifact signature does not match"):
        GenericSearchArtifactStore(artifact_dir, signature=manifest["signature"])
    assert (artifact_dir / "manifest_mismatch.json").is_file()
