"""V02 signed dedicated whole-image controller, candidate and restart tests."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from simple_adp.config import SearchSettings
from simple_adp.visual_inner import ResidualAdapterSpec
from simple_adp.visual_search import SUPPORTED_V02_METHODS, VisualResidualSearch


def samples():
    rng = np.random.default_rng(112)
    images = rng.normal(size=(24, 3, 8, 8)).astype(np.float32) / 3.0
    labels = np.asarray([0, 1] * 12, dtype=np.int64)
    images[labels == 1] += 0.4
    return images[:16], labels[:16], (images[16:], labels[16:])


def spec():
    return ResidualAdapterSpec(
        input_channels=3, backbone_channels=4, local_dim=3,
        classes=2, image_height=8, image_width=8,
        site_id="adapter:block1",
    )


def settings(seed=7):
    return SearchSettings(
        seed=seed, device="cpu", init_width=2, init_depth=1,
        max_width=4, max_depth=2, max_neurons=12,
        ex_k_width=1, max_candidates=1, max_epochs=1,
        batch_size=4, patience_es=1,
        patience_width_exp=1, patience_depth_exp=1,
    )


@pytest.mark.parametrize("method", sorted(SUPPORTED_V02_METHODS))
def test_v02_supported_methods_execute_real_composite_selection(method):
    x, y, validation = samples()
    kwargs = {"fixed_grid": ((2,), (3,))} if method == "MLP-Fixed" else {}
    search = VisualResidualSearch(method, spec(), settings=settings(), **kwargs)
    search.fit(x, y, validation_data=validation)
    assert search.model_ is not None
    assert search.result_ is not None
    assert len(search.history_) >= 1
    assert all(r["site"] == "adapter:block1" for r in search.history_)
    assert search.model_.frozen_state_sha256() == search.model_._frozen_baseline_sha256
    assert all(m.training is False for m in search.model_._fixed_modules())
    assert search.predict_proba(validation[0]).shape == (8, 2)
    assert len(search.history_) == 2
    assert search.history_[0]["accepted"] is True
    if method == "MLP-Fixed":
        assert [r["architecture"] for r in search.history_] == [[2], [3]]
        assert all(r["stage"] == "fixed" for r in search.history_)
    elif method in {"ADP-D-Dcur", "ADP-D-D1"}:
        assert search.history_[1]["architecture"] == [2, 2]
        assert search.history_[1]["stage"] == "depth"
    else:
        assert search.history_[1]["stage"] == "width"
        assert sum(search.history_[1]["architecture"]) > sum(search.history_[0]["architecture"])


def test_v02_rejects_unimplemented_canonical_methods():
    with pytest.raises(NotImplementedError, match="composite-controller"):
        VisualResidualSearch("ADP-WD-RR-D1", spec(), settings=settings())


def test_v02_exact_batch_resume_and_completed_idempotence(tmp_path):
    x, y, valid = samples()
    baseline = VisualResidualSearch("ADP-W-RR", spec(), settings=settings(seed=23))
    baseline.fit(x, y, validation_data=valid)
    path = tmp_path / "inner-v02.ckpt"
    class SimulatedEviction(Exception):
        pass
    interrupted = VisualResidualSearch("ADP-W-RR", spec(), settings=settings(seed=23))
    def evict(record):
        if (record["phase"] == "width"
                and record["stage_status"] == "active"
                and record["batches_done"] >= 1):
            raise SimulatedEviction()
    with pytest.raises(SimulatedEviction):
        interrupted.fit(x, y, validation_data=valid, checkpoint_path=path,
                        progress_callback=evict)
    assert path.is_file()
    resumed = VisualResidualSearch("ADP-W-RR", spec(), settings=settings(seed=23))
    resumed.fit(x, y, validation_data=valid, checkpoint_path=path)
    assert resumed.history_ == baseline.history_
    assert resumed.result_.score == baseline.result_.score
    assert np.array_equal(resumed.predict_proba(valid[0]), baseline.predict_proba(valid[0]))
    # An already completed search is safe to open without retraining.
    reopened = VisualResidualSearch("ADP-W-RR", spec(), settings=settings(seed=23))
    reopened.fit(x, y, validation_data=valid, checkpoint_path=path)
    assert reopened.history_ == baseline.history_
    with pytest.raises(ValueError, match="signature"):
        VisualResidualSearch("ADP-W-RR", spec(), settings=settings(seed=24)).fit(
            x, y, validation_data=valid, checkpoint_path=path)
    with pytest.raises(ValueError, match="signature"):
        damaged = x.copy()
        damaged[0, 0, 0, 0] += 0.25
        VisualResidualSearch("ADP-W-RR", spec(), settings=settings(seed=23)).fit(
            damaged, y, validation_data=valid, checkpoint_path=path)


def test_v02_forces_output_contract_and_fixed_grid():
    x, y, validation = samples()
    with pytest.raises(ValueError, match="positive nonempty"):
        VisualResidualSearch("MLP-Fixed", spec(), settings=settings(), fixed_grid=[[]])
    with pytest.raises(ValueError, match="only for MLP-Fixed"):
        VisualResidualSearch("ADP-W-RR", spec(), settings=settings(), fixed_grid=[[2]])
    bad_spec = ResidualAdapterSpec(3, 4, 3, 3, 8, 8)
    with pytest.raises(ValueError, match="number of training classes"):
        VisualResidualSearch("ADP-W-RR", bad_spec, settings=settings()).fit(
            x, y, validation_data=validation)


def test_v02_progressive_d1_intermediates_and_exact_mid_warmup_resume(tmp_path):
    x, y, valid = samples()
    config = SearchSettings(**{
        **settings(seed=41).as_dict(), "init_width": 3,
        "max_width": 4, "max_depth": 2, "max_neurons": 30,
    })
    baseline_progress = []
    baseline = VisualResidualSearch("ADP-D-D1", spec(), settings=config)
    baseline.fit(x, y, validation_data=valid,
                 progress_callback=lambda state: baseline_progress.append(
                     (state["phase"], tuple(state["architecture"]), state["batches_done"])))
    assert len(baseline.history_) == 2
    assert baseline.history_[-1]["architecture"] == [3, 3]
    assert {("depth_warmup", (3, 1)), ("depth_warmup", (3, 2))}.issubset({
        (phase, widths) for phase, widths, _ in baseline_progress
    })
    assert ("depth", (3, 3)) in {(phase, widths) for phase, widths, _ in baseline_progress}
    # Intermediate widths must not be mistaken for independent accepted depths.
    assert all(x["stage"] != "depth_warmup" for x in baseline.history_)
    path = tmp_path / "d1-progressive.pt"

    class Evicted(Exception):
        pass
    def interrupt(record):
        if (record["phase"] == "depth_warmup"
                and record["architecture"] == [3, 2]
                and record["batches_done"] == 1
                and record["stage_status"] == "active"):
            raise Evicted()
    with pytest.raises(Evicted):
        VisualResidualSearch("ADP-D-D1", spec(), settings=config).fit(
            x, y, validation_data=valid,
            checkpoint_path=path, progress_callback=interrupt,
        )
    resumed = VisualResidualSearch("ADP-D-D1", spec(), settings=config)
    resumed.fit(x, y, validation_data=valid, checkpoint_path=path)
    assert resumed.history_ == baseline.history_
    assert resumed.result_.score == baseline.result_.score
    assert np.array_equal(resumed.predict_proba(valid[0]), baseline.predict_proba(valid[0]))
