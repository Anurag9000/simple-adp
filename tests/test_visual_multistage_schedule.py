"""V08 scheduled early/middle/late proposal growth with site-local source semantics."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from simple_adp.config import SearchSettings
from simple_adp.visual_multistage_schedule import ScheduledMultistageSpec, VisualMultistageScheduled
from simple_adp.visual_native_ffn import build_visual_composite, restore_visual_composite
from simple_adp.visual_search import VisualResidualSearch


def spec():
    return ScheduledMultistageSpec(
        3, 4, 2, 8, 12, fixed_seed=53,
    )


def settings():
    return SearchSettings(
        init_width=3, init_depth=1, max_width=6, max_depth=2,
        max_neurons=70, max_candidates=3, max_epochs=1,
        batch_size=4, ex_k_width=1,
        patience_es=1, patience_width_exp=5, patience_depth_exp=5,
        device="cpu", seed=71,
    )


def data():
    rng = np.random.default_rng(78)
    x = rng.normal(size=(20, 3, 8, 12)).astype(np.float32) / 4
    y = np.asarray([0, 1] * 10, dtype=np.int64)
    x[y == 1] += 0.17
    return x[:12], y[:12], (x[12:], y[12:])


def test_v08_scheduled_sites_are_real_separate_mlps_and_one_active_per_stage():
    model = build_visual_composite(spec(), (3, 3))
    assert isinstance(model, VisualMultistageScheduled)
    assert set(model.growth_targets) == {"adapter:early", "adapter:middle", "adapter:late"}
    assert {s: m.hidden_widths for s, m in model.site_mlps.items()} == {
        s: [3, 3] for s in ("early", "middle", "late")
    }
    frozen = model.frozen_state_sha256()
    assert model._active == "early"
    model(torch.randn(4, 3, 8, 12)).square().mean().backward()
    assert model.site_mlps["early"].output_layer.weight.grad.abs().sum() > 0
    assert all(p.grad is None for s in ("middle", "late")
               for p in model.site_mlps[s].parameters())
    assert model.frozen_state_sha256() == frozen
    model.activate("middle")
    model.zero_grad(set_to_none=True)
    model(torch.randn(4, 3, 8, 12)).square().mean().backward()
    assert model.site_mlps["middle"].output_layer.weight.grad.abs().sum() > 0
    assert all(p.grad is None for s in ("early", "late")
               for p in model.site_mlps[s].parameters())
    assert all(not p.requires_grad for m in model._fixed_modules() for p in m.parameters())
    assert model.frozen_state_sha256() == frozen
    with pytest.raises(ValueError, match="undeclared"):
        model.activate("fourth")
    with pytest.raises(ValueError, match="NCHW"):
        model(torch.randn(2, 3, 8, 11))


def test_v08_scheduled_per_site_width_depth_cursor_and_strict_snapshot():
    model = VisualMultistageScheduled(spec(), (3, 3))
    base = model.frozen_state_sha256()
    grown = model.grow_width(site_id=spec().site_id, operator="round_robin",
                              increment=1, max_width=6)
    assert grown is not None
    assert grown.site_mlps["early"].hidden_widths == [4, 3]
    assert grown.site_mlps["middle"].hidden_widths == [3, 3]
    assert grown.site_mlps["late"].hidden_widths == [3, 3]
    assert grown.site_mlps["early"]._width_cursor == 1
    grown.activate("middle")
    grown2 = grown.grow_width(site_id=spec().site_id, operator="round_robin",
                               increment=1, max_width=6)
    assert grown2 is not None
    assert grown2.site_mlps["middle"].hidden_widths == [4, 3]
    assert grown2.site_mlps["early"].hidden_widths == [4, 3]
    assert grown2.site_mlps["early"]._width_cursor == 1
    grown2.activate("late")
    d1 = grown2.grow_depth(site_id=spec().site_id, operator="D1", max_depth=3)
    dcur = grown2.grow_depth(site_id=spec().site_id, operator="Dcur", max_depth=3)
    assert d1 is not None and d1.site_mlps["late"].hidden_widths == [3, 3, 1]
    assert dcur is not None and dcur.site_mlps["late"].hidden_widths == [3, 3, 3]
    all_layers = grown2.grow_width(site_id=spec().site_id, operator="all_layers",
                                    increment=1, max_width=6)
    assert all_layers is not None and all_layers.site_mlps["late"].hidden_widths == [4, 4]
    assert all_layers.site_mlps["early"].hidden_widths == [4, 3]
    assert all_layers.site_mlps["middle"].hidden_widths == [4, 3]
    assert all(any(isinstance(mlp_layer, nn.GELU) for mlp_layer in model.site_mlps[s].modules())
               for s in ("early", "middle", "late"))
    rng = torch.get_rng_state().clone()
    recovered = restore_visual_composite(grown2.snapshot(cursor=grown2._width_cursor))
    assert torch.equal(rng, torch.get_rng_state())
    assert recovered._active == "late"
    assert recovered.site_mlps["early"]._width_cursor == 1
    assert recovered.site_mlps["middle"]._width_cursor == 1
    assert recovered.frozen_state_sha256() == base
    x = torch.randn(3, 3, 8, 12)
    with torch.no_grad():
        assert torch.equal(recovered.eval()(x), grown2.eval()(x))
        recovered.stem.weight.add_(1)
    with pytest.raises(ValueError, match="frozen"):
        recovered.verify_frozen()


def test_v08_fixed_schedule_three_proposals_and_mid_middle_batch_resume(tmp_path):
    x, y, valid = data()
    baseline = VisualResidualSearch("ADP-W-RR", spec(), settings=settings())
    baseline.fit(x, y, validation_data=valid)
    assert len(baseline.history_) == 4
    assert [r["site"] for r in baseline.history_] == [
        "adapter:early", "adapter:early", "adapter:middle", "adapter:late",
    ]
    assert baseline.search_state_["current"]["site_widths"] == {
        "early": [4], "middle": [4], "late": [4],
    }
    assert baseline.search_state_["current"]["site_cursors"] == {
        "early": 0, "middle": 0, "late": 0,
    }
    checkpoint = tmp_path / "v08-site-rr.ckpt"

    class Evicted(Exception):
        pass

    def interrupt(row):
        if (row["growth_site"] == "adapter:middle"
                and row["stage_status"] == "active" and row["batches_done"] == 1):
            raise Evicted()

    with pytest.raises(Evicted):
        VisualResidualSearch("ADP-W-RR", spec(), settings=settings()).fit(
            x, y, validation_data=valid, checkpoint_path=checkpoint,
            progress_callback=interrupt,
        )
    resumed = VisualResidualSearch("ADP-W-RR", spec(), settings=settings())
    resumed.fit(x, y, validation_data=valid, checkpoint_path=checkpoint)
    assert resumed.history_ == baseline.history_
    assert resumed.result_.score == baseline.result_.score
    assert resumed.search_state_["current"]["site_widths"] == baseline.search_state_["current"]["site_widths"]
    assert np.array_equal(resumed.predict_proba(valid[0]), baseline.predict_proba(valid[0]))
    with pytest.raises(ValueError, match="signature"):
        altered = SearchSettings(**{**settings().as_dict(), "seed": 72})
        VisualResidualSearch("ADP-W-RR", spec(), settings=altered).fit(
            x, y, validation_data=valid, checkpoint_path=checkpoint,
        )


def test_v08_fixed_site_schedule_joint_budget_applies_to_all_three_stacks():
    x, y, valid = data()
    config = SearchSettings(**{**settings().as_dict(), "max_neurons": 21})
    search = VisualResidualSearch("ADP-W-RR", spec(), settings=config)
    search.fit(x, y, validation_data=valid)
    assert len(search.history_) == 1
    assert search.search_state_["proposals"] == 0


def test_v08_scheduled_fixed_and_generic_routes():
    x, y, valid = data()
    fixed = VisualResidualSearch(
        "MLP-Fixed", spec(), settings=settings(), fixed_grid=((3,), (4,)),
    )
    fixed.fit(x, y, validation_data=valid)
    assert [r["architecture"] for r in fixed.history_] == [[3], [4]]
    generic = VisualResidualSearch(
        "ADP-WD-RR-Dcur", spec(), settings=settings(),
        generic_profile="corrected",
    )
    generic.fit(x, y, validation_data=valid)
    assert len(generic.history_) >= 2
    assert generic.history_[1]["profile"] == "corrected"


def test_v08_schedule_rejects_bad_order_and_site():
    with pytest.raises(ValueError, match="order"):
        ScheduledMultistageSpec(3, 4, 2, 8, 12, site_order=("late", "middle", "early"))
    with pytest.raises(ValueError, match="scheduled site"):
        ScheduledMultistageSpec(3, 4, 2, 8, 12, site_id="adapter:unknown")
