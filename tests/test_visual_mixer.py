"""V07 authentic token/channel MLP-Mixer sites and signed ADP resume."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from simple_adp.config import SearchSettings
from simple_adp.visual_mixer import MixerSpec, VisualMixer
from simple_adp.visual_native_ffn import build_visual_composite, restore_visual_composite
from simple_adp.visual_search import VisualResidualSearch


def spec(site):
    return MixerSpec(
        input_channels=3, local_dim=4, classes=2,
        image_height=8, image_width=12, patch_size=2,
        site_id=site, fixed_seed=19, fixed_other_width=5,
    )


def data():
    rng = np.random.default_rng(31)
    x = rng.normal(size=(20, 3, 8, 12)).astype(np.float32) / 4
    y = np.asarray([0, 1] * 10)
    x[y == 1] += 0.2
    return x[:12], y[:12], (x[12:], y[12:])


def settings():
    return SearchSettings(
        device="cpu", seed=48, init_width=3, init_depth=1,
        max_width=5, max_depth=2, max_neurons=45, max_candidates=1,
        max_epochs=1, batch_size=4, ex_k_width=1,
        patience_es=1, patience_width_exp=1, patience_depth_exp=1,
    )


@pytest.mark.parametrize("site", ["token:block0", "channel:block0"])
def test_mixer_dual_real_mlp_forward_gradient_and_frozen_opposite(site):
    model = VisualMixer(spec(site), (3,))
    shape = (4, 3, 8, 12)
    frozen_before = model.frozen_state_sha256()
    assert model.local_mlp.input_dim == model.local_mlp.output_dim == spec(site).target_dim
    assert spec(site).tokens == 24 and spec(site).local_dim == 4
    assert model.growth_targets == {site: model.local_mlp}
    assert all(not p.requires_grad for p in model.other_mlp.parameters())
    assert all(p.requires_grad for p in model.local_mlp.parameters())
    assert any(isinstance(m, nn.GELU) for m in model.other_mlp.modules())
    assert any(isinstance(m, nn.GELU) for m in model.local_mlp.modules())
    out = model(torch.randn(shape))
    assert out.shape == (4, 2)
    out.square().mean().backward()
    assert model.local_mlp.output_layer.weight.grad is not None
    assert model.local_mlp.output_layer.weight.grad.abs().sum().item() > 0
    assert all(p.grad is None for p in model.other_mlp.parameters())
    assert model.frozen_state_sha256() == frozen_before
    with pytest.raises(ValueError, match="NCHW"):
        model(torch.randn(4, 3, 9, 12))
    with pytest.raises(ValueError, match="growth site"):
        model.grow_width(site_id="not:the:site", operator="all_layers",
                         increment=1, max_width=6)


@pytest.mark.parametrize("site", ["token:block0", "channel:block0"])
def test_mixer_real_width_depth_and_strict_snapshot(site):
    model = build_visual_composite(spec(site), (3, 3))
    before = model.frozen_state_sha256()
    weight = model.local_mlp.hidden_layers[0].weight.detach().clone()
    wider = model.grow_width(site_id=site, operator="round_robin", increment=1, max_width=6)
    assert wider is not None
    assert wider.hidden_widths == [4, 3]
    assert torch.equal(wider.local_mlp.hidden_layers[0].weight[:3], weight)
    assert wider.frozen_state_sha256() == before
    all_layers = model.grow_width(site_id=site, operator="all_layers", increment=1, max_width=6)
    assert all_layers is not None and all_layers.hidden_widths == [4, 4]
    den = model.grow_width(site_id=site, operator="den_round_robin", increment=1, max_width=6)
    assert den is not None and den.hidden_widths == [4, 3]
    depth1 = model.grow_depth(site_id=site, operator="D1", max_depth=3)
    depthcur = model.grow_depth(site_id=site, operator="Dcur", max_depth=3)
    assert depth1 is not None and depth1.hidden_widths == [3, 3, 1]
    assert depthcur is not None and depthcur.hidden_widths == [3, 3, 3]
    assert all(any(isinstance(m, nn.GELU) for m in it.local_mlp.modules())
               for it in (wider, all_layers, den, depth1, depthcur))
    rng = torch.get_rng_state().clone()
    restored = restore_visual_composite(wider.snapshot(cursor=wider._width_cursor))
    assert torch.equal(rng, torch.get_rng_state())
    assert restored.spec == wider.spec
    assert restored.hidden_widths == wider.hidden_widths
    x = torch.randn(2, 3, 8, 12)
    with torch.no_grad():
        assert torch.equal(wider.eval()(x), restored.eval()(x))
    with torch.no_grad():
        restored.other_mlp.output_layer.weight.add_(1)
    with pytest.raises(ValueError, match="frozen"):
        restored.verify_frozen()


@pytest.mark.parametrize("site", ["token:block0", "channel:block0"])
def test_mixer_exact_mid_batch_resume_and_data_site_identity(site, tmp_path):
    x, y, validation = data()
    baseline = VisualResidualSearch("ADP-W-RR", spec(site), settings=settings())
    baseline.fit(x, y, validation_data=validation)
    assert [r["architecture"] for r in baseline.history_] == [[3], [4]]
    assert baseline.history_[1]["site"] == site
    assert baseline.model_.frozen_state_sha256() == baseline.model_._frozen_baseline_sha256
    checkpoint = tmp_path / "mixer.ckpt"

    class Evicted(Exception):
        pass

    def evict(record):
        if (record["phase"] == "width" and record["stage_status"] == "active"
                and record["batches_done"] == 1):
            raise Evicted()
    with pytest.raises(Evicted):
        VisualResidualSearch("ADP-W-RR", spec(site), settings=settings()).fit(
            x, y, validation_data=validation,
            checkpoint_path=checkpoint, progress_callback=evict)
    resumed = VisualResidualSearch("ADP-W-RR", spec(site), settings=settings())
    resumed.fit(x, y, validation_data=validation, checkpoint_path=checkpoint)
    assert resumed.history_ == baseline.history_
    assert resumed.result_.score == baseline.result_.score
    assert np.array_equal(resumed.predict_proba(validation[0]),
                          baseline.predict_proba(validation[0]))
    other = "channel:block0" if site == "token:block0" else "token:block0"
    with pytest.raises(ValueError, match="signature"):
        VisualResidualSearch("ADP-W-RR", spec(other), settings=settings()).fit(
            x, y, validation_data=validation, checkpoint_path=checkpoint)


@pytest.mark.parametrize("site", ["token:block0", "channel:block0"])
def test_mixer_generic_and_fixed_composite_controllers(site):
    x, y, validation = data()
    fixed = VisualResidualSearch(
        "MLP-Fixed", spec(site), settings=settings(),
        fixed_grid=((3,), (4,)),
    )
    fixed.fit(x, y, validation_data=validation)
    assert [r["architecture"] for r in fixed.history_] == [[3], [4]]
    generic = VisualResidualSearch(
        "ADP-DW-Dcur-All", spec(site), settings=settings(),
        generic_profile="corrected",
    )
    generic.fit(x, y, validation_data=validation)
    assert generic.history_[1]["stage"] == "depth"
    assert generic.history_[1]["profile"] == "corrected"


def test_mixer_site_selection_validation():
    with pytest.raises(ValueError, match="token:block0"):
        spec("invalid")
    with pytest.raises(ValueError, match="divide"):
        MixerSpec(3, 4, 2, 7, 12, 2, site_id="token:block0")



def test_mixer_explicit_both_scope_grows_two_real_internal_mlps():
    both = spec("both:block0")
    model = VisualMixer(both, (3, 3))
    frozen = model.frozen_state_sha256()
    assert model.local_mlp.input_dim == model.local_mlp.output_dim == both.tokens
    assert model.other_mlp.input_dim == model.other_mlp.output_dim == both.local_dim
    assert set(model.growth_targets) == {"token:block0", "channel:block0"}
    assert all(p.requires_grad for p in model.local_mlp.parameters())
    assert all(p.requires_grad for p in model.other_mlp.parameters())
    logits = model(torch.randn(4, 3, 8, 12))
    logits.square().mean().backward()
    assert model.local_mlp.output_layer.weight.grad.abs().sum() > 0
    assert model.other_mlp.output_layer.weight.grad.abs().sum() > 0
    assert all(p.grad is None for m in model._fixed_modules() for p in m.parameters())
    assert model.frozen_state_sha256() == frozen

    grown = model.grow_width(site_id="both:block0", operator="round_robin",
                              increment=1, max_width=6)
    assert grown is not None
    assert grown.hidden_widths == [4, 3]
    assert grown.other_mlp.hidden_widths == [4, 3]
    assert grown._width_cursor == 1
    assert getattr(grown.other_mlp, "_width_cursor") == 1
    assert grown.frozen_state_sha256() == frozen
    assert sum(p.numel() for p in grown.parameters()) > sum(p.numel() for p in model.parameters())
    restored = restore_visual_composite(grown.snapshot(cursor=grown._width_cursor))
    assert restored.spec == grown.spec
    assert getattr(restored.other_mlp, "_width_cursor") == 1
    further = restored.grow_width(site_id="both:block0", operator="round_robin",
                                  increment=1, max_width=6)
    assert further is not None
    assert further.hidden_widths == [4, 4]
    assert further.other_mlp.hidden_widths == [4, 4]
    all_layers = model.grow_width(site_id="both:block0", operator="all_layers",
                                  increment=1, max_width=6)
    assert all_layers is not None
    assert all_layers.hidden_widths == all_layers.other_mlp.hidden_widths == [4, 4]
    d1 = model.grow_depth(site_id="both:block0", operator="D1", max_depth=3)
    dcur = model.grow_depth(site_id="both:block0", operator="Dcur", max_depth=3)
    assert d1 is not None and d1.hidden_widths == d1.other_mlp.hidden_widths == [3, 3, 1]
    assert dcur is not None and dcur.hidden_widths == dcur.other_mlp.hidden_widths == [3, 3, 3]
    with pytest.raises(ValueError, match="both:block0"):
        model.grow_width(site_id="token:block0", operator="all_layers",
                         increment=1, max_width=5)


def test_mixer_both_scope_signed_resume_and_joint_budget(tmp_path):
    x, y, validation = data()
    both = spec("both:block0")
    baseline = VisualResidualSearch("ADP-W-RR", both, settings=settings())
    baseline.fit(x, y, validation_data=validation)
    assert [r["architecture"] for r in baseline.history_] == [[3], [4]]
    assert baseline.search_state_["current"]["other_widths"] == [4]
    checkpoint = tmp_path / "both-scope.ckpt"

    class Evicted(Exception):
        pass

    def evict(progress):
        if (progress["phase"] == "width" and progress["stage_status"] == "active"
                and progress["batches_done"] == 1):
            raise Evicted()

    with pytest.raises(Evicted):
        VisualResidualSearch("ADP-W-RR", both, settings=settings()).fit(
            x, y, validation_data=validation, checkpoint_path=checkpoint,
            progress_callback=evict,
        )
    restored = VisualResidualSearch("ADP-W-RR", both, settings=settings())
    restored.fit(x, y, validation_data=validation, checkpoint_path=checkpoint)
    assert restored.history_ == baseline.history_
    assert restored.result_.score == baseline.result_.score
    assert np.array_equal(restored.predict_proba(validation[0]),
                          baseline.predict_proba(validation[0]))
    with pytest.raises(ValueError, match="signature"):
        VisualResidualSearch("ADP-W-RR", spec("token:block0"),
                             settings=settings()).fit(
            x, y, validation_data=validation, checkpoint_path=checkpoint,
        )
    cap = SearchSettings(**{**settings().as_dict(), "max_neurons": 35})
    constrained = VisualResidualSearch("ADP-W-RR", both, settings=cap)
    constrained.fit(x, y, validation_data=validation)
    # The source counts hidden + output units per site: 27+7=34 at
    # initialization, then 28+8=36 after one synchronized growth.
    assert len(constrained.history_) == 1


def test_mixer_both_scope_progressive_depth_warmup_stays_synchronized():
    x, y, validation = data()
    config = SearchSettings(**{**settings().as_dict(), "max_neurons": 60})
    search = VisualResidualSearch("ADP-D-D1", spec("both:block0"), settings=config)
    trace = []
    search.fit(x, y, validation_data=validation,
               progress_callback=lambda state: trace.append(
                   (state["phase"], tuple(state["architecture"]))))
    assert ("depth_warmup", (3, 1)) in trace
    assert ("depth_warmup", (3, 2)) in trace
    assert search.history_[-1]["architecture"] == [3, 3]
    assert search.search_state_["current"]["other_widths"] == [3, 3]
    assert all(entry["stage"] != "depth_warmup" for entry in search.history_)
