"""V08 three-stage fixed-site residual adapter source growth and exact resume."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from simple_adp.config import SearchSettings
from simple_adp.visual_multistage import MultistageSpec, VisualMultistage
from simple_adp.visual_native_ffn import build_visual_composite, restore_visual_composite
from simple_adp.visual_search import VisualResidualSearch


def spec(site: str) -> MultistageSpec:
    return MultistageSpec(
        input_channels=3, backbone_channels=4, classes=2,
        image_height=8, image_width=12,
        site_id=f"adapter:{site}", fixed_seed=21,
        fixed_other_width=5,
    )


def images():
    rng = np.random.default_rng(83)
    x = rng.normal(size=(20, 3, 8, 12)).astype(np.float32) / 5
    y = np.asarray([0, 1] * 10, dtype=np.int64)
    x[y == 1] += 0.18
    return x[:12], y[:12], (x[12:], y[12:])


def settings():
    return SearchSettings(
        init_width=3, init_depth=1, max_width=5, max_depth=2,
        max_neurons=30, max_epochs=1, max_candidates=1,
        batch_size=4, ex_k_width=1,
        patience_es=1, patience_width_exp=1, patience_depth_exp=1,
        device="cpu", seed=67,
    )


@pytest.mark.parametrize("site", ["early", "middle", "late"])
def test_v08_genuine_frozen_multistage_mlp_at_declared_site(site):
    model = VisualMultistage(spec(site), (3,))
    assert list(model.fixed_adapters) == [x for x in ("early", "middle", "late") if x != site]
    assert len(model.stage_convs) == 3
    assert model.growth_targets == {f"adapter:{site}": model.local_mlp}
    sha = model.frozen_state_sha256()
    model.train()
    assert all(not module.training for module in model._fixed_modules())
    x = torch.randn(4, 3, 8, 12)
    logits = model(x)
    assert logits.shape == (4, 2)
    logits.square().mean().backward()
    assert model.local_mlp.output_layer.weight.grad is not None
    assert model.local_mlp.output_layer.weight.grad.abs().sum() > 0
    assert all(p.grad is None for m in model._fixed_modules() for p in m.parameters())
    assert model.frozen_state_sha256() == sha
    assert any(isinstance(a, nn.GELU) for a in model.local_mlp.hidden_layers)
    with pytest.raises(ValueError, match="resolution"):
        model(torch.randn(2, 3, 8, 13))


@pytest.mark.parametrize("site", ["early", "middle", "late"])
def test_v08_source_growth_weights_interfaces_and_restoration(site):
    model = build_visual_composite(spec(site), (3, 3))
    sha = model.frozen_state_sha256()
    weight = model.local_mlp.hidden_layers[0].weight.detach().clone()
    rr = model.grow_width(site_id=f"adapter:{site}", operator="round_robin",
                          increment=1, max_width=6)
    assert rr is not None
    assert rr.hidden_widths == [4, 3]
    assert torch.equal(rr.local_mlp.hidden_layers[0].weight[:3], weight)
    assert rr.frozen_state_sha256() == sha
    assert rr.input_dim == rr.output_dim == spec(site).backbone_channels
    all_layers = model.grow_width(site_id=f"adapter:{site}", operator="all_layers",
                                  increment=1, max_width=6)
    den = model.grow_width(site_id=f"adapter:{site}", operator="den_round_robin",
                            increment=1, max_width=6)
    assert all_layers is not None and all_layers.hidden_widths == [4, 4]
    assert den is not None and den.hidden_widths == [4, 3]
    d1 = model.grow_depth(site_id=f"adapter:{site}", operator="D1", max_depth=3)
    dcur = model.grow_depth(site_id=f"adapter:{site}", operator="Dcur", max_depth=3)
    assert d1 is not None and d1.hidden_widths == [3, 3, 1]
    assert dcur is not None and dcur.hidden_widths == [3, 3, 3]
    state = torch.get_rng_state().clone()
    restored = restore_visual_composite(rr.snapshot(cursor=rr._width_cursor))
    assert torch.equal(state, torch.get_rng_state())
    assert restored.spec == rr.spec
    image = torch.randn(2, 3, 8, 12)
    with torch.no_grad():
        assert torch.equal(rr.eval()(image), restored.eval()(image))
    assert restored.frozen_state_sha256() == sha
    with pytest.raises(ValueError, match="growth site"):
        model.grow_width(site_id="adapter:wrong", operator="all_layers",
                         increment=1, max_width=6)
    with torch.no_grad():
        restored.fixed_adapters[next(iter(restored.fixed_adapters))].output_layer.weight.add_(1)
    with pytest.raises(ValueError, match="frozen"):
        restored.verify_frozen()


@pytest.mark.parametrize("site", ["early", "middle", "late"])
def test_v08_single_site_exact_mid_batch_restart_and_signature(site, tmp_path):
    x, y, valid = images()
    baseline = VisualResidualSearch("ADP-W-RR", spec(site), settings=settings())
    baseline.fit(x, y, validation_data=valid)
    assert [r["architecture"] for r in baseline.history_] == [[3], [4]]
    assert baseline.history_[1]["site"] == f"adapter:{site}"
    path = tmp_path / "v08.ckpt"

    class Evicted(Exception):
        pass

    def evict(row):
        if row["phase"] == "width" and row["stage_status"] == "active" and row["batches_done"] == 1:
            raise Evicted()

    with pytest.raises(Evicted):
        VisualResidualSearch("ADP-W-RR", spec(site), settings=settings()).fit(
            x, y, validation_data=valid, checkpoint_path=path,
            progress_callback=evict)
    resumed = VisualResidualSearch("ADP-W-RR", spec(site), settings=settings())
    resumed.fit(x, y, validation_data=valid, checkpoint_path=path)
    assert resumed.history_ == baseline.history_
    assert np.array_equal(resumed.predict_proba(valid[0]),
                          baseline.predict_proba(valid[0]))
    another = next(s for s in ("early", "middle", "late") if s != site)
    with pytest.raises(ValueError, match="signature"):
        VisualResidualSearch("ADP-W-RR", spec(another), settings=settings()).fit(
            x, y, validation_data=valid, checkpoint_path=path)


@pytest.mark.parametrize("site", ["early", "middle", "late"])
def test_v08_fixed_and_generic_controls(site):
    x, y, valid = images()
    fixed = VisualResidualSearch(
        "MLP-Fixed", spec(site), settings=settings(), fixed_grid=((3,), (4,)),
    )
    fixed.fit(x, y, validation_data=valid)
    assert [r["architecture"] for r in fixed.history_] == [[3], [4]]
    generic = VisualResidualSearch(
        "ADP-WD-All-Dcur", spec(site), settings=settings(),
        generic_profile="corrected",
    )
    generic.fit(x, y, validation_data=valid)
    assert generic.history_[1]["profile"] == "corrected"


def test_v08_rejects_undeclared_site_and_zero_dimensions():
    with pytest.raises(ValueError, match="growth site"):
        spec("nonexistent")
    with pytest.raises(ValueError, match="positive"):
        MultistageSpec(3, 4, 2, 0, 12, "adapter:early")
