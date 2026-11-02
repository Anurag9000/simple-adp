"""V03 token-wise ViT / V04 ConvNeXt channel FFN real growth and resume."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from simple_adp.config import SearchSettings
from simple_adp.visual_native_ffn import (
    NativeFFNSpec, VisualNativeFFN, build_visual_composite, restore_visual_composite,
)
from simple_adp.visual_search import VisualResidualSearch


def spec(architecture: str) -> NativeFFNSpec:
    return NativeFFNSpec(
        architecture_id=architecture, input_channels=3, local_dim=4,
        classes=2, image_height=8, image_width=12, patch_size=2,
        attention_heads=2, fixed_seed=19,
        site_id={"V03": "ffn:block0", "V04": "channel:stage0.block0",
                 "V05": "channel:poolformer.block0"}[architecture],
    )


def sample():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(20, 3, 8, 12)).astype(np.float32) / 5
    labels = np.asarray([0, 1] * 10, dtype=np.int64)
    x[labels == 1] += 0.1
    return x[:12], labels[:12], (x[12:], labels[12:])


def settings():
    return SearchSettings(
        init_width=3, init_depth=1, ex_k_width=1,
        max_width=5, max_depth=2, max_neurons=30,
        max_epochs=1, batch_size=4, max_candidates=1,
        patience_es=1, patience_width_exp=1, patience_depth_exp=1,
        device="cpu", seed=31,
    )


@pytest.mark.parametrize("arch", ["V03", "V04", "V05"])
def test_native_ffn_is_inner_and_frozen_backbone_remains_exact(arch):
    torch.manual_seed(103)
    model = VisualNativeFFN(spec(arch), (3,))
    sha = model.frozen_state_sha256()
    x = torch.randn(5, 3, 8, 12)
    assert model(x[:1]).shape == (1, 2)
    assert model(x).shape == (5, 2)
    assert all(isinstance(layer, (nn.Linear, nn.GELU)) for layer in model.local_mlp.hidden_layers)
    model.train()
    assert all(not module.training for module in model._fixed_modules())
    loss = model(x[:4]).square().mean()
    loss.backward()
    grad = model.local_mlp.output_layer.weight.grad
    assert grad is not None and grad.abs().sum() > 0
    assert all(p.grad is None for m in model._fixed_modules() for p in m.parameters())
    assert model.frozen_state_sha256() == sha
    with pytest.raises(ValueError, match="NCHW"):
        model(torch.randn(2, 3, 7, 12))
    with pytest.raises(ValueError, match="growth site"):
        NativeFFNSpec(arch, 3, 4, 2, 8, 12, site_id="adapter:wrong")
    with pytest.raises(ValueError, match="divisible"):
        NativeFFNSpec(arch, 3, 4, 2, 7, 12, site_id=spec(arch).site_id)


@pytest.mark.parametrize("arch", ["V03", "V04", "V05"])
def test_native_ffn_source_growth_and_snapshot_preserve_interfaces(arch):
    model = build_visual_composite(spec(arch), (3, 3))
    before = model.frozen_state_sha256()
    baseline = model.local_mlp.hidden_layers[0].weight.detach().clone()
    wider = model.grow_width(site_id=model.spec.site_id, operator="round_robin",
                             increment=1, max_width=6)
    assert wider is not None and wider.hidden_widths == [4, 3]
    assert torch.equal(wider.local_mlp.hidden_layers[0].weight[:3], baseline)
    assert wider.frozen_state_sha256() == before
    assert wider.input_dim == wider.output_dim == spec(arch).local_dim
    assert wider.spec.token_count == spec(arch).token_count
    assert any(isinstance(x, nn.GELU) for x in wider.local_mlp.hidden_layers)
    all_layers = model.grow_width(site_id=model.spec.site_id, operator="all_layers",
                                  increment=1, max_width=6)
    den_rr = model.grow_width(site_id=model.spec.site_id, operator="den_round_robin",
                              increment=1, max_width=6)
    assert all_layers is not None and all_layers.hidden_widths == [4, 4]
    assert den_rr is not None and den_rr.hidden_widths == [4, 3]
    d1 = model.grow_depth(site_id=model.spec.site_id, operator="D1", max_depth=3)
    dcur = model.grow_depth(site_id=model.spec.site_id, operator="Dcur", max_depth=3)
    assert d1 is not None and d1.hidden_widths == [3, 3, 1]
    assert dcur is not None and dcur.hidden_widths == [3, 3, 3]
    assert any(isinstance(x, nn.GELU) for x in d1.local_mlp.hidden_layers)
    state = torch.get_rng_state().clone()
    restored = restore_visual_composite(wider.snapshot(cursor=wider._width_cursor))
    assert torch.equal(state, torch.get_rng_state())
    assert restored.frozen_state_sha256() == before
    assert restored.hidden_widths == [4, 3]
    images = torch.randn(2, 3, 8, 12)
    with torch.no_grad():
        assert torch.equal(wider.eval()(images), restored.eval()(images))
    assert torch.equal(torch.get_rng_state(), torch.get_rng_state())
    with pytest.raises(ValueError, match="unknown"):
        model.grow_width(site_id="ffn:illegal", operator="all_layers",
                         increment=1, max_width=5)
    with torch.no_grad():
        restored.stem.weight.add_(1)
    with pytest.raises(ValueError, match="frozen"):
        restored.verify_frozen()


@pytest.mark.parametrize("arch", ["V03", "V04", "V05"])
def test_native_ffn_composite_search_actual_growth_and_restart(arch, tmp_path):
    x, y, validation = sample()
    base = VisualResidualSearch("ADP-W-RR", spec(arch), settings=settings())
    base.fit(x, y, validation_data=validation)
    assert len(base.history_) == 2
    assert base.history_[1]["architecture"] == [4]
    assert base.history_[0]["architecture"] == [3]
    assert base.history_[1]["site"] == spec(arch).site_id
    assert np.isfinite(base.predict_proba(validation[0])).all()
    assert base.model_.frozen_state_sha256() == base.model_._frozen_baseline_sha256
    path = tmp_path / f"{arch}-exact.ckpt"
    class Evicted(Exception):
        pass
    def evict(state):
        if (state["phase"] == "width"
                and state["stage_status"] == "active"
                and state["batches_done"] == 1):
            raise Evicted()
    with pytest.raises(Evicted):
        VisualResidualSearch("ADP-W-RR", spec(arch), settings=settings()).fit(
            x, y, validation_data=validation,
            checkpoint_path=path, progress_callback=evict)
    resumed = VisualResidualSearch("ADP-W-RR", spec(arch), settings=settings())
    resumed.fit(x, y, validation_data=validation, checkpoint_path=path)
    assert resumed.history_ == base.history_
    assert np.array_equal(resumed.predict_proba(validation[0]),
                          base.predict_proba(validation[0]))
    with pytest.raises(ValueError, match="signature"):
        changed = x.copy()
        changed[0, 0, 0, 0] += 1
        VisualResidualSearch("ADP-W-RR", spec(arch), settings=settings()).fit(
            changed, y, validation_data=validation, checkpoint_path=path)


@pytest.mark.parametrize("arch", ["V03", "V04", "V05"])
def test_native_ffn_fixed_model_and_generic_phase_search(arch):
    x, y, validation = sample()
    control = VisualResidualSearch(
        "MLP-Fixed", spec(arch), settings=settings(), fixed_grid=((3,), (4,)),
    )
    control.fit(x, y, validation_data=validation)
    assert [r["architecture"] for r in control.history_] == [[3], [4]]
    generic = VisualResidualSearch(
        "ADP-WD-All-Dcur", spec(arch), settings=settings(),
        generic_profile="corrected",
    )
    generic.fit(x, y, validation_data=validation)
    assert len(generic.history_) == 2
    assert generic.history_[1]["profile"] == "corrected"



def test_poolformer_parameter_free_mixer_and_frozen_contract():
    model = VisualNativeFFN(spec("V05"), (3,))
    x = torch.randn(3, 3, 8, 12)
    stemmed = model.stem(x)
    mixed = model.pooling_mixer(stemmed) - stemmed
    # PoolFormer explicitly defines a parameter-free residual Pool - Identity.
    assert torch.allclose(stemmed + mixed, model.pooling_mixer(stemmed))
    assert not any(model.pooling_mixer.parameters())
    assert model(x).shape == (3, 2)
    sha = model.frozen_state_sha256()
    with torch.no_grad():
        model.local_mlp.output_layer.weight.add_(0.1)
    assert model.frozen_state_sha256() == sha
    model.pooling_mixer.kernel_size = 5
    with pytest.raises(ValueError, match="frozen"):
        model.verify_frozen()
