"""Real torchvision ConvNeXt source FFN transplant, frozen graph and native growth."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from torch import nn

from simple_adp.config import SearchSettings
from simple_adp.visual_source_convnext import (
    SourceConvNeXtSpec, VisualSourceConvNeXtFFN, file_sha256,
)
from simple_adp.visual_native_ffn import build_visual_composite, restore_visual_composite
from simple_adp.visual_search import VisualResidualSearch


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    torchvision = pytest.importorskip("torchvision")
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(813)
        full = torchvision.models.convnext_tiny(weights=None).eval()
    path = tmp_path_factory.mktemp("convnext-full-source") / "full_state_dict.pt"
    torch.save(full.state_dict(), path)
    return path, file_sha256(path), full


def spec(source, *, channels=3):
    path, sha, _ = source
    return SourceConvNeXtSpec(
        classes=2, image_height=32, image_width=32,
        input_channels=channels, weights_path=str(path),
        weights_sha256=sha, fixed_seed=24,
    )


def test_v04_exact_torchvision_source_ffn_transfer_preserves_frozen_features(source):
    path, sha, original = source
    before_rng = torch.get_rng_state().clone()
    adapted = VisualSourceConvNeXtFFN(spec(source), (384,))
    assert torch.equal(before_rng, torch.get_rng_state())
    assert adapted.source_ffn_exact_at_initialization
    assert adapted.hidden_widths == [384]
    assert adapted.input_dim == adapted.output_dim == 96
    assert adapted.spec.weights_sha256 == sha
    assert len(adapted.growth_targets) == 1
    assert isinstance(adapted.source.features[1][0].block[3], nn.Module)
    source_block = original.features[1][0].block
    local = adapted.local_mlp
    assert torch.equal(local.hidden_layers[0].weight, source_block[3].weight)
    assert torch.equal(local.hidden_layers[0].bias, source_block[3].bias)
    assert torch.equal(local.output_layer.weight, source_block[5].weight)
    assert torch.equal(local.output_layer.bias, source_block[5].bias)
    assert isinstance(local.hidden_layers[1], nn.GELU)

    images = torch.rand(1, 3, 32, 32)
    normalized = (images - adapted.input_mean) / adapted.input_std
    with torch.no_grad():
        reference_features = original.features(normalized)
        new_features = adapted.source.features(normalized)
    assert torch.equal(reference_features, new_features), (
        "initial source FFN transplant must exactly preserve all frozen features"
    )
    adapted.train(True)
    assert adapted.task_head.training and adapted.local_mlp.training
    assert not adapted.source.training
    adapted.verify_frozen()
    out = adapted(images)
    assert out.shape == (1, 2)
    out.square().mean().backward()
    assert adapted.local_mlp.output_layer.weight.grad.abs().sum() > 0
    assert adapted.task_head.weight.grad.abs().sum() > 0
    assert all(p.grad is None for n, p in adapted.source.named_parameters()
               if not n.startswith(("features.1.0.block.3.", "classifier.2.")))
    adapted.verify_frozen()


def test_v04_real_source_width_depth_growth_and_exact_snapshots(source):
    model = build_visual_composite(spec(source), (384,))
    frozen = model.frozen_state_sha256()
    out_original = model.local_mlp.output_layer.weight.detach().clone()
    grown = model.grow_width(
        site_id=model.spec.site_id, operator="round_robin",
        increment=1, max_width=385,
    )
    assert grown is not None
    assert grown.hidden_widths == [385]
    assert grown.source_ffn_exact_at_initialization
    assert grown.frozen_state_sha256() == frozen
    assert torch.equal(grown.local_mlp.output_layer.weight[:, :384], out_original)
    assert grown.local_mlp.output_layer.weight.shape == (96, 385)
    assert isinstance(grown.local_mlp.hidden_layers[1], nn.GELU)
    assert sum(p.numel() for p in grown.local_mlp.parameters()) > sum(
        p.numel() for p in model.local_mlp.parameters()
    )
    with pytest.raises(ValueError, match="noneligible"):
        model.grow_width(site_id="convnext:other", operator="all_layers",
                         increment=1, max_width=390)

    rng = torch.get_rng_state().clone()
    restored = restore_visual_composite(grown.snapshot(cursor=grown._width_cursor))
    assert torch.equal(rng, torch.get_rng_state())
    assert restored.spec == grown.spec
    assert restored.hidden_widths == [385]
    assert restored._width_cursor == 0
    assert restored.source_ffn_exact_at_initialization
    assert restored.frozen_state_sha256() == frozen
    with torch.no_grad():
        x = torch.rand(1, 3, 32, 32)
        assert torch.equal(restored.eval()(x), grown.eval()(x))

    deeper = model.grow_depth(
        site_id=model.spec.site_id, operator="D1", max_depth=2,
    )
    assert deeper is not None
    assert deeper.hidden_widths == [384, 1]
    assert deeper.input_dim == deeper.output_dim == 96
    assert deeper.frozen_state_sha256() == frozen
    model.train(False)
    with torch.no_grad():
        next(model.source.features[0].parameters()).add_(0.02)
    with pytest.raises(ValueError, match="frozen state"):
        model.verify_frozen()


def test_v04_source_strict_state_checksum_and_pretrained_initial_width(source, tmp_path):
    source_spec = spec(source)
    with pytest.raises(ValueError, match="SHA256"):
        VisualSourceConvNeXtFFN(SourceConvNeXtSpec(
            **{**source_spec.__dict__, "weights_sha256": "0" * 64},
        ))
    with pytest.raises(ValueError, match="one eligible"):
        SourceConvNeXtSpec(**{**source_spec.__dict__, "site_id": "convnext:other"})
    settings = SearchSettings(
        init_width=8, init_depth=1, max_width=390,
        max_depth=2, max_neurons=1000, max_epochs=1,
        max_candidates=0, batch_size=2, patience_es=1,
        device="cpu", seed=1,
    )
    x = np.random.default_rng(2).random((4, 3, 32, 32), dtype=np.float32)
    y = np.asarray([0, 1, 0, 1])
    with pytest.raises(ValueError, match="exactly 384"):
        VisualResidualSearch(
            "ADP-W-RR", source_spec, settings=settings,
        ).fit(x[:2], y[:2], validation_data=(x[2:], y[2:]))



def test_v04_native_controller_trains_full_image_with_frozen_source(source):
    """One real source-block candidate through the existing ADP stage trainer."""
    x = np.random.default_rng(11).random((8, 3, 32, 32), dtype=np.float32)
    y = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    settings = SearchSettings(
        init_width=384, init_depth=1, max_width=385,
        max_depth=2, max_neurons=1000, max_epochs=1,
        max_candidates=0, batch_size=2, patience_es=1,
        device="cpu", seed=17,
    )
    search = VisualResidualSearch(
        "MLP-Fixed", spec(source), settings=settings,
        fixed_grid=((384,),),
    )
    search.fit(x[:4], y[:4], validation_data=(x[4:], y[4:]))
    assert len(search.history_) == 1
    assert search.history_[0]["architecture"] == [384]
    assert search.model_.hidden_widths == [384]
    assert search.model_.source_ffn_exact_at_initialization
    search.model_.verify_frozen()
    probs = search.predict_proba(x[4:])
    assert probs.shape == (4, 2)
    assert np.isfinite(probs).all()



def test_v04_source_exact_mid_batch_checkpoint_restart(source, tmp_path):
    """True 28M-param source with exact optimizer/batch replay, not tiny proxy."""
    x = np.random.default_rng(34).random((8, 3, 32, 32), dtype=np.float32)
    y = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    cfg = SearchSettings(
        init_width=384, init_depth=1, max_width=384,
        max_depth=2, max_neurons=1000, max_epochs=1,
        max_candidates=1, batch_size=2, patience_es=1,
        patience_width_exp=1, device="cpu", seed=31,
    )
    args = (x[:4], y[:4])
    valid = (x[4:], y[4:])
    baseline = VisualResidualSearch("ADP-W-RR", spec(source), settings=cfg)
    baseline.fit(*args, validation_data=valid)
    filename = tmp_path / "source-v04.checkpoint.pt"

    class Eviction(Exception):
        pass

    def interrupt(progress):
        if (progress["phase"] == "initial"
                and progress["stage_status"] == "active"
                and progress["batches_done"] == 1):
            raise Eviction()

    with pytest.raises(Eviction):
        VisualResidualSearch("ADP-W-RR", spec(source), settings=cfg).fit(
            *args, validation_data=valid, checkpoint_path=filename,
            progress_callback=interrupt,
        )
    assert filename.exists()
    resumed = VisualResidualSearch("ADP-W-RR", spec(source), settings=cfg)
    resumed.fit(*args, validation_data=valid, checkpoint_path=filename)
    assert resumed.history_ == baseline.history_
    assert resumed.model_.hidden_widths == baseline.model_.hidden_widths == [384]
    assert resumed.result_.score == baseline.result_.score
    assert np.array_equal(resumed.predict_proba(valid[0]),
                          baseline.predict_proba(valid[0]))
    assert resumed.model_.frozen_state_sha256() == baseline.model_.frozen_state_sha256()
    with pytest.raises(ValueError, match="signature"):
        other = SearchSettings(**{**cfg.as_dict(), "seed": 32})
        VisualResidualSearch("ADP-W-RR", spec(source), settings=other).fit(
            *args, validation_data=valid, checkpoint_path=filename,
        )
