"""True torchvision ViT-B/16 source FFN fidelity, growth, and signed recovery."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from simple_adp.config import SearchSettings
from simple_adp.visual_source_vit import SourceViTSpec, VisualSourceViTFFN
from simple_adp.visual_source_convnext import file_sha256
from simple_adp.visual_native_ffn import build_visual_composite, restore_visual_composite
from simple_adp.visual_search import VisualResidualSearch


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    torchvision = pytest.importorskip("torchvision")
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(461)
        original = torchvision.models.vit_b_16(weights=None).eval()
    path = tmp_path_factory.mktemp("source-vit") / "vit_b16_complete.pt"
    torch.save(original.state_dict(), path)
    return path, file_sha256(path), original


def spec(source, channels=3):
    file, sha, _ = source
    return SourceViTSpec(
        classes=2, image_height=32, image_width=32,
        input_channels=channels, weights_path=str(file),
        weights_sha256=sha, fixed_seed=17,
    )


def test_true_vit_first_ffn_transplant_exact_and_frozen(source):
    _, sha, original = source
    rng = torch.get_rng_state().clone()
    adapted = VisualSourceViTFFN(spec(source), (3072,))
    assert torch.equal(rng, torch.get_rng_state())
    assert adapted.source_ffn_exact_at_initialization
    assert adapted.input_dim == 768 and adapted.output_dim == 768
    assert adapted.hidden_widths == [3072]
    assert sha == adapted.spec.weights_sha256
    assert adapted.growth_targets.keys() == {adapted.spec.site_id}
    old = original.encoder.layers[0].mlp
    new = adapted.local_mlp
    assert torch.equal(new.hidden_layers[0].weight, old[0].weight)
    assert torch.equal(new.hidden_layers[0].bias, old[0].bias)
    assert torch.equal(new.output_layer.weight, old[3].weight)
    assert torch.equal(new.output_layer.bias, old[3].bias)
    assert isinstance(new.hidden_layers[1], nn.GELU)

    # The actual source transformer FFN output is exact in eval mode.
    tokens = torch.randn(2, 3, 768)
    with torch.no_grad():
        assert torch.equal(old(tokens), adapted.source.encoder.layers[0].mlp(tokens))
    # Whole model must still use original 224x224 position-embedding geometry.
    images = torch.rand(1, 3, 32, 32)
    normalized = torch.nn.functional.interpolate(
        images, size=(224, 224), mode="bilinear", align_corners=False)
    normalized = (normalized - adapted.input_mean) / adapted.input_std
    with torch.no_grad():
        source_tokens = original._process_input(normalized)
        encoded = torch.cat(
            [original.class_token.expand(source_tokens.shape[0], -1, -1),
             source_tokens], dim=1)
        original_features = original.encoder(encoded)
        new_features = adapted.source.encoder(encoded)
        assert torch.equal(original_features, new_features)
    adapted.train(True)
    adapted.verify_frozen()
    assert adapted.local_mlp.training and adapted.task_head.training
    assert not adapted.source.training
    assert adapted(images).shape == (1, 2)
    with pytest.raises(ValueError, match="input"):
        adapted(torch.rand(1, 1, 32, 32))


def test_true_vit_growth_and_snapshot_preserve_source_identity(source):
    model = build_visual_composite(spec(source), (3072,))
    frozen = model.frozen_state_sha256()
    original_weights = model.local_mlp.output_layer.weight.detach().clone()
    wider = model.grow_width(
        site_id=model.spec.site_id, operator="round_robin",
        increment=1, max_width=3073,
    )
    assert wider is not None and wider.hidden_widths == [3073]
    assert torch.equal(wider.local_mlp.output_layer.weight[:, :3072], original_weights)
    assert isinstance(wider.local_mlp.hidden_layers[1], nn.GELU)
    assert wider.frozen_state_sha256() == frozen
    assert wider.source_ffn_exact_at_initialization
    with pytest.raises(ValueError, match="noneligible"):
        model.grow_width(site_id="vit:wrong", operator="all_layers",
                         increment=1, max_width=3073)
    rng = torch.get_rng_state().clone()
    restored = restore_visual_composite(wider.snapshot(cursor=wider._width_cursor))
    assert torch.equal(rng, torch.get_rng_state())
    assert restored.hidden_widths == wider.hidden_widths
    assert restored.frozen_state_sha256() == frozen
    assert restored.spec == wider.spec
    assert restored.source_ffn_exact_at_initialization
    with torch.no_grad():
        next(model.source.conv_proj.parameters()).add_(0.01)
    with pytest.raises(ValueError, match="frozen"):
        model.verify_frozen()


def test_true_vit_rejects_incompatible_full_source_and_width(source):
    file, sha, _ = source
    with pytest.raises(ValueError, match="SHA256"):
        VisualSourceViTFFN(SourceViTSpec(
            **{**spec(source).__dict__, "weights_sha256": "0"*64}))
    with pytest.raises(ValueError, match="eligible"):
        SourceViTSpec(**{**spec(source).__dict__, "site_id": "vit:wrong"})
    wrong = SearchSettings(
        init_width=8, init_depth=1, max_width=3073, max_depth=2,
        max_neurons=5000, max_epochs=1, max_candidates=1,
        batch_size=2, device="cpu",
    )
    x = np.zeros((4, 3, 32, 32), dtype=np.float32)
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="exactly 3072"):
        VisualResidualSearch("ADP-W-RR", spec(source), settings=wrong).fit(
            x[:2], y[:2], validation_data=(x[2:], y[2:]))


def test_true_vit_native_fixed_stage_and_gradient(source):
    x = np.random.default_rng(7).random((4, 3, 32, 32), dtype=np.float32)
    y = np.array([0, 1, 0, 1])
    cfg = SearchSettings(
        init_width=3072, init_depth=1, max_width=3072,
        max_depth=2, max_neurons=5000, max_epochs=1,
        max_candidates=1, batch_size=2, patience_es=1,
        device="cpu", seed=7,
    )
    search = VisualResidualSearch(
        "MLP-Fixed", spec(source), settings=cfg,
        fixed_grid=((3072,),))
    search.fit(x[:2], y[:2], validation_data=(x[2:], y[2:]))
    assert len(search.history_) == 1
    assert search.model_.hidden_widths == [3072]
    assert search.model_.source_ffn_exact_at_initialization
    search.model_.verify_frozen()
    probs = search.predict_proba(x[2:])
    assert probs.shape == (2, 2)
    assert np.isfinite(probs).all()



def test_true_vit_exact_interrupted_batch_relaunch(source, tmp_path):
    """Signed 86M-param original source resumes the *same* batch/optimizer state."""
    x = np.random.default_rng(24).random((8, 3, 32, 32), dtype=np.float32)
    y = np.asarray([0, 1] * 4, dtype=np.int64)
    config = SearchSettings(
        init_width=3072, init_depth=1, max_width=3072,
        max_depth=2, max_neurons=5000, max_epochs=1,
        max_candidates=1, batch_size=2, patience_es=1,
        device="cpu", seed=29,
    )
    fit, valid = (x[:4], y[:4]), (x[4:], y[4:])
    baseline = VisualResidualSearch("ADP-W-RR", spec(source), settings=config)
    baseline.fit(*fit, validation_data=valid)

    class Evicted(Exception):
        pass

    def evict_after_first_batch(progress):
        if (progress["phase"] == "initial"
                and progress["stage_status"] == "active"
                and progress["batches_done"] == 1):
            raise Evicted()

    checkpoint = tmp_path / "vit-full-ffn.checkpoint.pt"
    with pytest.raises(Evicted):
        VisualResidualSearch("ADP-W-RR", spec(source), settings=config).fit(
            *fit, validation_data=valid, checkpoint_path=checkpoint,
            progress_callback=evict_after_first_batch,
        )
    assert checkpoint.is_file()
    resumed = VisualResidualSearch("ADP-W-RR", spec(source), settings=config)
    resumed.fit(*fit, validation_data=valid, checkpoint_path=checkpoint)
    assert resumed.history_ == baseline.history_
    assert resumed.model_.hidden_widths == baseline.model_.hidden_widths
    assert resumed.result_.score == baseline.result_.score
    assert resumed.model_.frozen_state_sha256() == baseline.model_.frozen_state_sha256()
    assert np.array_equal(resumed.predict_proba(valid[0]),
                          baseline.predict_proba(valid[0]))
    with pytest.raises(ValueError, match="signature"):
        drift = SearchSettings(**{**config.as_dict(), "seed": 30})
        VisualResidualSearch("ADP-W-RR", spec(source), settings=drift).fit(
            *fit, validation_data=valid, checkpoint_path=checkpoint,
        )
