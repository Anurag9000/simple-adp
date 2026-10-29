"""Bounded V01 tests: actual ADP head training and frozen image encoder."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from simple_adp.config import SearchSettings
from simple_adp.growth import expand_width_round_robin
from simple_adp.visual import VisualADPEstimator, _encoder_digest


def _encoder():
    return nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1),
        nn.BatchNorm2d(4),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )


def _data():
    torch.manual_seed(127)
    images = torch.rand(24, 1, 8, 8)
    labels = np.asarray([0, 1] * 12, dtype=np.int64)
    images[labels == 1] += 1.0
    return images, labels


def _settings():
    return SearchSettings(
        init_width=2, init_depth=1, ex_k_width=1,
        max_width=5, max_neurons=20, max_depth=2,
        max_epochs=1, max_candidates=1, patience_es=1,
        patience_width_exp=1, batch_size=4, seed=7, device="cpu",
    )


def test_v01_real_training_fixed_control_save_load(tmp_path):
    images, labels = _data()
    torch.manual_seed(22)
    encoder = _encoder()
    identity = _encoder_digest(encoder)
    manifest = {"source": "deterministic-image-test", "train": list(range(16)), "valid": list(range(16, 24)),
                "normalization": "identity", "seed": 127}
    estimator = VisualADPEstimator(
        encoder, encoder_id="tiny-conv", weights_id="seed-22",
        dataset_manifest=manifest, model_id="MLP-Fixed", fixed_grid=[[2]],
        pool="vector", expected_resolution=(8, 8), encoder_batch_size=5,
        settings=_settings(),
    )
    estimator.fit(images[:16], labels[:16], validation_data=(images[16:], labels[16:]))
    assert encoder.training is False
    assert all(not p.requires_grad for p in encoder.parameters())
    assert _encoder_digest(encoder) == identity
    assert estimator.head.model_.hidden_widths == [2]
    prob = estimator.predict_proba(images[16:])
    assert prob.shape == (8, 2)
    assert np.isfinite(prob).all()
    assert estimator.features(images[:3]).shape == (3, 4)
    path = tmp_path / "vision.pt"
    estimator.save(path)
    restored = VisualADPEstimator.load(
        path, encoder=encoder,
        encoder_id="tiny-conv", weights_id="seed-22", dataset_manifest=manifest,
    )
    assert np.allclose(restored.predict_proba(images[16:]), prob, atol=1e-6)
    with pytest.raises(ValueError, match="identity mismatch|mismatch"):
        VisualADPEstimator.load(
            path, encoder=encoder, encoder_id="tiny-conv",
            weights_id="different-pretraining", dataset_manifest=manifest,
        )


def test_v01_native_growth_and_frozen_invariants():
    images, labels = _data()
    encoder = _encoder()
    model = VisualADPEstimator(
        encoder, encoder_id="tiny-conv", weights_id="seed-experiment",
        dataset_manifest={"source": "synthetic", "splits": "first16-next8"},
        model_id="ADP-W-RR", pool="vector", expected_resolution=(8, 8),
        settings=_settings(),
    )
    initial = model.encoder_sha256
    model.fit(images[:16], labels[:16], validation_data=(images[16:], labels[16:]))
    assert model.head.history_
    original = model.head.model_
    candidate = expand_width_round_robin(original, 1, max_width=6)
    assert candidate is not None
    assert sum(candidate.hidden_widths) > sum(original.hidden_widths)
    assert candidate(images[:2].new_zeros((2, 4))).shape == (2, 2)
    assert _encoder_digest(encoder) == initial
    with pytest.raises(ValueError, match="resolution"):
        model.predict(images[:2, :, :7])
    with torch.no_grad():
        next(encoder.parameters()).add_(0.1)
    with pytest.raises(ValueError, match="changed"):
        model.features(images[:2])
