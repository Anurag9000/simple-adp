"""Offline V01 real-image experiment harness contracts; no real data is claimed."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch
from torch import nn

from simple_adp.config import SearchSettings
from simple_adp.vision_data import ImagePartitions, make_image_partitions
from simple_adp.vision_encoders import VisionInputAdapter
from simple_adp.vision_experiment import run_vision_experiment
from simple_adp.visual import VisualADPEstimator


def fixture():
    rng = np.random.default_rng(56)
    train = rng.integers(0, 65, size=(28, 8, 8), dtype=np.uint8)
    labels = np.array([0, 1] * 14, dtype=np.int64)
    train[labels == 1] += 140
    test = rng.integers(0, 65, size=(12, 8, 8), dtype=np.uint8)
    test_labels = np.array([0, 1] * 6, dtype=np.int64)
    test[test_labels == 1] += 140
    parts = make_image_partitions(
        train, labels, test, test_labels, source="synthetic-only-test-fixture",
        revision="fixture-1", seed=7, train_limit=16, validation_limit=6,
        test_limit=6,
    )
    torch.manual_seed(17)
    encoder = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1),
        nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
    )
    metadata = {
        "architecture_id": "resnet18",
        "weights_id": "synthetic-fixture-seed-17",
        "weights_source": "random_initialization",
        "pretrained": False,
        "feature_dim": 4,
    }
    return parts, encoder, metadata


def settings():
    return SearchSettings(
        init_width=3, init_depth=1, max_width=5, max_depth=2,
        max_neurons=35, max_candidates=1, max_epochs=1,
        batch_size=4, seed=2, device="cpu", patience_es=1,
    )


def kwargs(tmp_path):
    partitions, encoder, metadata = fixture()
    return {
        "dataset": "fashion_mnist", "root": tmp_path / "no-net",
        "output_dir": tmp_path / "experiment",
        "architecture_id": "resnet18", "model_id": "MLP-Fixed",
        "fixed_grid": ((3,),), "settings": settings(),
        "allow_random_backbone": True,
        "partitions": partitions, "encoder": encoder, "encoder_metadata": metadata,
        "encoder_batch_size": 7,
    }


def test_grayscale_repeat_resize_and_fixed_eval_training_contract():
    identity = nn.Identity()
    gray = VisionInputAdapter(
        identity, architecture_id="resnet18",
        input_channels=1, input_resolution=(8, 9),
    )
    x = torch.full((2, 1, 8, 9), 0.2)
    transformed = gray(x)
    assert transformed.shape == (2, 3, 32, 32)
    assert torch.allclose(transformed, torch.full_like(transformed, 0.2))
    assert gray.preprocessing["channel_policy"] == "repeat grayscale across RGB"
    assert "bilinear" in gray.preprocessing["resize"]
    gray.train(True)
    assert not gray.training

    rgb = VisionInputAdapter(
        identity, architecture_id="convnext_tiny",
        input_channels=3, input_resolution=(32, 40),
    )
    image = torch.rand(1, 3, 32, 40)
    assert torch.equal(rgb(image), image)
    assert rgb.preprocessing["resize"] == "none"
    vit = VisionInputAdapter(
        identity, architecture_id="vit_b_16",
        input_channels=1, input_resolution=(28, 28),
    )
    assert vit(torch.ones(1, 1, 28, 28)).shape == (1, 3, 224, 224)
    with pytest.raises(ValueError, match="declared input"):
        gray(torch.zeros(1, 3, 8, 9))
    with pytest.raises(ValueError, match="finite"):
        gray(torch.full((1, 1, 8, 9), float("nan")))
    with pytest.raises(ValueError, match="channels"):
        VisionInputAdapter(identity, architecture_id="resnet18",
                           input_channels=2, input_resolution=(8, 8))


def test_experiment_writes_signed_manifest_results_and_inference_artifact(tmp_path):
    cfg = kwargs(tmp_path)
    completed = run_vision_experiment(**cfg)
    assert completed["status"] == "completed"
    assert completed["dataset"] == "fashion_mnist"
    assert completed["weights_source"] == "random_initialization"
    assert completed["pretrained"] is False
    assert completed["selected_architecture"] == [3]
    assert set(completed["sample_counts"]) == {"train", "validation", "official_test"}
    assert completed["sample_counts"] == {"train": 16, "validation": 6, "official_test": 6}
    for name in ("manifest.json", "result.json", "estimator.model.pt",
                 "estimator.model.pt.vision.json", "estimator.checkpoint.pt"):
        assert (cfg["output_dir"] / name).exists()
    recorded = json.loads((cfg["output_dir"] / "manifest.json").read_text())
    assert recorded["format"] == "simple-adp-vision-v01-experiment-v1"
    assert recorded["partitions"]["source"] == "synthetic-only-test-fixture"
    assert recorded["partitions"]["source_test_indices"] == cfg["partitions"].evaluation_indices.tolist()
    assert recorded["split_policy"].endswith("after fit")
    assert recorded["input_transform"]["raw_channels"] == 1
    assert recorded["input_transform"]["output_resolution"] == [32, 32]
    assert completed["run_signature"] == recorded["run_signature"]
    assert len(completed["source_encoder_state_sha256"]) == 64
    assert 0 <= completed["test_metrics"]["accuracy"] <= 1
    assert completed["test_metrics"]["n_samples"] if "n_samples" in completed["test_metrics"] else True
    assert run_vision_experiment(**cfg) == completed


def test_official_test_materialization_occurs_only_after_model_fit(tmp_path, monkeypatch):
    cfg = kwargs(tmp_path)
    events = []
    original_materialize = ImagePartitions.materialize
    original_fit = VisualADPEstimator.fit

    def tracked_materialize(self, partition):
        events.append("read:" + partition)
        return original_materialize(self, partition)

    def tracked_fit(self, *args, **kwargs):
        events.append("fit")
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(ImagePartitions, "materialize", tracked_materialize)
    monkeypatch.setattr(VisualADPEstimator, "fit", tracked_fit)
    run_vision_experiment(**cfg)
    assert events[:3] == ["read:train", "read:validation", "fit"]
    assert events[-1] == "read:test"


def test_run_manifest_blocks_mismatched_encoder_settings_and_wrong_artifacts(tmp_path):
    cfg = kwargs(tmp_path)
    run_vision_experiment(**cfg)
    changed = {**cfg, "settings": SearchSettings(**{**settings().as_dict(), "seed": 27})}
    with pytest.raises(ValueError, match="manifest identity mismatch"):
        run_vision_experiment(**changed)
    altered = {**cfg}
    torch.manual_seed(100)
    altered["encoder"] = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
    )
    with pytest.raises(ValueError, match="manifest identity mismatch"):
        run_vision_experiment(**altered)
    path = cfg["output_dir"] / "estimator.model.pt"
    with path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="model checksum changed"):
        run_vision_experiment(**cfg)


def test_random_encoder_requires_explicit_opt_in_and_rejects_identity_drift(tmp_path):
    cfg = kwargs(tmp_path)
    with pytest.raises(ValueError, match="local full-model weights required"):
        run_vision_experiment(**{**cfg, "allow_random_backbone": False})
    with pytest.raises(ValueError, match="source encoder metadata"):
        run_vision_experiment(**{**cfg, "encoder_metadata": {
            **cfg["encoder_metadata"], "architecture_id": "convnext_tiny",
        }})
    with pytest.raises(ValueError, match="metadata"):
        run_vision_experiment(**{**cfg, "encoder_metadata": None})
    with pytest.raises(ValueError, match="no weights_path"):
        run_vision_experiment(**{**cfg, "weights_path": tmp_path / "not-real.pt"})
    with pytest.raises(ValueError, match="dataset must be"):
        run_vision_experiment(**{**cfg, "dataset": "arbitrary"})
