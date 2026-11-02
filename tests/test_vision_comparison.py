"""Offline paired V01 comparison integration: no network or real-data claims."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch import nn

from simple_adp.config import SearchSettings
from simple_adp.vision_data import make_image_partitions
from simple_adp.vision_comparison import run_vision_comparison


def fixture(seed=13):
    rng = np.random.default_rng(55)
    train = rng.integers(0, 50, size=(32, 8, 8), dtype=np.uint8)
    labels = np.asarray([0, 1] * 16, dtype=np.int64)
    train[labels == 1] += 125
    test = rng.integers(0, 50, size=(16, 8, 8), dtype=np.uint8)
    ty = np.asarray([0, 1] * 8, dtype=np.int64)
    test[ty == 1] += 125
    parts = make_image_partitions(
        train, labels, test, ty, source="offline-comparison-test-only",
        revision="offline-v1", seed=seed,
        train_limit=16, validation_limit=8, test_limit=8,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(103)
        encoder = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
    metadata = {
        "architecture_id": "resnet18", "feature_dim": 4,
        "weights_source": "random_initialization",
        "weights_id": "offline-random-fixture-seed103",
        "pretrained": False,
    }
    return parts, encoder, metadata


def config():
    return SearchSettings(
        init_width=3, init_depth=1, max_width=5,
        max_depth=2, max_neurons=30, max_candidates=1,
        max_epochs=1, batch_size=4, seed=0, device="cpu",
        patience_es=1, patience_width_exp=1,
    )


def params(tmp_path):
    parts, encoder, metadata = fixture()
    return {
        "dataset": "fashion_mnist",
        "root": tmp_path / "absent_official_data",
        "output_dir": tmp_path / "paired",
        "architecture_id": "resnet18",
        "weights_path": None,
        "allow_random_backbone": True,
        "seeds": (13,),
        "settings": config(),
        "fixed_width": 3,
        "partitions": parts,
        "encoder": encoder,
        "encoder_metadata": metadata,
        "encoder_batch_size": 8,
    }


def test_paired_frozen_loss_variance_same_split_source_and_signed_jobs(tmp_path):
    cfg = params(tmp_path)
    summary = run_vision_comparison(**cfg)
    assert summary["status"] == "completed"
    assert summary["completed_jobs"] == 3
    assert [r["route"] for r in summary["rows"]] == [
        "fixed", "adp_loss", "adp_variance",
    ]
    assert [r["method_id"] for r in summary["rows"]] == [
        "MLP-Fixed", "ADP-W-RR", "ADP-W-RR",
    ]
    assert [r["criterion"] for r in summary["rows"]] == [
        "loss", "loss", "variance_stagnation",
    ]
    assert len({r["run_signature"] for r in summary["rows"]}) == 3
    assert len({r["official_split_sha256"] for r in summary["rows"]}) == 1
    assert len({r["source_encoder_state_sha256"] for r in summary["rows"]}) == 1
    plan = json.loads((cfg["output_dir"] / "comparison_manifest.json").read_text())
    assert plan["comparison_signature"] == summary["comparison_signature"]
    assert "no official-test-based winner selection" in plan["comparison_decision"]
    for row in summary["rows"]:
        result_file = cfg["output_dir"] / row["result_file"]
        assert result_file.is_file()
        record = json.loads(result_file.read_text())
        assert row["run_signature"] == record["run_signature"]
        assert record["history_entries"] > 0
        assert record["status"] == "completed"
        assert 0 <= row["validation_metrics"]["accuracy"] <= 1
        assert 0 <= row["official_test_metrics_descriptive_only"]["accuracy"] <= 1
        assert record["sample_counts"] == {
            "train": 16, "validation": 6, "official_test": 8,
        }
    assert run_vision_comparison(**cfg) == summary


def test_paired_plan_or_result_tamper_fails_closed(tmp_path):
    cfg = params(tmp_path)
    run_vision_comparison(**cfg)
    with pytest.raises(ValueError, match="plan identity mismatch"):
        run_vision_comparison(**{**cfg, "fixed_width": 4})
    with pytest.raises(ValueError, match="plan identity mismatch"):
        run_vision_comparison(**{
            **cfg, "settings": SearchSettings(**{**config().as_dict(), "max_width": 7}),
        })
    target = cfg["output_dir"] / "runs" / "seed13" / "adp_loss" / "result.json"
    with target.open("a", encoding="utf-8") as stream:
        stream.write("tampered")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        run_vision_comparison(**cfg)


def test_paired_interrupted_second_job_relaunch_preserves_first(tmp_path, monkeypatch):
    cfg = params(tmp_path)
    import simple_adp.vision_comparison as module
    original = module.run_vision_experiment
    interrupted = {"raised": False}

    def crash_second(*args, **kwargs):
        if kwargs["model_id"] == "ADP-W-RR" and kwargs["settings"].expansion_criterion == "loss" and not interrupted["raised"]:
            interrupted["raised"] = True
            raise RuntimeError("simulated job eviction before next job")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "run_vision_experiment", crash_second)
    with pytest.raises(RuntimeError, match="simulated job eviction"):
        run_vision_comparison(**cfg)
    fixed = cfg["output_dir"] / "runs" / "seed13" / "fixed" / "result.json"
    assert fixed.exists()
    original_result = fixed.read_bytes()
    assert not (cfg["output_dir"] / "comparison_summary.json").exists()
    monkeypatch.setattr(module, "run_vision_experiment", original)
    resumed = run_vision_comparison(**cfg)
    assert fixed.read_bytes() == original_result
    assert resumed["completed_jobs"] == 3


def test_paired_comparison_refuses_unverified_random_or_ambiguous_seeds(tmp_path):
    cfg = params(tmp_path)
    with pytest.raises(ValueError, match="requires local source weights"):
        run_vision_comparison(**{**cfg, "allow_random_backbone": False})
    with pytest.raises(ValueError, match="nonnegative, nonempty and unique"):
        run_vision_comparison(**{**cfg, "seeds": (13, 13)})
    with pytest.raises(ValueError, match="exactly one seed"):
        run_vision_comparison(**{**cfg, "seeds": (13, 14)})
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        run_vision_comparison(**{
            **cfg, "expected_weights_sha256": "0" * 64,
        })
    with pytest.raises(ValueError, match="strict source"):
        run_vision_comparison(**{
            **cfg, "encoder": None, "encoder_metadata": None,
            "allow_random_backbone": False,
        })
