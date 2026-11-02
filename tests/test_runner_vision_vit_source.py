"""Explicit source-faithful ViT V03 43-job route and one actual worker stage."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.plan import SuitePlan
from simple_adp.runner.plugins import load_registries
from simple_adp.runner.worker import run_spec
from simple_adp.vision_data import make_image_partitions
from simple_adp.visual_source_convnext import file_sha256


@pytest.fixture(scope="module")
def vit_source(tmp_path_factory):
    torchvision = pytest.importorskip("torchvision")
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(117)
        source = torchvision.models.vit_b_16(weights=None).eval()
    path = tmp_path_factory.mktemp("source-vit-runner") / "vit_b16_full.pt"
    torch.save(source.state_dict(), path)
    return path, file_sha256(path)


def enable(monkeypatch, path, sha, *, source_gate=True):
    monkeypatch.setenv("SIMPLE_ADP_ENABLE_VISION", "1")
    monkeypatch.delenv("SIMPLE_ADP_ENABLE_V04_SOURCE", raising=False)
    if source_gate:
        monkeypatch.setenv("SIMPLE_ADP_ENABLE_V03_SOURCE", "1")
    else:
        monkeypatch.delenv("SIMPLE_ADP_ENABLE_V03_SOURCE", raising=False)
    monkeypatch.setenv("SIMPLE_ADP_V03_SOURCE_WEIGHTS", str(path))
    monkeypatch.setenv("SIMPLE_ADP_V03_SOURCE_SHA256", sha)
    monkeypatch.setenv("TABULAR_SKIP_SYSTEMD_SCOPE", "1")
    monkeypatch.setenv("TABULAR_BACKEND", "cpu")
    monkeypatch.setenv("SIMPLE_ADP_DISABLE_GPU_ACCELERATORS", "1")
    monkeypatch.setenv("TRAINING_CONTROL_CPU_ONLY", "1")


def settings():
    return RunnerSettings(
        tasks=("vision",), run_hardness_screen=False,
        repeat_count=1, missing_present_task_repeats=1, model_seeds=(0,),
        width_to_depth_width=2, width_to_depth_depth=1,
        max_width=3073, max_neurons=4000, max_candidates=1,
        max_epochs=1, batch_size=2, smoke_test=False, cpu_only=True,
    )


def plan(tmp_path):
    datasets, models = load_registries()
    return SuitePlan.build(
        settings=settings(), results_dir=tmp_path / "results",
        run_root=tmp_path / "runs",
        dataset_providers=tuple(datasets.providers.values()),
        dataset_suites=(("vision", "toy"),),
        model_adapters=tuple(models.adapters.values()),
    )


def test_v03_source_optin_preserves_817_and_registers_exact_43_matched(vit_source, monkeypatch, tmp_path):
    file, sha = vit_source
    enable(monkeypatch, file, sha, source_gate=False)
    assert "vision_v03_source" not in load_registries()[1].adapters
    original = plan(tmp_path / "original")
    assert len(original.runs) == 817

    enable(monkeypatch, file, sha, source_gate=True)
    extended = plan(tmp_path / "extended")
    source = [r for r in extended.runs if r.model_adapter == "vision_v03_source"]
    originals = [r for r in source
                 if r.worker_settings.get("expansion_criterion") != "variance_stagnation"]
    paired = [r for r in source
              if r.worker_settings.get("expansion_criterion") == "variance_stagnation"]
    assert len(extended.runs) == 860
    assert len(source) == 43 and len(originals) == 22 and len(paired) == 21
    assert len({j.identity.run_id for j in extended.runs}) == 860
    assert all(j.route.route_label.startswith("V03-source:sha" + sha[:20] + ":")
               for j in source)
    assert all(j.route.init_width == 3072 and j.route.init_depth == 1 for j in source)
    assert sum(j.route.model_id == "MLP-Fixed" for j in originals) == 1
    for other in paired:
        parent = next(j for j in originals
                      if j.identity.run_id == other.worker_settings["counterpart_of"])
        assert parent.route.model_id == other.route.model_id
        assert parent.route.generic_profile == other.route.generic_profile
        assert parent.identity.split_sha256 == other.identity.split_sha256
        assert parent.identity.seed == other.identity.seed


def test_v03_source_integrity_rejects_wrong_or_missing_weights(vit_source, monkeypatch, tmp_path):
    file, sha = vit_source
    enable(monkeypatch, file, "0" * 64)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_registries()[1].get("vision_v03_source").enumerate_routes(settings=settings())
    enable(monkeypatch, file, sha)
    route = next(r for r in load_registries()[1].get("vision_v03_source").enumerate_routes(
        settings=settings()) if r.model_id == "ADP-W-RR")
    monkeypatch.setenv("SIMPLE_ADP_V03_SOURCE_WEIGHTS", str(tmp_path / "nonexistent.pt"))
    with pytest.raises(ValueError, match="does not exist"):
        load_registries()[1].get("vision_v03_source").enumerate_routes(settings=settings())
    assert sha[:20] in route.route_label


def test_v03_source_worker_uses_full_vit_and_writes_signed_result(vit_source, monkeypatch, tmp_path):
    file, sha = vit_source
    enable(monkeypatch, file, sha)
    # Only 4 train / 2 valid / 2 test pixels; exercise true full source model,
    # not benchmark quality, and avoid repeated 86M-param checkpoint writes.
    from simple_adp.runner.vision import VisionDatasetProvider
    rng = np.random.default_rng(61)
    x = rng.integers(0, 60, size=(12, 1, 8, 8), dtype=np.uint8)
    y = np.array([0, 1] * 6, dtype=np.int64)
    x[y == 1] += 150
    test_x = rng.integers(0, 60, size=(4, 1, 8, 8), dtype=np.uint8)
    test_y = np.array([0, 1] * 2, dtype=np.int64)
    test_x[test_y == 1] += 150
    parts = make_image_partitions(
        x, y, test_x, test_y, source="vit-source-small-offline-test-only",
        revision="tiny-v1", seed=11,
        train_limit=4, validation_limit=2, test_limit=2,
    )
    monkeypatch.setattr(VisionDatasetProvider, "_toy", staticmethod(lambda: parts))
    suite = plan(tmp_path)
    fixed = next(
        x for x in suite.runs
        if x.model_adapter == "vision_v03_source" and x.route.model_id == "MLP-Fixed"
        and x.worker_settings.get("expansion_criterion") != "variance_stagnation"
    )
    output = run_spec(fixed.command[-1])
    assert output["status"] == "completed"
    assert output["architecture_id"] == "V03:torchvision-vit-b16-source-ffn"
    assert output["selected_architecture"] == [3072]
    assert output["growth_site"] == "vit:encoder_layer_0.channel_ffn"
    assert output["source_weights_sha256"] == sha
    assert output["source_ffn_exact_at_initialization"] is True
    assert output["artifact_path"] and Path(output["artifact_path"]).is_file()
    assert len(output["frozen_sha256"]) == 64
    assert 0 <= output["test_metrics"]["accuracy"] <= 1
    saved_spec = json.loads(Path(fixed.command[-1]).read_text(encoding="utf-8"))
    assert saved_spec["model_adapter"] == "vision_v03_source"
