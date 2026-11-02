"""Explicitly gated source V04 central runner and true 43-job pair ledger."""
from __future__ import annotations

import json

import pytest
import torch

from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.plan import SuitePlan
from simple_adp.runner.plugins import load_registries
from simple_adp.runner.worker import run_spec
from simple_adp.visual_source_convnext import file_sha256


@pytest.fixture(scope="module")
def source_file(tmp_path_factory):
    torchvision = pytest.importorskip("torchvision")
    torch.set_num_threads(2)
    folder = tmp_path_factory.mktemp("v04-source-runner")
    path = folder / "convnext_full_toy_random.pt"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(91)
        source = torchvision.models.convnext_tiny(weights=None).eval()
    torch.save(source.state_dict(), path)
    # A complete random torchvision checkpoint is NOT claimed to be pretrained.
    return path, file_sha256(path)


def configure(monkeypatch, file, sha, *, gate=True):
    monkeypatch.setenv("SIMPLE_ADP_ENABLE_VISION", "1")
    if gate:
        monkeypatch.setenv("SIMPLE_ADP_ENABLE_V04_SOURCE", "1")
    else:
        monkeypatch.delenv("SIMPLE_ADP_ENABLE_V04_SOURCE", raising=False)
    monkeypatch.setenv("SIMPLE_ADP_V04_SOURCE_WEIGHTS", str(file))
    monkeypatch.setenv("SIMPLE_ADP_V04_SOURCE_SHA256", sha)
    monkeypatch.setenv("TABULAR_SKIP_SYSTEMD_SCOPE", "1")
    monkeypatch.setenv("TABULAR_BACKEND", "cpu")
    monkeypatch.setenv("SIMPLE_ADP_DISABLE_GPU_ACCELERATORS", "1")
    monkeypatch.setenv("TRAINING_CONTROL_CPU_ONLY", "1")


def options():
    return RunnerSettings(
        tasks=("vision",), run_hardness_screen=False,
        repeat_count=1, missing_present_task_repeats=1, model_seeds=(0,),
        width_to_depth_width=2, width_to_depth_depth=1,
        max_width=385, max_neurons=1000, max_candidates=1,
        max_epochs=1, batch_size=16, smoke_test=False, cpu_only=True,
    )


def plan(tmp_path):
    datasets, models = load_registries()
    return SuitePlan.build(
        settings=options(), results_dir=tmp_path / "results",
        run_root=tmp_path / "runs",
        dataset_providers=tuple(datasets.providers.values()),
        dataset_suites=(("vision", "toy"),),
        model_adapters=tuple(models.adapters.values()),
    )


def test_v04_source_opt_in_does_not_change_original_817_plan(source_file, monkeypatch, tmp_path):
    file, sha = source_file
    configure(monkeypatch, file, sha, gate=False)
    datasets, models = load_registries()
    assert "vision_v04_source" not in models.adapters
    original = plan(tmp_path / "baseline")
    assert len(original.runs) == 817

    configure(monkeypatch, file, sha, gate=True)
    datasets, models = load_registries()
    assert "vision_v04_source" in models.adapters
    extended = plan(tmp_path / "enabled")
    source = [r for r in extended.runs if r.model_adapter == "vision_v04_source"]
    originals = [
        r for r in source
        if r.worker_settings.get("expansion_criterion") != "variance_stagnation"
    ]
    counterparts = [
        r for r in source
        if r.worker_settings.get("expansion_criterion") == "variance_stagnation"
    ]
    assert len(extended.runs) == 860
    assert len(source) == 43
    assert len(originals) == 22
    assert len(counterparts) == 21
    assert len({r.identity.run_id for r in extended.runs}) == 860
    assert all("V04-source:sha" + sha[:20] in r.route.route_label for r in source)
    for variant in counterparts:
        parent = next(
            (r for r in originals
             if r.identity.run_id == variant.worker_settings["counterpart_of"]),
            None,
        )
        assert parent is not None
        assert variant.route.model_id == parent.route.model_id
        assert variant.route.generic_profile == parent.route.generic_profile
        assert variant.identity.seed == parent.identity.seed
        assert variant.identity.split_sha256 == parent.identity.split_sha256
    assert sum(r.route.model_id == "MLP-Fixed" for r in originals) == 1
    assert all(r.route.init_width == 384 and r.route.init_depth == 1 for r in source)


def test_v04_source_cannot_reroute_different_or_missing_checkpoint(
    source_file, monkeypatch, tmp_path,
):
    file, sha = source_file
    configure(monkeypatch, file, "0" * 64)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_registries()[1].get("vision_v04_source").enumerate_routes(
            settings=options())
    configure(monkeypatch, file, sha)
    registered = load_registries()[1].get("vision_v04_source")
    route = next(
        r for r in registered.enumerate_routes(settings=options())
        if r.model_id == "ADP-W-RR"
    )
    from simple_adp.runner.vision_source import VisionSourceConvNeXtModelAdapter
    source = VisionSourceConvNeXtModelAdapter()
    monkeypatch.setenv("SIMPLE_ADP_V04_SOURCE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        source._source()
    monkeypatch.setenv("SIMPLE_ADP_V04_SOURCE_SHA256", sha)
    monkeypatch.setenv("SIMPLE_ADP_V04_SOURCE_WEIGHTS", str(tmp_path / "missing.pt"))
    with pytest.raises(ValueError, match="does not exist"):
        source._source()
    # The route itself remains explicitly source-bound.
    assert sha[:20] in route.route_label


def test_v04_source_one_real_worker_and_resume_artifact(source_file, monkeypatch, tmp_path):
    file, sha = source_file
    configure(monkeypatch, file, sha)
    suite = plan(tmp_path)
    original = next(
        r for r in suite.runs
        if r.model_adapter == "vision_v04_source" and r.route.model_id == "MLP-Fixed"
        and r.worker_settings.get("expansion_criterion") != "variance_stagnation"
    )
    record = run_spec(original.command[-1])
    assert record["status"] == "completed"
    assert record["architecture_id"] == "V04:torchvision-convnext-tiny-source-ffn"
    assert record["selected_architecture"] == [384]
    assert record["growth_site"] == "convnext:features.1.0.channel_ffn"
    assert record["source_weights_sha256"] == sha
    assert record["source_ffn_exact_at_initialization"] is True
    assert len(record["frozen_sha256"]) == 64
    assert 0 <= record["test_metrics"]["accuracy"] <= 1
    assert record["artifact_path"]
    import os
    assert os.path.isfile(record["artifact_path"])
    assert json.loads(
        open(original.command[-1], encoding="utf-8").read()
    )["model_adapter"] == "vision_v04_source"
