"""Independent process high-water evidence through the original signed runner."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.plan import SuitePlan
from simple_adp.runner.plugins import load_registries


def isolated_driver():
    filename = Path(__file__).resolve().parents[1] / "scripts" / "visual_official_fashion_matrix.py"
    spec = importlib.util.spec_from_file_location("isolated_official_matrix", filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_worker_in_fresh_process_has_own_os_pid_and_signed_result(monkeypatch, tmp_path):
    monkeypatch.setenv("SIMPLE_ADP_ENABLE_VISION", "1")
    monkeypatch.setenv("SIMPLE_ADP_DISABLE_GPU_ACCELERATORS", "1")
    monkeypatch.setenv("TRAINING_CONTROL_CPU_ONLY", "1")
    monkeypatch.setenv("TABULAR_BACKEND", "cpu")
    monkeypatch.setenv("TABULAR_SKIP_SYSTEMD_SCOPE", "1")
    registry, models = load_registries()
    options = RunnerSettings(
        tasks=("vision",), run_hardness_screen=False, repeat_count=1,
        missing_present_task_repeats=1, model_seeds=(0,),
        width_to_depth_width=2, width_to_depth_depth=1,
        max_width=512, max_depth=5, max_neurons=128,
        max_candidates=1, max_epochs=1, batch_size=16,
        smoke_test=True, cpu_only=True,
    )
    plan = SuitePlan.build(
        settings=options, results_dir=tmp_path / "results",
        run_root=tmp_path / "runs",
        dataset_providers=tuple(registry.providers.values()),
        dataset_suites=(("vision", "toy"),),
        model_adapters=tuple(models.adapters.values()),
    )
    job = next(
        job for job in plan.runs
        if job.model_adapter == "vision_inner"
        and job.route.model_id == "MLP-Fixed"
        and job.worker_settings.get("counterpart_of") is None
    )
    log = tmp_path / "worker_logs" / "one.log"
    record = isolated_driver()._run_worker_isolated(job.command[-1], log)
    assert record["status"] == "completed"
    assert record["job_id"] == job.identity.run_id
    assert record["run_signature"] == job.identity.run_signature
    observed = record["fit_process_resources"]
    assert observed["pid"] != os.getpid()
    assert observed["device"] == "cpu"
    assert observed["exact_fit_window_peak_rss_bytes"] is None
    assert observed["cuda_pytorch_allocator"] is None
    assert observed["rss_current_after_bytes"] > 0
    assert observed["lifetime_peak_rss_after_bytes"] >= (
        observed["rss_current_after_bytes"]
    )
    assert log.exists()
    assert (tmp_path / "results" / job.identity.run_id / "result.json").exists()


def test_isolated_worker_fails_closed_on_child_failure_and_bad_signature(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace

    import pytest

    module = isolated_driver()
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    spec = tmp_path / "signed.json"
    spec.write_text(json.dumps({
        "result_dir": str(result_dir),
        "identity": {"run_signature": "expected-immutable-signature"},
    }))
    log_path = tmp_path / "logs" / "worker.log"

    def fake_failure(args, stdout, stderr, check):
        assert args[-2] == "--spec"
        assert args[-1] == str(spec)
        assert check is False
        stdout.write("worker failure\n")
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(module.subprocess, "run", fake_failure)
    with pytest.raises(RuntimeError, match="exit=7"):
        module._run_worker_isolated(spec, log_path)
    assert log_path.read_text() == "worker failure\n"

    def fake_success(args, stdout, stderr, check):
        (result_dir / "result.json").write_text(json.dumps({
            "status": "completed", "run_signature": "wrong-signature",
        }))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_success)
    with pytest.raises(ValueError, match="does not match signed run"):
        module._run_worker_isolated(spec, log_path)
