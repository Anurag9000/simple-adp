from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import opf_massive_suite_runner as scheduler

PINNED_SCHEDULER_GIT_BLOB = "d1bf6afe6edda9bbcaac4fe8dde79346b4db6034"


def _git_blob_sha(data: bytes) -> str:
    import hashlib

    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def _runtime(tmp_path: Path):
    spec = scheduler.JobSpec(
        job_id="job-1",
        phase="training",
        family="test",
        case_name="case",
        repeat_index=0,
        depth=1,
        width=1,
        seed=1,
        results_dir=str(tmp_path / "results"),
        command=(sys.executable, "train.py"),
        device_capable=True,
    )
    return scheduler.JobRuntime(
        spec=spec,
        state_path=tmp_path / "job_state.json",
        log_path=tmp_path / "job.log",
    )


def test_inexact_resume_error_is_not_classified_as_retryable(tmp_path: Path) -> None:
    path = tmp_path / "job.log"
    path.write_text("ValueError: exact-resume failure: checkpoint lacks scheduler state\n")
    assert scheduler.is_inexact_resume_log(path)


def test_old_attempt_error_does_not_classify_the_current_retry(tmp_path: Path) -> None:
    path = tmp_path / "job.log"
    path.write_text(
        "=== LAUNCH job attempt=1 ===\n"
        "ValueError: exact-resume failure: checkpoint lacks global_batch\n"
        "RuntimeError: CUDA out of memory\n"
        "=== FINISH job exit=1 ===\n"
        "=== RETRY job exit=1 ===\n"
        "=== LAUNCH job attempt=2 ===\n"
        "RuntimeError: GPU backend unavailable\n"
        "=== FINISH job exit=1 ===\n"
    )
    assert not scheduler.is_inexact_resume_log(path)
    assert not scheduler.is_gpu_oom_log(path)


def test_current_attempt_error_is_classified(tmp_path: Path) -> None:
    path = tmp_path / "job.log"
    path.write_text(
        "=== LAUNCH job attempt=1 ===\n"
        "CUDA out of memory\n"
        "=== FINISH job exit=1 ===\n"
    )
    assert scheduler.is_gpu_oom_log(path)


def test_cublas_allocation_failure_is_classified_as_gpu_oom(tmp_path: Path) -> None:
    path = tmp_path / "job.log"
    path.write_text(
        "=== LAUNCH job attempt=1 ===\n"
        "RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate(handle)\n"
        "=== FINISH job exit=1 ===\n"
    )
    assert scheduler.is_gpu_oom_log(path)


def _args(**overrides):
    values = {
        "cpu_only": False,
        "host_ram_resume_avail_mib": 1280.0,
        "gpu_resume_avail_mib": 500.0,
        "max_active_gpu_jobs": 0,
        "max_active_jobs": 0,
        "scheduler": "gpu_first",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_scheduler_source_matches_current_audited_literal_blob() -> None:
    path = ROOT / "utils" / "opf_massive_suite_runner.py"
    assert _git_blob_sha(path.read_bytes()) == PINNED_SCHEDULER_GIT_BLOB


def test_pressure_resume_requires_every_paused_resource_domain() -> None:
    kwargs = dict(
        pause_reason="gpu_memory_pressure+host_ram_pressure",
        device_mode="gpu",
        host_available_mib=400.0,
        gpu_free_mib=900.0,
        host_used_mib=15000.0,
        gpu_used_mib=7000.0,
        host_resume_avail_mib=1280.0,
        gpu_resume_avail_mib=500.0,
        host_peak_used_mib=15000.0,
        gpu_peak_used_mib=7000.0,
        external_drop_threshold_mib=500.0,
    )
    assert scheduler.pressure_pause_reopened(**kwargs) is False
    kwargs["host_available_mib"] = 1600.0
    assert scheduler.pressure_pause_reopened(**kwargs) is True


def test_external_process_memory_drop_reopens_matching_pressure_domain() -> None:
    assert scheduler.pressure_pause_reopened(
        pause_reason="host_ram_pressure",
        device_mode="cpu",
        host_available_mib=200.0,
        gpu_free_mib=None,
        host_used_mib=15000.0,
        gpu_used_mib=None,
        host_resume_avail_mib=1280.0,
        gpu_resume_avail_mib=500.0,
        host_peak_used_mib=15000.0,
        gpu_peak_used_mib=0.0,
        external_drop_threshold_mib=500.0,
        external_host_drop_mib=600.0,
    ) is True
    assert scheduler.pressure_pause_reopened(
        pause_reason="gpu_memory_pressure",
        device_mode="gpu",
        host_available_mib=4000.0,
        gpu_free_mib=100.0,
        host_used_mib=12000.0,
        gpu_used_mib=7900.0,
        host_resume_avail_mib=1280.0,
        gpu_resume_avail_mib=500.0,
        host_peak_used_mib=12000.0,
        gpu_peak_used_mib=7900.0,
        external_drop_threshold_mib=500.0,
        external_gpu_drop_mib=600.0,
    ) is True


def test_cuda_oom_retries_use_ordinary_pressure_aware_lane_admission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("CPU_ONLY", "TRAINING_CONTROL_CPU_ONLY", "OPF_ADP_DISABLE_GPU_ACCELERATORS", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    runtime = _runtime(tmp_path)
    host = scheduler.MemorySample(
        total_mib=16000,
        available_mib=6000,
        used_pct=62.5,
        swap_total_mib=0,
        swap_used_mib=0,
        swap_used_pct=0.0,
    )
    gpu = scheduler.GpuMemorySample(total_mib=8000, used_mib=2000, free_mib=6000, used_pct=25.0)

    assert scheduler.select_launch_device(runtime, _args(), host, gpu, 0, 0) == "gpu"
    assert scheduler.select_launch_device(
        runtime,
        _args(),
        host,
        gpu,
        0,
        0,
        gpu_admission_enabled=False,
    ) == "cpu"

    runtime.force_cpu_after_cuda_oom = True
    assert scheduler.select_launch_device(runtime, _args(), host, gpu, 0, 0) == "gpu"
    assert scheduler.select_launch_device(
        runtime, _args(), host, gpu, 0, 0, gpu_admission_enabled=False
    ) == "cpu"



def test_cpu_child_masks_all_optional_gpu_backends(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.delenv("CPU_ONLY", raising=False)
    env = scheduler.make_child_env(
        _args(gpu_device_index=0), _runtime(tmp_path), 2, "cpu", tmp_path
    )
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["OPF_ADP_DISABLE_GPU_ACCELERATORS"] == "1"
    assert env["TRAINING_CONTROL_CPU_ONLY"] == "1"
    assert env["HIP_VISIBLE_DEVICES"] == ""
    assert env["ROCR_VISIBLE_DEVICES"] == ""
    assert env["JAX_PLATFORMS"] == "cpu"


def test_gpu_child_preserves_outer_single_gpu_assignment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("CPU_ONLY", "TRAINING_CONTROL_CPU_ONLY", "OPF_ADP_DISABLE_GPU_ACCELERATORS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    env = scheduler.make_child_env(
        _args(gpu_device_index=0), _runtime(tmp_path), 2, "gpu", tmp_path
    )
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert env["TRAINING_CONTROL_LOCAL_GPU_INDEX"] == "0"
    assert env["OPF_ADP_DISABLE_GPU_ACCELERATORS"] == "0"


def test_gpu_child_cannot_escape_cpu_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPF_ADP_DISABLE_GPU_ACCELERATORS", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="central CPU admission"):
        scheduler.make_child_env(_args(gpu_device_index=0), _runtime(tmp_path), 1, "gpu", tmp_path)
    host = scheduler.MemorySample(16000, 6000, 62.5, 0, 0, 0.0)
    gpu = scheduler.GpuMemorySample(8000, 2000, 6000, 25.0)
    assert scheduler.select_launch_device(
        _runtime(tmp_path), _args(), host, gpu, 0, 0
    ) == "cpu"


def test_masked_cuda_cannot_be_unmasked_by_gpu_admission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPF_ADP_DISABLE_GPU_ACCELERATORS", raising=False)
    monkeypatch.delenv("CPU_ONLY", raising=False)
    monkeypatch.delenv("TRAINING_CONTROL_CPU_ONLY", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(RuntimeError, match="central CPU admission"):
        scheduler.make_child_env(_args(gpu_device_index=0), _runtime(tmp_path), 1, "gpu", tmp_path)



@pytest.mark.parametrize("admission", (
    {"TRAINING_CONTROL_BACKEND": "cpu", "CUDA_VISIBLE_DEVICES": "0"},
    {"CUDA_VISIBLE_DEVICES": " -1 "},
    {"CUDA_VISIBLE_DEVICES": "  "},
))
def test_parent_cpu_admission_cannot_be_overridden_by_gpu_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, admission: dict,
) -> None:
    for name in (
        "TRAINING_CONTROL_BACKEND", "CPU_ONLY",
        "TRAINING_CONTROL_CPU_ONLY", "OPF_ADP_DISABLE_GPU_ACCELERATORS",
        "CUDA_VISIBLE_DEVICES",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in admission.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match="central CPU admission"):
        scheduler.make_child_env(
            _args(gpu_device_index=0), _runtime(tmp_path), 1, "gpu", tmp_path,
        )
    host = scheduler.MemorySample(16000, 6000, 62.5, 0, 0, 0.0)
    gpu = scheduler.GpuMemorySample(8000, 2000, 6000, 25.0)
    assert scheduler.select_launch_device(_runtime(tmp_path), _args(), host, gpu, 0, 0) == "cpu"

def test_retry_budget_zero_is_unbounded(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.retry_count = 0
    assert scheduler.should_retry(runtime, max_retries=1) is True
    runtime.retry_count = 1
    assert scheduler.should_retry(runtime, max_retries=1) is False
    assert scheduler.should_retry(runtime, max_retries=0) is True


def test_resume_launch_failure_is_persisted_as_paused_resume_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime(tmp_path)
    runtime.paused = True
    runtime.pause_reason = "gpu_memory_pressure"
    captured: dict = {}

    def fail_launch(*_args, **_kwargs):
        raise OSError("synthetic relaunch failure")

    monkeypatch.setattr(scheduler, "launch_job", fail_launch)
    monkeypatch.setattr(scheduler, "save_job_state", lambda _path, payload: captured.update(payload))
    monkeypatch.setattr(scheduler, "write_compat_state_views", lambda *_args, **_kwargs: None)

    with pytest.raises(OSError, match="synthetic relaunch failure"):
        scheduler._resume_runtime_from_pause(
            runtime,
            SimpleNamespace(),
            1,
            tmp_path,
            device_mode="gpu",
            active=[],
        )

    assert runtime.paused is True
    assert runtime.pause_reason == "gpu_memory_pressure"
    assert captured["status"] == "paused"
    assert captured["reason"] == "resume_launch_error"
    assert captured["pid"] is None


def test_inline_gate_close_and_external_reopen_logic_remains_present() -> None:
    text = (ROOT / "utils" / "opf_massive_suite_runner.py").read_text(encoding="utf-8")
    required = (
        "gpu_child_pressure = (last_child_gpu_vram_mib + gpu_buffer) > float(gpu_free_effective)",
        "host_child_pressure = (last_child_host_rss_mib + host_buffer) > float(host_available_effective)",
        "if not gpu_gate_closed and (gpu_pressure or gpu_child_pressure):",
        "if not host_gate_closed and (host_pressure or host_child_pressure):",
        "if gpu_gate_closed and gpu_external_drop:",
        "if host_gate_closed and host_external_drop:",
    )
    for snippet in required:
        assert snippet in text
