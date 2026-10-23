"""Differential contracts against the immutable copied runner snapshot."""

import subprocess
import sys
from pathlib import Path

import pytest


def test_admission_and_failure_classification_match_reference():
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from textwrap import dedent

repository = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
reference = importlib.import_module("utils.opf_massive_suite_runner")
from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.scheduler import SuiteCoordinator
from simple_adp.runner import admission
from simple_adp.runner.admission import (
    AdmissionState, LaneCounts, choose_device, pressure_pause_reopened,
    reopen_external_gates, resolve_launch_capacity, update_soft_gates,
)
from simple_adp.runner.resources import GPUMemorySample, HostMemorySample

messages = [
    "", "CUDA out of memory", "CUDNN_STATUS_ALLOC_FAILED",
    "CUBLAS_STATUS_ALLOC_FAILED", "CUBLASCreate(handle) failed",
    "out of memory in CUDA allocator", "cannot allocate memory",
    "MemoryError: failed", "process killed by OOM",
    "exact-resume failure: absent RNG", "lacks exact-resume guarantees",
    "ordinary nonzero exit", "cuda memoryerror", "out of memory",
]

def expected(path):
    if reference.is_inexact_resume_log(path):
        return "inexact_checkpoint_resume"
    if reference.is_gpu_oom_log(path):
        return "cuda_oom"
    if reference.is_memory_oom_log(path):
        return "memory_oom"
    return "nonzero_exit:9"

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    coordinator = SuiteCoordinator(
        SimpleNamespace(), RunnerSettings(), run_root=root / "run",
        results_dir=root / "results", repository_root=repository,
    )
    cases = []
    for index, message in enumerate(messages):
        path = root / f"marker-{index}.log"
        path.write_text(f"=== LAUNCH 1 ===\n{message}\n", encoding="utf-8")
        cases.append(path)

    marker = b"=== LAUNCH 1 ===\n"
    boundary = root / "boundary.log"
    boundary.write_bytes(
        b"x" * (64 * 1024 - len(marker) + 3) + marker
        + b"CUBLAS_STATUS_ALLOC_FAILED" + b"z" * 4100
    )
    cases.append(boundary)

    older_attempt = root / "older-attempt.log"
    older_attempt.write_text(
        "=== LAUNCH old ===\nCUDA out of memory\n"
        "=== LAUNCH latest ===\nordinary failure\n",
        encoding="utf-8",
    )
    cases.append(older_attempt)

    for path in cases:
        source_result = expected(path)
        generic_result = coordinator._classify_failure(
            SimpleNamespace(log_path=str(path)), 9,
        )
        assert generic_result == source_result, (path.name, source_result, generic_result)

# Compare the actual source selector and its caller-supplied admission booleans
# with the generic selector over CPU masks, host readiness, caps, gates, and
# lane holds. This includes the source CPU-only early-return edge at a full cap.
checked_selector = 0
source_parent_cpu_masked = reference._parent_cpu_masked
generic_parent_cpu_masked = admission.parent_cpu_masked
for cpu_only in (False, True):
    for parent_masked in (False, True):
        reference._parent_cpu_masked = lambda value=parent_masked: value
        admission.parent_cpu_masked = lambda value=parent_masked: value
        for host_available in (1279.0, 1280.0):
            for max_active_jobs in (0, 1):
                for active_count in (0, 1):
                    for cpu_enabled in (False, True):
                        for host_gate in (False, True):
                            for cpu_hold_ready in (False, True):
                                now = 10.0
                                host = HostMemorySample(4096, host_available, 50, 0, 0, 0)
                                state = AdmissionState(
                                    cpu_launches_enabled=cpu_enabled,
                                    host_gate_closed=host_gate,
                                    cpu_hold_until=0 if cpu_hold_ready else now + 1,
                                )
                                settings = RunnerSettings(
                                    cpu_only=cpu_only,
                                    max_active_jobs=max_active_jobs,
                                )
                                args = SimpleNamespace(
                                    cpu_only=cpu_only,
                                    scheduler="gpu_first",
                                    host_ram_resume_avail_mib=1280,
                                    gpu_resume_avail_mib=500,
                                    max_active_jobs=max_active_jobs,
                                    max_active_gpu_jobs=0,
                                )
                                source_host_admission = (
                                    cpu_enabled and not host_gate
                                    and host_available >= 1280 and cpu_hold_ready
                                )
                                reference_device = reference.select_launch_device(
                                    SimpleNamespace(), args,
                                    SimpleNamespace(available_mib=host_available),
                                    None, 0, active_count,
                                    host_admission_enabled=source_host_admission,
                                    gpu_admission_enabled=False,
                                )
                                generic_device = choose_device(
                                    settings, state, host=host, gpu=None,
                                    counts=LaneCounts(active_count, 0, active_count),
                                    now=now,
                                )
                                assert generic_device == reference_device, (
                                    cpu_only, parent_masked, host_available,
                                    max_active_jobs, active_count, cpu_enabled,
                                    host_gate, cpu_hold_ready, reference_device,
                                    generic_device,
                                )
                                checked_selector += 1
assert checked_selector == 256
reference._parent_cpu_masked = source_parent_cpu_masked
admission.parent_cpu_masked = generic_parent_cpu_masked

# Capacity is resolved before the controller separately applies its
# active_count + 1 ceiling. Compare that pure helper across zero, sub-unit,
# exact, and non-exact reserve boundaries, including a missing GPU sensor.
checked_capacity = 0
for concurrency in (0, 1, 2, 5):
    for host_available in (0.0, 0.1, 1.0, 9.0, 10.0):
        for host_reserve in (0.0, 0.1, 1.0, 3.0):
            for gpu_free in (None, 0.0, 0.1, 1.0, 9.0, 10.0):
                for gpu_reserve in (0.0, 0.1, 1.0, 3.0):
                    for max_active_jobs in (0, 1, 4):
                        args = SimpleNamespace(
                            concurrency=concurrency,
                            concurrency_file=None,
                            launch_host_reserve_mib=host_reserve,
                            launch_gpu_reserve_mib=gpu_reserve,
                            max_active_jobs=max_active_jobs,
                        )
                        settings = RunnerSettings(
                            concurrency=concurrency,
                            launch_host_reserve_mib=host_reserve,
                            launch_gpu_reserve_mib=gpu_reserve,
                            max_active_jobs=max_active_jobs,
                        )
                        source_capacity = reference.resolve_launch_capacity(
                            args, host_available, gpu_free,
                        )
                        generic_capacity = resolve_launch_capacity(
                            settings,
                            concurrency=settings.effective_concurrency,
                            host_available_mib=host_available,
                            gpu=None if gpu_free is None else GPUMemorySample(
                                total_mib=max(gpu_free, 0.0),
                                used_mib=0.0,
                                free_mib=gpu_free,
                                used_percent=0.0,
                            ),
                            active_count=0,
                        )
                        assert generic_capacity == source_capacity, (
                            concurrency, host_available, host_reserve, gpu_free,
                            gpu_reserve, max_active_jobs, source_capacity,
                            generic_capacity,
                        )
                        checked_capacity += 1
assert checked_capacity == 5760

# Compare every required pressure domain for CPU/GPU pauses across strict and
# inclusive resource/drop boundaries, including missing GPU sensors and the
# fallback GPU-domain inference for unlabelled GPU pauses.
checked_resume = 0
reasons = (
    "", "host_ram_pressure", "swap_pressure", "gpu_memory_pressure",
    "host_ram_pressure+gpu_memory_pressure", "peer_cuda_oom",
)
for reason in reasons:
    for device in ("cpu", "gpu"):
        for host_available in (1279.0, 1280.0):
            for gpu_free in (None, 499.0, 500.0):
                for host_used in (900.0, 1000.0):
                    for host_peak in (0.0, 1499.0, 1500.0):
                        for host_drop in (0.0, 499.0, 500.0):
                            for gpu_used in (None, 400.0, 500.0):
                                for gpu_peak in (0.0, 999.0, 1000.0):
                                    for gpu_drop in (0.0, 499.0, 500.0):
                                        args = SimpleNamespace(
                                            host_resume_avail_mib=1280.0,
                                            gpu_resume_avail_mib=500.0,
                                            external_drop_threshold_mib=500.0,
                                        )
                                        settings = RunnerSettings(
                                            host_ram_resume_avail_mib=1280.0,
                                            gpu_resume_avail_mib=500.0,
                                            external_memory_drop_mib=500.0,
                                        )
                                        source_ready = reference.pressure_pause_reopened(
                                            pause_reason=reason,
                                            device_mode=device,
                                            host_available_mib=host_available,
                                            gpu_free_mib=gpu_free,
                                            host_used_mib=host_used,
                                            gpu_used_mib=gpu_used,
                                            host_resume_avail_mib=1280.0,
                                            gpu_resume_avail_mib=500.0,
                                            host_peak_used_mib=host_peak,
                                            gpu_peak_used_mib=gpu_peak,
                                            external_drop_threshold_mib=500.0,
                                            external_host_drop_mib=host_drop,
                                            external_gpu_drop_mib=gpu_drop,
                                        )
                                        generic_ready = pressure_pause_reopened(
                                            pause_reason=reason,
                                            device=device,
                                            host_available_mib=host_available,
                                            gpu_free_mib=gpu_free,
                                            host_used_mib=host_used,
                                            gpu_used_mib=gpu_used,
                                            settings=settings,
                                            host_peak_used_mib=host_peak,
                                            gpu_peak_used_mib=gpu_peak,
                                            external_host_drop_mib=host_drop,
                                            external_gpu_drop_mib=gpu_drop,
                                        )
                                        assert generic_ready is source_ready, (
                                            reason, device, host_available, gpu_free,
                                            host_used, host_peak, host_drop, gpu_used,
                                            gpu_peak, gpu_drop, source_ready,
                                            generic_ready,
                                        )
                                        checked_resume += 1
assert checked_resume == 34992

# Execute the source's actual inline gate-transition statements, rather than a
# hand-copied oracle, and compare close/reopen state with the generic helpers.
reference_lines = (
    repository / "references" / "opf_adp_runner_suite" / "utils"
    / "opf_massive_suite_runner.py"
).read_text(encoding="utf-8").splitlines()
source_gate_block = compile(
    dedent("\n".join(reference_lines[3074:3124])),
    "opf_massive_suite_runner.py:3075-3124",
    "exec",
)
class NullLogger:
    @staticmethod
    def info(*_args, **_kwargs):
        pass

checked_gates = 0
for gpu_closed in (False, True):
    for host_closed in (False, True):
        for gpu_enabled in (False, True):
            for cpu_enabled in (False, True):
                for latest_gpu in (None, "gpu-job"):
                    for latest_host in (None, "host-job"):
                        for gpu_pressure_value in (False, True):
                            for host_pressure_value in (False, True):
                                for gpu_free in (None, 500.0, 501.0):
                                    for child_gpu in (399.0, 400.0, 401.0):
                                        for host_available in (500.0, 1000.0, 1001.0):
                                            for child_host in (0.0, 499.0, 500.0):
                                                for gpu_drop in (False, True):
                                                    for host_drop in (False, True):
                                                        settings = RunnerSettings(
                                                            child_gpu_gate_buffer_mib=100.0,
                                                            child_host_gate_buffer_mib=500.0,
                                                        )
                                                        state = AdmissionState(
                                                            gpu_gate_closed=gpu_closed,
                                                            host_gate_closed=host_closed,
                                                            gpu_gate_lock_job_id="old-gpu-lock" if gpu_closed else None,
                                                            host_gate_lock_job_id="old-host-lock" if host_closed else None,
                                                            gpu_launches_enabled=gpu_enabled,
                                                            cpu_launches_enabled=cpu_enabled,
                                                        )
                                                        update_soft_gates(
                                                            state,
                                                            settings=settings,
                                                            host_pressure=host_pressure_value,
                                                            gpu_pressure=gpu_pressure_value,
                                                            gpu_sample_present=gpu_free is not None,
                                                            latest_gpu_job=latest_gpu,
                                                            latest_host_job=latest_host,
                                                            latest_child_gpu_mib=child_gpu,
                                                            latest_child_host_mib=child_host,
                                                            host_available_mib=host_available,
                                                            gpu_free_mib=gpu_free,
                                                        )
                                                        reopen_external_gates(
                                                            state, host_drop=host_drop,
                                                            gpu_drop=gpu_drop,
                                                        )
                                                        def runtime(job_id):
                                                            return None if job_id is None else SimpleNamespace(
                                                                spec=SimpleNamespace(job_id=job_id),
                                                            )
                                                        source_state = {
                                                            "args": SimpleNamespace(
                                                                gpu_child_gate_close_buffer_mib=100.0,
                                                                host_child_gate_close_buffer_mib=500.0,
                                                            ),
                                                            "latest_gpu_runtime": runtime(latest_gpu),
                                                            "latest_runtime": runtime(latest_host),
                                                            "gpu_free_effective": gpu_free,
                                                            "last_child_gpu_vram_mib": child_gpu,
                                                            "last_child_host_rss_mib": child_host,
                                                            "gpu_pressure": gpu_pressure_value,
                                                            "host_pressure": host_pressure_value,
                                                            "gpu_gate_closed": gpu_closed,
                                                            "host_gate_closed": host_closed,
                                                            "gpu_gate_lock_job_id": "old-gpu-lock" if gpu_closed else None,
                                                            "host_gate_lock_job_id": "old-host-lock" if host_closed else None,
                                                            "gpu_launches_enabled": gpu_enabled,
                                                            "cpu_launches_enabled": cpu_enabled,
                                                            "gpu_external_drop": gpu_drop,
                                                            "host_external_drop": host_drop,
                                                            "host_available_effective": host_available,
                                                            "logger": NullLogger(),
                                                        }
                                                        exec(source_gate_block, source_state)
                                                        actual = (
                                                            state.gpu_gate_closed,
                                                            state.host_gate_closed,
                                                            state.gpu_gate_lock_job_id,
                                                            state.host_gate_lock_job_id,
                                                            state.gpu_launches_enabled,
                                                            state.cpu_launches_enabled,
                                                        )
                                                        expected = tuple(
                                                            source_state[key] for key in (
                                                                "gpu_gate_closed", "host_gate_closed",
                                                                "gpu_gate_lock_job_id", "host_gate_lock_job_id",
                                                                "gpu_launches_enabled", "cpu_launches_enabled",
                                                            )
                                                        )
                                                        assert actual == expected, (
                                                            gpu_closed, host_closed, gpu_enabled,
                                                            cpu_enabled, latest_gpu, latest_host,
                                                            gpu_pressure_value, host_pressure_value,
                                                            gpu_free, child_gpu, host_available,
                                                            child_host, gpu_drop, host_drop,
                                                            expected, actual,
                                                        )
                                                        checked_gates += 1
assert checked_gates == 82944

# Execute the source's actual candidate filtering and sorting statements and
# compare the chosen victim with the generic coordinator. Exercise input-order
# permutations, GPU/host/both-pressure priority, lane filters, stale/paused
# runtimes, memory-footprint ordering, and architecture/job-id tie breaks.
from itertools import permutations
from simple_adp.runner.resources import ProcessMemorySample

source_pressure_block = compile(
    dedent("\n".join(reference_lines[3311:3350])),
    "opf_massive_suite_runner.py:3312-3350",
    "exec",
)
class FakeProcess:
    def __init__(self, alive):
        self.alive = alive
    def poll(self):
        return None if self.alive else 1

runtime_specs = (
    # id, lane, alive, paused, depth, width, host RSS, GPU VRAM
    ("gpu-z", "gpu", True, False, 2, 4, 80.0, 300.0),
    ("gpu-a", "gpu", True, False, 2, 4, 80.0, 300.0),
    ("cpu-big", "cpu", True, False, 9, 99, 900.0, 0.0),
    ("gpu-paused", "gpu", True, True, 12, 99, 999.0, 999.0),
    ("gpu-dead", "gpu", False, False, 13, 99, 999.0, 999.0),
)
checked_victims = 0
for gpu_pressure_value, host_pressure_value in ((True, False), (False, True), (True, True)):
    for ordering in permutations(runtime_specs):
        runtimes = []
        samples = {}
        for job_id, lane, alive, paused, depth, width, host_rss, gpu_vram in ordering:
            route = SimpleNamespace(fixed_grid=(), init_depth=depth, init_width=width)
            spec = SimpleNamespace(
                depth=depth, width=width, job_id=job_id,
                route=route, worker_settings={"init_depth": depth, "init_width": width},
            )
            runtime = SimpleNamespace(
                spec=spec, proc=FakeProcess(alive), paused=paused,
                device_mode=lane, live=alive and not paused,
                is_gpu=lane == "gpu", job_id=job_id,
            )
            runtimes.append(runtime)
            samples[job_id] = (host_rss, gpu_vram)

        source_state = {
            "pressure_kind": "gpu" if gpu_pressure_value else "host",
            "gpu_pressure": gpu_pressure_value,
            "host_pressure": host_pressure_value,
            "pressure_reasons": (
                (["gpu_memory_pressure"] if gpu_pressure_value else [])
                + (["host_ram_pressure"] if host_pressure_value else [])
            ),
            "args": SimpleNamespace(gpu_device_index=0),
            "active": runtimes,
            "sample_runtime_child_memory": lambda runtime, _device: samples[runtime.spec.job_id],
            "_job_pressure_score": lambda runtime: (
                int(runtime.spec.depth), int(runtime.spec.width), str(runtime.spec.job_id),
            ),
        }
        exec(source_pressure_block, source_state)
        source_selected = source_state["candidates"][0][1].spec.job_id if source_state["candidates"] else None

        coordinator = SuiteCoordinator.__new__(SuiteCoordinator)
        coordinator.memory = SimpleNamespace(values={})
        coordinator.admission = SimpleNamespace(
            pressure_backoff_pending=False, last_backoff_reason=None,
        )
        coordinator._sample_runtime_child = lambda runtime: ProcessMemorySample(
            host_rss_mib=samples[runtime.job_id][0],
            gpu_vram_mib=samples[runtime.job_id][1],
        )
        generic_selected = []
        coordinator._pause = lambda runtime, *_args: generic_selected.append(runtime.job_id) or False
        coordinator._save_controller_state = lambda: None
        snapshot = SimpleNamespace(
            gpu_pressure=gpu_pressure_value, host_pressure=host_pressure_value,
            gpu=SimpleNamespace(used_mib=10.0),
            host=SimpleNamespace(total_mib=1000.0, available_mib=100.0),
        )
        coordinator._suspend_pressure_victim(runtimes, [], snapshot)
        generic_choice = generic_selected[0] if generic_selected else None
        assert source_selected == generic_choice, (
            gpu_pressure_value, host_pressure_value,
            [runtime.job_id for runtime in runtimes], source_selected, generic_choice,
        )
        checked_victims += 1
assert checked_victims == 360

# Compare pure Linux runtime-tuning helpers against the copied implementation
# over bounded partitions, parser edge strings, stable slots, and environment /
# explicit concurrency-hint precedence.
from simple_adp.runner import runtime_tuning as generic_tuning
source_tuning = importlib.import_module("utils.runtime_tuning")
checked_tuning = 0
for cpu_count in range(9):
    cpus = tuple(range(2, 2 + cpu_count))
    for parts in range(-2, 11):
        for slot in range(-2, 11):
            assert generic_tuning.partition_cpus(cpus, parts, slot) == source_tuning._partition_cpus(
                cpus, parts, slot,
            ), (cpus, parts, slot)
            checked_tuning += 1

for text in (
    "", ",,,", "0", "0-0", "0-3", "3-0", " 1, 4-6, 9 ",
    "1,1,2", "-1", "1--2", "1-x", "1-2-3", "  ",
):
    outcomes = []
    for parser in (generic_tuning.parse_cpu_list, source_tuning._parse_cpu_list):
        try:
            outcomes.append(("value", parser(text)))
        except Exception as exc:
            outcomes.append(("error", type(exc)))
    assert outcomes[0] == outcomes[1], (text, outcomes)
    checked_tuning += 1

for key in ("", "job", "run/04", "Δ-job"):
    for parts in range(-2, 11):
        assert generic_tuning.stable_cpu_slot(key, parts) == source_tuning._deterministic_slot(
            key, parts,
        ), (key, parts)
        checked_tuning += 1

original_hint = __import__("os").environ.get("TABULAR_CPU_JOB_CONCURRENCY")
try:
    for env_hint in (None, "", "0", "-1", "3", "bad", " 4 "):
        if env_hint is None:
            __import__("os").environ.pop("TABULAR_CPU_JOB_CONCURRENCY", None)
        else:
            __import__("os").environ["TABULAR_CPU_JOB_CONCURRENCY"] = env_hint
        for explicit_hint in (None, 0, -2, "3", "bad", True, 2.5):
            assert generic_tuning.current_concurrency_hint(explicit_hint) == source_tuning.current_concurrency_hint(
                explicit_hint,
            ), (env_hint, explicit_hint)
            checked_tuning += 1
finally:
    if original_hint is None:
        __import__("os").environ.pop("TABULAR_CPU_JOB_CONCURRENCY", None)
    else:
        __import__("os").environ["TABULAR_CPU_JOB_CONCURRENCY"] = original_hint
assert checked_tuning == 1635

# Compare the central source and generic child-environment CPU controls over
# core counts, explicit and inherited concurrency hints, optional affinity
# overrides/slots, and inherited shared-mode truth values.
import os

original_global_hint = os.environ.get("TABULAR_CPU_JOB_CONCURRENCY")
original_global_affinity = os.environ.get("TABULAR_CPU_AFFINITY_CPUS")
source_detect = source_tuning.detect_cpu_cores
source_affinity = source_tuning._current_affinity_cpus
generic_detect = generic_tuning.detected_cpus
checked_child_env = 0
tuning_keys = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "GOTO_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "TORCH_NUM_THREADS", "TORCH_INTEROP_THREADS", "OMP_DYNAMIC",
    "MKL_DYNAMIC", "OMP_WAIT_POLICY", "TABULAR_CPU_THREADS",
    "TABULAR_CPU_WORKERS", "TABULAR_CPU_CORES",
    "TABULAR_CPU_JOB_CONCURRENCY", "TABULAR_CPU_AFFINITY_CPUS",
    "GOMP_CPU_AFFINITY", "OMP_PROC_BIND", "OMP_PLACES", "KMP_AFFINITY",
)
try:
    os.environ.pop("TABULAR_CPU_AFFINITY_CPUS", None)
    for cpu_count in (1, 2, 5):
        cpus = tuple(2 + index * 2 for index in range(cpu_count))
        source_tuning.detect_cpu_cores = lambda count=cpu_count: count
        source_tuning._current_affinity_cpus = lambda values=cpus: values
        generic_tuning.detected_cpus = lambda values=cpus: values
        for inherited_hint in (None, "3", "bad", "0"):
            if inherited_hint is None:
                os.environ.pop("TABULAR_CPU_JOB_CONCURRENCY", None)
            else:
                os.environ["TABULAR_CPU_JOB_CONCURRENCY"] = inherited_hint
            for explicit_hint in (None, 1, 2, 9):
                for affinity_value in (None, "2,4,6"):
                    for shared_value in (None, "1", "0"):
                        base_env = {"CUDA_VISIBLE_DEVICES": "0"}
                        if affinity_value is not None:
                            base_env["TABULAR_CPU_AFFINITY_CPUS"] = affinity_value
                        if shared_value is not None:
                            base_env["TABULAR_CHILD_SHARED_CPU"] = shared_value
                        source_env = source_tuning.launcher_child_env(
                            base_env, concurrency_hint=explicit_hint,
                            job_key="differential-job", affinity_slot=1,
                            shared_cpu=False,
                        )
                        generic_env = generic_tuning.launcher_child_env(
                            base_env, concurrency_hint=explicit_hint,
                            job_key="differential-job", affinity_slot=1,
                            shared_cpu=False,
                        )
                        for key in tuning_keys:
                            assert generic_env.get(key) == source_env.get(key), (
                                cpu_count, inherited_hint, explicit_hint,
                                affinity_value, shared_value, key,
                                source_env.get(key), generic_env.get(key),
                            )
                        checked_child_env += 1
finally:
    source_tuning.detect_cpu_cores = source_detect
    source_tuning._current_affinity_cpus = source_affinity
    generic_tuning.detected_cpus = generic_detect
    if original_global_hint is None:
        os.environ.pop("TABULAR_CPU_JOB_CONCURRENCY", None)
    else:
        os.environ["TABULAR_CPU_JOB_CONCURRENCY"] = original_global_hint
    if original_global_affinity is None:
        os.environ.pop("TABULAR_CPU_AFFINITY_CPUS", None)
    else:
        os.environ["TABULAR_CPU_AFFINITY_CPUS"] = original_global_affinity
assert checked_child_env == 288

# Compare device-specific mapping performed by the source job builder with the
# generic runtime environment builder. Only the intentionally renamed generic
# variables are normalized; acceptance/rejection and device visibility must
# otherwise agree across masks and parent CPU-only controls.
source_core_detector = source_tuning.detect_cpu_cores
source_cpu_order = source_tuning._current_affinity_cpus
generic_cpu_detector = generic_tuning.detected_cpus
source_child_builder = reference.make_child_env
device_env_keys = (
    "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
    "JAX_PLATFORMS", "JAX_PLATFORM_NAME", "TRAINING_CONTROL_CPU_ONLY",
    "TRAINING_CONTROL_LOCAL_GPU_INDEX", "TABULAR_LOCAL_GPU_INDEX",
    "PYTORCH_CUDA_ALLOC_CONF", "CUBLAS_WORKSPACE_CONFIG",
)
parent_mask_options = (
    {},
    {"TRAINING_CONTROL_CPU_ONLY": "1"},
    {"OPF_ADP_DISABLE_GPU_ACCELERATORS": "1"},
    {"TRAINING_CONTROL_BACKEND": "cpu"},
)
device_masks = (None, "", "-1", "0", "2", "0,2", "0,1,2", "0,,2", "bad", " ", " 2 ")
checked_device_env = 0
def capture_device(outcome):
    try:
        return ("value", outcome())
    except Exception as exc:
        return ("error", type(exc))

try:
    source_tuning.detect_cpu_cores = lambda: 4
    source_tuning._current_affinity_cpus = lambda: (0, 1, 2, 3)
    generic_tuning.detected_cpus = lambda: (0, 1, 2, 3)
    for inherited_mask in device_masks:
        for device_index in (0, 1, 2):
            for mode in ("cpu", "gpu"):
                for parent_flags in parent_mask_options:
                    for key in (
                        "CUDA_VISIBLE_DEVICES", "TRAINING_CONTROL_CPU_ONLY",
                        "OPF_ADP_DISABLE_GPU_ACCELERATORS", "TRAINING_CONTROL_BACKEND",
                    ):
                        os.environ.pop(key, None)
                    if inherited_mask is not None:
                        os.environ["CUDA_VISIBLE_DEVICES"] = inherited_mask
                    os.environ.update(parent_flags)
                    class SourceSpec:
                        results_dir = "/tmp/results"
                        job_id = "device-differential"
                        case_name = "case"
                        repeat_index = 1
                        phase = "phase"
                    runtime = SimpleNamespace(spec=SourceSpec())
                    source_outcome = capture_device(lambda: source_child_builder(
                        SimpleNamespace(gpu_device_index=device_index), runtime, 2,
                        mode, Path("/tmp/run"),
                    ))
                    generic_outcome = capture_device(lambda: generic_tuning.launcher_child_env(
                        dict(os.environ), job_key="device-differential",
                        launch_budget=2, shared_cpu=False,
                        cpu_only=(mode == "cpu"), device_index=device_index,
                        device_mode=mode,
                    ))
                    assert (source_outcome[0] == "value") == (generic_outcome[0] == "value"), (
                        inherited_mask, device_index, mode, parent_flags,
                        source_outcome, generic_outcome,
                    )
                    if source_outcome[0] == "value":
                        source_env = source_outcome[1]
                        generic_env = generic_outcome[1]
                        for source_key, generic_key in (
                            ("OPF_ADP_DISABLE_GPU_ACCELERATORS", "SIMPLE_ADP_DISABLE_GPU_ACCELERATORS"),
                            ("TRAINING_CONTROL_BACKEND", "TABULAR_BACKEND"),
                            ("TRAINING_CONTROL_LOCAL_GPU_INDEX", "TABULAR_LOCAL_GPU_INDEX"),
                        ):
                            assert generic_env.get(generic_key) == source_env.get(source_key), (
                                inherited_mask, device_index, mode, parent_flags,
                                source_key, source_env.get(source_key),
                                generic_key, generic_env.get(generic_key),
                            )
                        for key in device_env_keys:
                            if key in {"TRAINING_CONTROL_LOCAL_GPU_INDEX", "TABULAR_LOCAL_GPU_INDEX"}:
                                continue
                            assert generic_env.get(key) == source_env.get(key), (
                                inherited_mask, device_index, mode, parent_flags,
                                key, source_env.get(key), generic_env.get(key),
                            )
                    checked_device_env += 1
finally:
    source_tuning.detect_cpu_cores = source_core_detector
    source_tuning._current_affinity_cpus = source_cpu_order
    generic_tuning.detected_cpus = generic_cpu_detector
assert checked_device_env == 264

# The standalone wrapper's host and GPU resume conditions are separate gates;
# compare the extracted generic helper with the copied helper at strict and
# inclusive threshold boundaries, including disabled/nonpositive thresholds.
from simple_adp.runner.pressure_wrapper import _external_drop_reopens_gates
source_pressure = importlib.import_module("utils.pressure_aware_wrapper")
checked_wrapper_gates = 0
for host_ready in (False, True):
    for gpu_ready in (False, True):
        for host_drop in (-1.0, 0.0, 499.0, 500.0, 501.0):
            for gpu_drop in (-1.0, 0.0, 499.0, 500.0, 501.0):
                for threshold in (-1.0, 0.0, 0.1, 499.0, 500.0, 501.0):
                    source_result = source_pressure._external_drop_reopens_gates(
                        host_ready=host_ready, gpu_ready=gpu_ready,
                        host_drop_mib=host_drop, gpu_drop_mib=gpu_drop,
                        threshold_mib=threshold,
                    )
                    generic_result = _external_drop_reopens_gates(
                        host_ready=host_ready, gpu_ready=gpu_ready,
                        host_drop_mib=host_drop, gpu_drop_mib=gpu_drop,
                        threshold_mib=threshold,
                    )
                    assert generic_result is source_result, (
                        host_ready, gpu_ready, host_drop, gpu_drop, threshold,
                    )
                    checked_wrapper_gates += 1
assert checked_wrapper_gates == 600

# Compare source and generic GPU sensor parsing plus Python GPU usage parsing.
# The source and generic modules use the shared subprocess/psutil modules, so
# the same fixed command output is observed by both implementations.
from simple_adp.runner import pressure_wrapper as generic_pressure_module
import utils.pressure_aware_wrapper as source_pressure_module

original_check_output = source_pressure_module.subprocess.check_output
original_process_iter = source_pressure_module.psutil.process_iter
checked_wrapper_samplers = 0
gpu_payloads = (
    b"100,0\n", b"100.9,50.9\n", b"100,-2\n", b"100,200\n",
    b"0,0\n", b"-1,20\n", b"garbage\n", b"", b"100, 50\n100,40\n", None,
)
try:
    source_pressure_module.psutil.process_iter = lambda *_args, **_kwargs: iter(())
    for payload in gpu_payloads:
        def check_output(*_args, **_kwargs):
            if payload is None:
                raise OSError("synthetic nvidia-smi failure")
            return payload
        source_pressure_module.subprocess.check_output = check_output
        source_gpu = source_pressure_module.sample_gpu_memory_pressure(0)
        generic_gpu = generic_pressure_module._gpu_pressure_sample(0)
        assert generic_gpu == source_gpu, (payload, source_gpu, generic_gpu)

        source_python_gpu = source_pressure_module.sample_python_gpu_memory_mib(0)
        generic_python_gpu = generic_pressure_module._sample_python_usage(0)[1]
        assert generic_python_gpu == source_python_gpu, (payload, source_python_gpu, generic_python_gpu)
        checked_wrapper_samplers += 2
finally:
    source_pressure_module.subprocess.check_output = original_check_output
    source_pressure_module.psutil.process_iter = original_process_iter
assert checked_wrapper_samplers == 20

# Compare process-name RSS accounting and its intentionally asymmetric error
# boundary: documented per-process failures are skipped, while iterator-level
# and unexpected record failures propagate.
class SyntheticProcess:
    def __init__(self, info=None, error=None):
        self._info = info
        self._error = error
    @property
    def info(self):
        if self._error is not None:
            raise self._error
        return self._info

rss_cases = (
    [
        SyntheticProcess({"name": "python3", "memory_info": SimpleNamespace(rss=1048576)}),
        SyntheticProcess({"name": "MyPythonWorker", "memory_info": SimpleNamespace(rss=524288)}),
        SyntheticProcess({"name": "custom_runtime", "memory_info": SimpleNamespace(rss=999999)}),
        SyntheticProcess({"name": None, "memory_info": SimpleNamespace(rss=100)}),
    ],
    [
        SyntheticProcess(error=source_pressure_module.psutil.NoSuchProcess(1)),
        SyntheticProcess(error=source_pressure_module.psutil.AccessDenied(2)),
        SyntheticProcess(error=source_pressure_module.psutil.ZombieProcess(3)),
        SyntheticProcess(error=TypeError("bad record")),
        SyntheticProcess(error=KeyError("memory_info")),
        SyntheticProcess({"name": "python", "memory_info": SimpleNamespace(rss=2048)}),
    ],
    [SyntheticProcess({"name": "python", "memory_info": SimpleNamespace(rss=1)})],
    [SyntheticProcess(error=AttributeError("missing cached rss"))],
)

def capture(outcome):
    try:
        return ("value", outcome())
    except Exception as exc:
        return ("error", type(exc))

source_pressure_module.subprocess.check_output = lambda *_args, **_kwargs: b""
for processes in rss_cases:
    source_pressure_module.psutil.process_iter = lambda *_args, rows=processes, **_kwargs: iter(rows)
    source_result = capture(source_pressure_module.sample_python_host_memory_mib)
    generic_result = capture(lambda: generic_pressure_module._sample_python_usage(0)[0])
    assert generic_result == source_result, (processes, source_result, generic_result)
    checked_wrapper_samplers += 1

def broken_iteration():
    yield SyntheticProcess({"name": "python", "memory_info": SimpleNamespace(rss=100)})
    raise RuntimeError("iterator failed")

source_pressure_module.psutil.process_iter = lambda *_args, **_kwargs: broken_iteration()
source_result = capture(source_pressure_module.sample_python_host_memory_mib)
generic_result = capture(lambda: generic_pressure_module._sample_python_usage(0)[0])
assert generic_result == source_result == ("error", RuntimeError)
checked_wrapper_samplers += 1
def failed_iterator_construction(*_args, **_kwargs):
    raise OSError("process iterator unavailable")

source_pressure_module.psutil.process_iter = failed_iterator_construction
source_result = capture(source_pressure_module.sample_python_host_memory_mib)
generic_result = capture(lambda: generic_pressure_module._sample_python_usage(0)[0])
assert generic_result == source_result == ("error", OSError)
checked_wrapper_samplers += 1
assert checked_wrapper_samplers == 26
print(
    f"{len(cases)} classifier, {checked_selector} selector, and "
    f"{checked_capacity} capacity, {checked_resume} resume, and "
    f"{checked_gates} gate, {checked_victims} pressure-victim, "
    f"{checked_tuning} runtime-tuning, {checked_child_env} child-env, "
    f"{checked_device_env} device-env, and "
    f"{checked_wrapper_gates} "
    f"standalone-wrapper-gate and {checked_wrapper_samplers} "
    f"standalone-wrapper-sampler "
    f"source-vs-generic cases matched"
)
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository)],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == (
        "16 classifier, 256 selector, and 5760 capacity, 34992 resume, and "
        "82944 gate, 360 pressure-victim, 1635 runtime-tuning, 288 child-env, "
        "264 device-env, and 600 "
        "standalone-wrapper-gate and 26 standalone-wrapper-sampler "
        "source-vs-generic cases matched"
    )


def test_systemd_scope_command_matches_linux_reference():
    """Compare the exact systemd scope argv and version gate with the snapshot."""
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

repository = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
sys.path.insert(0, str(repository / "src"))
source = importlib.import_module("utils.runtime_tuning")
from simple_adp.runner import runtime_tuning as generic

def capture_source(version):
    captured = {}
    source.os.name = "posix"
    source.sys.argv = ["simple-adp-suite", "--probe"]
    source.os.environ.pop("TABULAR_SKIP_SYSTEMD_SCOPE", None)
    source.os.environ.pop("TABULAR_SYSTEMD_SCOPED", None)
    source.shutil.which = lambda name: "/usr/bin/" + name
    source._current_affinity_cpus = lambda: (1, 3, 4, 8)
    source.subprocess.run = lambda *args, **kwargs: SimpleNamespace(stdout=version)
    def execvpe(executable, argv, env):
        captured.update(executable=executable, argv=list(argv), env=dict(env))
        raise OSError("expected mocked exec fallback")
    source.os.execvpe = execvpe
    source._maybe_reexec_under_systemd_scope("audit")
    return captured

def capture_generic(version):
    captured = {}
    generic.os.name = "posix"
    generic.sys.argv = ["simple-adp-suite", "--probe"]
    generic.sys.orig_argv = [generic.sys.executable, "simple-adp-suite", "--probe"]
    generic.os.environ.pop("TABULAR_SKIP_SYSTEMD_SCOPE", None)
    generic.os.environ.pop("TABULAR_SYSTEMD_SCOPED", None)
    generic.shutil_which = lambda name: "/usr/bin/" + name
    generic.detected_cpus = lambda: (1, 3, 4, 8)
    generic.subprocess.run = lambda *args, **kwargs: SimpleNamespace(stdout=version)
    def execvpe(executable, argv, env):
        captured.update(executable=executable, argv=list(argv), env=dict(env))
        raise OSError("expected mocked exec fallback")
    generic.os.execvpe = execvpe
    assert generic.maybe_reexec_systemd_scope(label="audit") is False
    return captured

for version in ("systemd 252 (test)", "systemd 253 (test)", "systemd invalid", "other 999"):
    left = capture_source(version)
    right = capture_generic(version)
    assert left == right, (version, left, right)
print("4 exact systemd command/version cases matched")
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository)],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "4 exact systemd command/version cases matched"


def test_linux_child_cpu_environment_matches_reference_runtime_tuning():
    """Differentially compare CPU budgets, affinity, and thread env controls."""
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import itertools
import os
import sys
from pathlib import Path

repository = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
sys.path.insert(0, str(repository / "src"))
source = importlib.import_module("utils.runtime_tuning")
from simple_adp.runner import runtime_tuning as generic

cpus = (0, 1, 2, 3, 4, 5, 6, 7)
source._current_affinity_cpus = lambda: cpus
source.detect_cpu_cores = lambda: len(cpus)
generic.detected_cpus = lambda: cpus
source._cpu_capacity_khz = lambda _cpu: 0
base_controls = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "GOTO_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "TORCH_NUM_THREADS", "TORCH_INTEROP_THREADS", "OMP_DYNAMIC",
    "MKL_DYNAMIC", "OMP_WAIT_POLICY", "TABULAR_CPU_THREADS",
    "TABULAR_CPU_WORKERS", "TABULAR_CPU_CORES",
    "TABULAR_CPU_JOB_CONCURRENCY", "TABULAR_CPU_AFFINITY_CPUS",
    "GOMP_CPU_AFFINITY", "OMP_PROC_BIND", "OMP_PLACES", "KMP_AFFINITY",
)
checked = 0
for hint, shared_arg, inherited_shared, affinity, slot, job_key in itertools.product(
    (None, 1, 2, 5, 10**9),
    (False, True),
    (None, "1", "0"),
    (None, "0-7", "0,2,4,6", "7-4", "bad"),
    (None, 0, 1, 4, 9),
    ("", "route-a", "route-b"),
):
    os.environ.pop("TABULAR_CPU_JOB_CONCURRENCY", None)
    os.environ.pop("TABULAR_CPU_AFFINITY_CPUS", None)
    if hint is None:
        os.environ["TABULAR_CPU_JOB_CONCURRENCY"] = "3"
    if affinity is not None:
        os.environ["TABULAR_CPU_AFFINITY_CPUS"] = affinity
    base = {"CUDA_VISIBLE_DEVICES": "0"}
    if inherited_shared is not None:
        base["TABULAR_CHILD_SHARED_CPU"] = inherited_shared
    if affinity is not None:
        base["TABULAR_CPU_AFFINITY_CPUS"] = affinity
    left = source.launcher_child_env(
        base, concurrency_hint=hint, job_key=job_key or None,
        affinity_slot=slot, shared_cpu=shared_arg,
    )
    right = generic.launcher_child_env(
        base, job_key=job_key, concurrency_hint=hint,
        affinity_slot=slot, shared_cpu=shared_arg,
    )
    comparable = set(base_controls) & (set(left) | set(right))
    assert {key: left.get(key) for key in comparable} == {
        key: right.get(key) for key in comparable
    }, (hint, shared_arg, inherited_shared, affinity, slot, job_key, left, right)
    checked += 1
assert checked == 2250
print(f"{checked} Linux child CPU environment cases matched")
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository)],
        cwd=repository, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout
    assert result.stdout.strip() == "2250 Linux child CPU environment cases matched"


def test_linux_runtime_bootstrap_values_environment_and_call_order_match_reference():
    """Compare Linux bootstrap settings and best-effort hook ordering."""
    repository = Path(__file__).parents[1]
    script = r'''
import dataclasses
import importlib
import os
import sys
from pathlib import Path

repository = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
sys.path.insert(0, str(repository / "src"))
source = importlib.import_module("utils.runtime_tuning")
from simple_adp.runner import runtime_tuning as generic
import torch

cpus = (0, 1, 2, 3, 4, 5, 6, 7)
events = []
source.detect_cpu_cores = lambda: len(cpus)
source._current_affinity_cpus = lambda: cpus
generic.detected_cpus = lambda: cpus
source._apply_affinity_from_env = lambda: events.append("affinity")
generic._apply_affinity_from_env = lambda: events.append("affinity")
source._maybe_reexec_under_systemd_scope = lambda label: events.append(("scope", label))
generic.maybe_reexec_systemd_scope = lambda *, label: events.append(("scope", label)) or False
source._apply_process_priority = lambda: events.append("priority")
generic._apply_priority = lambda: events.append("priority")
source._enforce_no_swap = lambda: events.append("no_swap")
generic._enforce_no_swap = lambda: events.append("no_swap")
torch.set_num_threads = lambda value: events.append(("torch_threads", value))
torch.set_num_interop_threads = lambda value: events.append(("torch_interop", value))

baseline = {
    "TABULAR_CHILD_SHARED_CPU": "1",
    "OMP_NUM_THREADS": "stale",
    "TABULAR_CPU_AFFINITY_CPUS": "1,3,5,7",
    "TABULAR_SKIP_SYSTEMD_SCOPE": "1",
}
def source_case(hint):
    os.environ.clear()
    os.environ.update(baseline)
    if hint is not None:
        os.environ["TABULAR_CPU_JOB_CONCURRENCY"] = hint
    events.clear()
    raw = source.bootstrap_runtime("fidelity")
    value = {
        "label": raw["label"], "cores": raw["cpu_cores"],
        "workers": raw["num_workers"], "torch_threads": raw["torch_threads"],
        "torch_interop_threads": raw["torch_interop_threads"],
    }
    return value, dict(os.environ), list(events)

def generic_case(hint):
    os.environ.clear()
    os.environ.update(baseline)
    if hint is not None:
        os.environ["TABULAR_CPU_JOB_CONCURRENCY"] = hint
    events.clear()
    value = dataclasses.asdict(generic.bootstrap_runtime("fidelity"))
    return value, dict(os.environ), list(events)

for hint in (None, "0", "1", "3", "invalid"):
    left = source_case(hint)
    right = generic_case(hint)
    assert left == right, (hint, left, right)
    assert left[2] == [
        ("scope", "fidelity"), "affinity", "priority", "no_swap",
        ("torch_threads", left[0]["torch_threads"]), ("torch_interop", 1),
    ]
print("5 Linux runtime bootstrap cases matched")
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository)],
        cwd=repository, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout
    assert result.stdout.strip() == "5 Linux runtime bootstrap cases matched"


def test_cpu_backend_seed_sequence_matches_linux_reference():
    """Compare CPU seed precedence and Python/NumPy/Torch RNG sequences."""
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import os
import random
import sys
from pathlib import Path

repository = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
sys.path.insert(0, str(repository / "src"))
source = importlib.import_module("utils.ml_backends")
from simple_adp.runner import backend as generic
import numpy as np
import torch

os.environ["TRAINING_CONTROL_BACKEND"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OPF_ADP_GLOBAL_SEED"] = "bad"
os.environ.pop("TABULAR_BACKEND", None)
os.environ.pop("TABULAR_CPU_ONLY", None)
os.environ.pop("SIMPLE_ADP_DISABLE_GPU_ACCELERATORS", None)
assert source.resolve_seed(None) == generic.resolve_seed(None) == 42
assert source.resolve_seed("bad") == generic.resolve_seed("bad") == 42
assert source.resolve_seed(19) == generic.resolve_seed(19) == 19

source_status = source.BackendStatus(False, False, False, False)
generic_status = generic.BackendStatus(False, False, False, False)
source.enable_optional_gpu_accelerators = lambda: source_status
generic.enable_optional_gpu_accelerators = lambda: generic_status
source._STATUS = generic._STATUS = None
source._STATUS_CONTEXT = generic._STATUS_CONTEXT = None
source._PROBED_GPU_CONTEXT = generic._PROBED_GPU_CONTEXT = None

def draws(seed_fn):
    assert seed_fn(31) == 31
    return (random.random(), np.random.random(4).tolist(), torch.rand(4).tolist())

source_draws = draws(source.seed_everything)
generic_draws = draws(generic.seed_everything)
assert source_draws == generic_draws, (source_draws, generic_draws)
print("CPU seed precedence and Python/NumPy/Torch sequences matched")
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository)],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "CPU seed precedence and Python/NumPy/Torch sequences matched"


def test_process_launch_order_and_linux_popen_contract_match_reference(tmp_path):
    """Compare actual source and generic launch functions with mocked Popen."""
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

repository = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
reference = importlib.import_module("utils.opf_massive_suite_runner")
from simple_adp.runner import processes
from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.datasets import DatasetTaskRef
from simple_adp.runner.models import ModelRoute
from simple_adp.runner.plan import RunIdentity, RunSpec
from simple_adp.runner.state import JobRuntime, JobStateStore

events = []
launches = []

class Child:
    pid = 4242
    def poll(self):
        return None

def popen(command, **kwargs):
    events.append("popen")
    launches.append((list(command), kwargs))
    return Child()

reference.subprocess.Popen = popen
reference.REPO_ROOT = root
reference.make_child_env = lambda *_args, **_kwargs: (events.append("env") or {"PATH": "/usr/bin"})
reference.write_compat_state_views = lambda *_args, **_kwargs: None
original_source_save = reference.save_job_state
def source_save(path, payload):
    events.append("state:" + payload["status"])
    original_source_save(path, payload)
reference.save_job_state = source_save

source_spec = reference.JobSpec(
    job_id="source-job", phase="probe", family="ADP", case_name="toy",
    repeat_index=0, depth=2, width=4, seed=7,
    results_dir=str(root / "source-results"), command=("python", "worker.py"),
)
source_dir = reference.job_directory(root / "source-run", source_spec)
source_runtime = reference.JobRuntime(
    source_spec, source_dir / "job_state.json", source_dir / "job.log",
)
reference.launch_job(
    SimpleNamespace(), source_runtime, 5, "gpu", root / "source-run",
    launch_context={"resumed_from_pause": True},
)
source_order = list(events)
source_call = launches[-1]
source_runtime.log_handle.close()

events.clear()
identity = RunIdentity(
    provider="toy", suite="suite", task_id="1", split_repeat=0,
    fold=0, sample=0, suite_repeat=1, model_route="route", seed=7,
    split_sha256="split", run_signature="signature",
)
generic_spec = RunSpec(
    identity, DatasetTaskRef("toy", "suite", "1"), "probe",
    ModelRoute("ADP-W-All"), "split.npz", str(root / "generic-results"), {},
    ("python", "worker.py"),
)
generic_runtime = JobRuntime(generic_spec)
store = JobStateStore(root / "generic-run")
original_generic_save = store.save
def generic_save(runtime, **kwargs):
    events.append("state:" + runtime.status.value)
    original_generic_save(runtime, **kwargs)
store.save = generic_save
processes.launcher_child_env = lambda *_args, **_kwargs: (events.append("env") or {"PATH": "/usr/bin"})
generic_order_start = len(events)
processes.launch_job(
    generic_runtime, settings=RunnerSettings(), state_store=store,
    repository_root=root, launch_budget=5, device="gpu",
    launch_context={"resumed_from_pause": True},
)
generic_order = events[generic_order_start:]
generic_call = launches[-1]
generic_runtime.log_handle.close()

assert source_order == generic_order == [
    "state:launching", "env", "popen", "state:running",
], (source_order, generic_order)
source_command, source_kwargs = source_call
generic_command, generic_kwargs = generic_call
assert source_command == generic_command == ["python", "worker.py"]
assert source_kwargs["cwd"] == generic_kwargs["cwd"] == str(root)
assert source_kwargs["env"] == {"PATH": "/usr/bin"}
assert generic_kwargs["env"]["PATH"] == "/usr/bin"
for kwargs in (source_kwargs, generic_kwargs):
    assert kwargs["text"] is True
    assert kwargs["stderr"] == reference.subprocess.STDOUT
    assert kwargs["start_new_session"] is True
    assert "stdin" not in kwargs
    assert kwargs["stdout"].mode == "a"
assert generic_kwargs["creationflags"] == 0
print("source and generic launch ordering plus POSIX Popen contract matched")
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository), str(tmp_path)],
        cwd=repository, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "source and generic launch ordering plus POSIX Popen contract matched"


def test_process_launch_failure_states_match_reference(tmp_path):
    """Compare failure persistence and log cleanup at both launch boundaries."""
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

repository = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
reference = importlib.import_module("utils.opf_massive_suite_runner")
from simple_adp.runner import processes
from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.datasets import DatasetTaskRef
from simple_adp.runner.models import ModelRoute
from simple_adp.runner.plan import RunIdentity, RunSpec
from simple_adp.runner.state import JobRuntime, JobStateStore

source_state_save = reference.save_job_state

def make_pair(name):
    source_spec = reference.JobSpec(
        job_id="source-" + name, phase="probe", family="ADP", case_name="toy",
        repeat_index=0, depth=2, width=4, seed=7,
        results_dir=str(root / ("source-results-" + name)), command=("python", "worker.py"),
    )
    source_dir = reference.job_directory(root / ("source-run-" + name), source_spec)
    source_runtime = reference.JobRuntime(
        source_spec, source_dir / "job_state.json", source_dir / "job.log",
    )
    identity = RunIdentity(
        provider="toy", suite="suite", task_id=name, split_repeat=0,
        fold=0, sample=0, suite_repeat=1, model_route="route", seed=7,
        split_sha256="split", run_signature="signature",
    )
    generic_spec = RunSpec(
        identity, DatasetTaskRef("toy", "suite", name), "probe",
        ModelRoute("ADP-W-All"), "split.npz", str(root / ("generic-results-" + name)), {},
        ("python", "worker.py"),
    )
    generic_runtime = JobRuntime(generic_spec)
    store = JobStateStore(root / ("generic-run-" + name))
    return source_runtime, generic_runtime, store

for boundary in ("environment", "popen"):
    events = []
    source_runtime, generic_runtime, store = make_pair(boundary)
    def source_save(path, payload):
        events.append("state:" + payload["status"])
        source_state_save(path, payload)
    reference.save_job_state = source_save
    reference.write_compat_state_views = lambda *_args, **_kwargs: None
    def build_environment(*_args, **_kwargs):
        events.append("env")
        if boundary == "environment":
            raise RuntimeError(boundary + " failure")
        return {"PATH": "/usr/bin"}
    reference.make_child_env = build_environment
    processes.launcher_child_env = build_environment
    def fail_popen(*_args, **_kwargs):
        events.append("popen")
        raise OSError("spawn failure")
    reference.subprocess.Popen = fail_popen
    original_store_save = store.save
    def generic_save(runtime, **kwargs):
        events.append("state:" + runtime.status.value)
        original_store_save(runtime, **kwargs)
    store.save = generic_save

    for label, launch in (
        ("source", lambda: reference.launch_job(
            SimpleNamespace(), source_runtime, 5, "gpu", root / ("source-run-" + boundary),
        )),
        ("generic", lambda: processes.launch_job(
            generic_runtime, settings=RunnerSettings(), state_store=store,
            repository_root=root, launch_budget=5, device="gpu",
        )),
    ):
        before = len(events)
        try:
            launch()
        except (RuntimeError, OSError):
            pass
        else:
            raise AssertionError(label + " launch unexpectedly succeeded")
        observed = events[before:]
        expected = ["state:launching", "env"] if boundary == "environment" else [
            "state:launching", "env", "popen",
        ]
        assert observed == expected, (boundary, label, observed)

    source_state = reference.load_job_state(source_runtime.state_path)
    generic_state = store.load(generic_runtime.job_id)
    assert source_state["status"] == generic_state["status"] == "launching"
    assert source_runtime.attempt == generic_runtime.attempt == 1
    assert source_runtime.proc is None and generic_runtime.process is None
    if boundary == "environment":
        assert not source_runtime.log_path.exists()
        assert not (store.job_dir(generic_runtime) / "job.log").exists()
    else:
        assert source_runtime.log_handle is None and generic_runtime.log_handle is None
        assert "=== LAUNCH " in source_runtime.log_path.read_text()
        assert "=== LAUNCH " in (store.job_dir(generic_runtime) / "job.log").read_text()
print("environment-build and Popen failure state/log behavior matched")
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository), str(tmp_path)],
        cwd=repository, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "environment-build and Popen failure state/log behavior matched"


def test_source_main_and_generic_coordinator_retry_trajectory_match(tmp_path):
    """Run one synthetic CPU child with a retry through both coordinator loops."""
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

repository = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
reference = importlib.import_module("utils.opf_massive_suite_runner")
from simple_adp.runner import processes
from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.datasets import DatasetTaskRef
from simple_adp.runner.models import ModelRoute
from simple_adp.runner.plan import RunIdentity, RunSpec, SuitePlan
from simple_adp.runner.resources import (
    CallbackResourceProvider, HostMemorySample, ProcessMemorySample,
)
from simple_adp.runner.scheduler import SuiteCoordinator
from simple_adp.runner.state import JobRuntime, RunStatus
from simple_adp.runner.state import JobStateStore

class Child:
    def __init__(self, pid, return_code):
        self.pid = pid
        self.return_code = return_code
    def poll(self):
        return self.return_code

popen_calls = []
def fake_popen(command, **kwargs):
    return_code = (7, 0, 7, 0)[len(popen_calls)]
    child = Child(5000 + len(popen_calls), return_code)
    popen_calls.append((list(command), kwargs))
    if return_code != 0:
        kwargs["stdout"].write("ordinary nonzero exit\n")
        kwargs["stdout"].flush()
    return child

# The source entrypoint receives a single deterministic spec; resource and
# environment functions are replaced at the OS boundary, while its actual
# launch, state transitions, polling loop, retry logic, and finalization run.
source_root = root / "source"
source_results = source_root / "results"
source_spec = reference.JobSpec(
    job_id="synthetic-one", phase="probe", family="ADP", case_name="toy",
    repeat_index=0, depth=1, width=2, seed=7,
    results_dir=str(source_results / "one"), command=("python", "worker.py"),
    device_capable=True,
)
sys.argv = [
    "opf_massive_suite_runner", "--mode", "massive",
    "--results-dir", str(source_results), "--run-root", str(source_root / "run"),
    "--cpu-only", "--concurrency", "1", "--repeat-count", "1", "--max-retries", "1",
    "--pressure-poll-interval-sec", "0.1", "--post-launch-sample-delay-sec", "0",
]
source_args = reference.parse_args()
reference.parse_args = lambda: source_args
reference.validate_args = lambda _args: None
reference.configure_root_logging = lambda: None
reference.set_logging_context_provider = lambda _provider: None
reference.bootstrap_runtime = lambda *_args: None
reference.enable_optional_gpu_accelerators = lambda: None
reference.seed_run_root_from_source = lambda *_args: {"files_copied": 0, "files_skipped": 0}
reference.build_suite_jobs = lambda *_args: [source_spec]
reference.sample_host_memory = lambda: reference.MemorySample(8192, 8192, 0.0, 0, 0, 0.0)
reference.sample_gpu_memory = lambda *_args: None
reference.current_pids = lambda _runtimes: []
reference.sample_non_python_host_memory = lambda **_kwargs: (0.0, [])
reference.sample_non_python_gpu_memory = lambda *_args, **_kwargs: (0.0, [], [])
reference.sample_runtime_child_memory = lambda *_args, **_kwargs: (0.0, 0.0, {})
reference.make_child_env = lambda *_args, **_kwargs: {"PATH": "/usr/bin"}
reference.subprocess.Popen = fake_popen
reference.time.sleep = lambda _seconds: None
reference.REPO_ROOT = repository
reference.main()
source_state = reference.load_job_state(
    reference.job_directory(source_root / "run", source_spec) / "job_state.json",
)
source_report = json.loads((source_root / "run" / "final_report.json").read_text())
source_launch_count = len(popen_calls)

# Run the generic coordinator with the same one-route, CPU-only workload and
# the same fake child; its real run loop and state/report writers remain active.
generic_root = root / "generic"
generic_results = generic_root / "results"
identity = RunIdentity(
    provider="toy", suite="suite", task_id="one", split_repeat=0,
    fold=0, sample=0, suite_repeat=1, model_route="ADP-W-All", seed=7,
    split_sha256="split", run_signature="generic-signature",
)
generic_spec = RunSpec(
    identity, DatasetTaskRef("toy", "suite", "one"), "probe",
    ModelRoute("ADP-W-All"), "split.npz", str(generic_results / "one"), {},
    ("python", "worker.py"),
)
generic_plan = SuitePlan(
    "generic-signature", {"run_signature": "generic-signature", "jobs": [generic_spec.as_dict()]},
    (generic_spec,),
)
provider = CallbackResourceProvider(
    host_sampler=lambda: HostMemorySample(8192, 8192, 0.0, 0, 0, 0.0),
    gpu_sampler=lambda _index: None,
    tree_sampler=lambda pid: frozenset({pid}),
    child_sampler=lambda _pid, **_kwargs: ProcessMemorySample(),
    external_host_sampler=lambda _excluded: ProcessMemorySample(),
    external_gpu_sampler=lambda _index, _excluded: ProcessMemorySample(),
)
processes.launcher_child_env = lambda *_args, **_kwargs: {"PATH": "/usr/bin"}
store = JobStateStore(generic_root / "run")
coordinator = SuiteCoordinator(
    generic_plan,
    RunnerSettings(
        cpu_only=True, concurrency=1, max_retries=1,
            pressure_poll_seconds=0.1, post_launch_sample_delay_seconds=0.0,
        run_hardness_screen=False,
    ),
    run_root=generic_root / "run", results_dir=generic_results,
    repository_root=repository, resource_provider=provider,
)
store = coordinator.state_store
coordinator.sleep = lambda _seconds: None
generic_report = coordinator.run()
generic_state = store.load(generic_spec.identity.run_id)
generic_launch_count = len(popen_calls) - source_launch_count

assert source_state["status"] == generic_state["status"] == "completed"
assert source_state["attempt"] == generic_state["attempt"] == 2
assert source_state["retry_count"] == generic_state["retry_count"] == 1
assert source_state["last_exit_code"] == generic_state["last_exit_code"] == 0
assert source_report["completed_jobs"] == generic_report["completed"] == 1
assert source_report["failed_attempts"] == generic_report["failed_attempts"] == 1
assert source_report["retried_jobs"] == generic_report["retries"] == 1
assert source_launch_count == generic_launch_count == 2
assert all(call[0] == ["python", "worker.py"] for call in popen_calls)
source_log = (reference.job_directory(source_root / "run", source_spec) / "job.log").read_text()
generic_log = (generic_root / "run" / "jobs" / generic_spec.identity.run_id / "job.log").read_text()
source_retry = "=== RETRY synthetic-one exit=7 ===\n"
generic_retry = f"=== RETRY {generic_spec.identity.run_id} exit=7 ===\n"
source_second_launch = "\n=== LAUNCH synthetic-one attempt=2 device=cpu ===\n"
generic_second_launch = f"\n=== LAUNCH {generic_spec.identity.run_id} attempt=2 device=cpu ===\n"
assert source_retry in source_log and generic_retry in generic_log
assert source_log.index(source_retry) < source_log.index(source_second_launch)
assert generic_log.index(generic_retry) < generic_log.index(generic_second_launch)
assert source_log.count("=== RETRY ") == generic_log.count("=== RETRY ") == 1
print("source main and generic coordinator matched one synthetic CPU route with retry")
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository), str(tmp_path)],
        cwd=repository, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout


def test_source_and_generic_match_oom_and_hard_pressure_same_pass(tmp_path):
    """Compare OOM peer pausing followed by the separate hard-pressure victim."""
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import json
import sys
from pathlib import Path

repository = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
reference = importlib.import_module("utils.opf_massive_suite_runner")
from simple_adp.runner import processes
from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.datasets import DatasetTaskRef
from simple_adp.runner.models import ModelRoute
from simple_adp.runner.plan import RunIdentity, RunSpec, SuitePlan
from simple_adp.runner.resources import (
    CallbackResourceProvider, GPUMemorySample, HostMemorySample, ProcessMemorySample,
)
from simple_adp.runner.scheduler import SuiteCoordinator

names = ("failed-oom", "oom-peer", "pressure-peer", "low-peer")
depths = {"failed-oom": 1, "oom-peer": 9, "pressure-peer": 5, "low-peer": 3}
footprints = {
    "failed-oom": (100.0, 100.0), "oom-peer": (200.0, 200.0),
    "pressure-peer": (900.0, 900.0), "low-peer": (100.0, 100.0),
}
current = {"phase": "source", "pass": 0}
launches = []
memory_by_pid = {}

class Child:
    def __init__(self, pid, name, attempt):
        self.pid, self.name, self.attempt = pid, name, attempt
    def poll(self):
        if self.name == "failed-oom" and self.attempt == 1 and current["pass"] >= 5:
            return 9
        if current["pass"] >= 7:
            return 0
        return None

def fake_popen(command, **kwargs):
    name = command[-1]
    attempt = sum(
        1 for item in launches if item[0] == current["phase"] and item[1] == name
    ) + 1
    pid = 8000 + len(launches)
    launches.append((current["phase"], name, attempt, pid))
    memory_by_pid[pid] = footprints[name]
    if name == "failed-oom" and attempt == 1:
        kwargs["stdout"].write("CUDA out of memory\n")
        kwargs["stdout"].flush()
    return Child(pid, name, attempt)

source_root = root / "source"
source_results = source_root / "results"
source_specs = [
    reference.JobSpec(
        job_id=name, phase="probe", family="ADP", case_name="toy",
        repeat_index=0, depth=depths[name], width=2, seed=11 + index,
        results_dir=str(source_results / name),
        command=(sys.executable, "worker.py", name), device_capable=True,
    )
    for index, name in enumerate(names)
]
sys.argv = [
    "opf_massive_suite_runner", "--mode", "massive",
    "--results-dir", str(source_results), "--run-root", str(source_root / "run"),
    "--concurrency", "5", "--repeat-count", "1", "--max-retries", "1",
    "--pressure-poll-interval-sec", "0.1", "--post-launch-sample-delay-sec", "0",
]
source_args = reference.parse_args()
reference.parse_args = lambda: source_args
reference.validate_args = lambda _args: None
reference.configure_root_logging = lambda: None
reference.set_logging_context_provider = lambda _provider: None
reference.bootstrap_runtime = lambda *_args: None
reference.enable_optional_gpu_accelerators = lambda: None
reference.seed_run_root_from_source = lambda *_args: {"files_copied": 0, "files_skipped": 0}
reference.build_suite_jobs = lambda *_args: source_specs

def source_host():
    current["pass"] += 1
    available = 500.0 if current["pass"] == 5 else 8192.0
    used_pct = 100.0 - 100.0 * available / 8192.0
    return reference.MemorySample(8192.0, available, used_pct, 0, 0, 0.0)

def source_gpu(_index=0):
    if current["pass"] == 5:
        return reference.GpuMemorySample(1000.0, 950.0, 50.0, 95.0)
    return reference.GpuMemorySample(1000.0, 100.0, 900.0, 10.0)

reference.sample_host_memory = source_host
reference.sample_gpu_memory = source_gpu
reference.current_pids = lambda _runtimes: []
reference.sample_non_python_host_memory = lambda **_kwargs: (0.0, [])
reference.sample_non_python_gpu_memory = lambda *_args, **_kwargs: (0.0, [], [])
reference.sample_runtime_child_memory = lambda runtime, *_args, **_kwargs: (*footprints[runtime.spec.job_id], {})
reference.make_child_env = lambda *_args, **_kwargs: {"PATH": "/usr/bin"}
reference.subprocess.Popen = fake_popen
pause_order = []
reference._terminate_process_tree = lambda process, **_kwargs: pause_order.append((current["phase"], process.name))
reference.time.sleep = lambda _seconds: None
reference.REPO_ROOT = repository
reference.main()
source_states = [
    reference.load_job_state(reference.job_directory(source_root / "run", spec) / "job_state.json")
    for spec in source_specs
]
source_report = json.loads((source_root / "run" / "final_report.json").read_text())
source_launches = len(launches)
source_backoff = json.loads((source_root / "run" / "batch_backoff_state.json").read_text())

current.update(phase="generic", **{"pass": 0})
generic_root = root / "generic"
generic_results = generic_root / "results"
generic_specs = []
for index, name in enumerate(names):
    identity = RunIdentity(
        provider="toy", suite="suite", task_id="1", split_repeat=0,
        fold=index, sample=index, suite_repeat=1, model_route=name, seed=11 + index,
        split_sha256="split", run_signature="generic-signature",
    )
    generic_specs.append(RunSpec(
        identity, DatasetTaskRef("toy", "suite", "1"), "probe",
        ModelRoute("ADP-W-All", init_depth=depths[name], init_width=2),
        "split.npz", str(generic_results / name), {},
        (sys.executable, "worker.py", name),
    ))
plan = SuitePlan(
    "generic-signature", {"run_signature": "generic-signature", "jobs": [s.as_dict() for s in generic_specs]},
    tuple(generic_specs),
)
def generic_host():
    current["pass"] += 1
    available = 500.0 if current["pass"] == 5 else 8192.0
    return HostMemorySample(8192.0, available, 100.0 - 100.0 * available / 8192.0, 0, 0, 0.0)
def generic_gpu(_index):
    if current["pass"] == 5:
        return GPUMemorySample(1000.0, 950.0, 50.0, 95.0)
    return GPUMemorySample(1000.0, 100.0, 900.0, 10.0)
provider = CallbackResourceProvider(
    host_sampler=generic_host, gpu_sampler=generic_gpu,
    tree_sampler=lambda pid: frozenset({pid}),
    child_sampler=lambda pid, **_kwargs: ProcessMemorySample(
        host_rss_mib=memory_by_pid.get(pid, (0.0, 0.0))[0],
        gpu_vram_mib=memory_by_pid.get(pid, (0.0, 0.0))[1],
    ),
    external_host_sampler=lambda _excluded: ProcessMemorySample(),
    external_gpu_sampler=lambda _index, _excluded: ProcessMemorySample(),
)
processes.launcher_child_env = lambda *_args, **_kwargs: {"PATH": "/usr/bin"}
import simple_adp.runner.scheduler as scheduler_module
def generic_terminate(process, **_kwargs):
    pause_order.append((current["phase"], process.name))
scheduler_module.terminate_process_tree = generic_terminate
coordinator = SuiteCoordinator(
    plan,
    RunnerSettings(
        concurrency=5, max_retries=1, batch_backoff_factor=0.5,
        pressure_poll_seconds=0.1, post_launch_sample_delay_seconds=0,
        run_hardness_screen=False,
    ),
    run_root=generic_root / "run", results_dir=generic_results,
    repository_root=repository, resource_provider=provider,
)
coordinator.sleep = lambda _seconds: None
generic_report = coordinator.run()
generic_states = [coordinator.state_store.load(spec.identity.run_id) for spec in generic_specs]
generic_launches = len(launches) - source_launches
generic_backoff = json.loads((generic_root / "run" / "batch_backoff_state.json").read_text())

assert current["pass"] >= 7
assert source_launches == generic_launches == 7
assert pause_order == [
    ("source", "oom-peer"), ("source", "pressure-peer"),
    ("generic", "oom-peer"), ("generic", "pressure-peer"),
], pause_order
assert [item["status"] for item in source_states] == [
    item["status"] for item in generic_states
] == ["completed"] * 4
assert [item["device_mode"] for item in source_states] == [
    item["device_mode"] for item in generic_states
] == ["gpu", "gpu", "gpu", "cpu"]
assert source_states[0]["retry_count"] == generic_states[0]["retry_count"] == 1
assert source_states[0]["attempt"] == generic_states[0]["attempt"] == 2
assert source_states[1]["attempt"] == generic_states[1]["attempt"] == 2
assert source_states[2]["attempt"] == generic_states[2]["attempt"] == 2
assert source_states[1]["gpu_pause_count"] == generic_states[1]["gpu_pause_count"] == 1
assert source_states[2]["gpu_pause_count"] == generic_states[2]["gpu_pause_count"] == 1
assert source_states[3]["pause_count"] == generic_states[3]["pause_count"] == 0
assert source_report["completed_jobs"] == generic_report["completed"] == 4
assert source_report["failed_attempts"] == generic_report["failed_attempts"] == 1
assert source_backoff["batch_scale"] == generic_backoff["batch_scale"] == 1.0
assert source_backoff["last_backoff_reason"] == generic_backoff["last_backoff_reason"] is None
print("source and generic matched OOM retry, two ordered pauses, same-lane resumes, and completion")
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository), str(tmp_path)],
        cwd=repository, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout
    assert result.stdout.strip().endswith(
        "source and generic matched OOM retry, two ordered pauses, same-lane resumes, and completion"
    )


def test_source_and_generic_job_grid_order_and_repeats_match(tmp_path):
    """Compare the source route matrix and ordering with the generic planner."""
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

repository = Path(sys.argv[1]).resolve()
temporary = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repository))
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
sys.argv = ["grid-differential", "--mode", "massive"]
reference = importlib.import_module("utils.opf_massive_suite_runner")
from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.datasets import DatasetTaskRef
from simple_adp.runner.plan import SplitSpec, SuitePlan, _sha_indices
from simple_adp.runner.route import ModelRoute
from simple_adp.baselines import vanilla_mlp_grid

family_to_model = {
    "loss_guided_neural_densification_width_growth_stop_on_first_non_improvement": "LGND-W-Greedy",
    "patience_based_width_growth_round_robin_layer_wise": "ADP-W-RR",
    "patience_based_width_growth_all_layers_together": "ADP-W-All",
    "width_first_round_robin_widen_then_add_depth_seed_new_depth_at_width_1": "ADP-WD-RR-D1",
    "width_first_all_layers_widen_then_add_depth_seed_new_depth_at_current_width": "ADP-WD-All-Dcur",
    "width_first_round_robin_widen_then_add_depth_seed_new_depth_at_current_width": "ADP-WD-RR-Dcur",
    "width_first_all_layers_widen_then_add_depth_seed_new_depth_at_width_1": "ADP-WD-All-D1",
    "depth_first_round_robin_widen_then_add_width_seed_new_width_at_width_1": "ADP-DW-D1-RR",
    "depth_first_all_layers_widen_then_add_width_seed_new_width_at_current_width": "ADP-DW-Dcur-All",
    "depth_first_round_robin_widen_then_add_width_seed_new_width_at_current_width": "ADP-DW-Dcur-RR",
    "depth_first_all_layers_widen_then_add_width_seed_new_width_at_width_1": "ADP-DW-D1-All",
    "stagewise_depth_growth_start_new_stage_at_width_1": "ADP-D-D1",
    "stagewise_depth_growth_start_new_stage_at_current_width": "ADP-D-Dcur",
    "vanilla": "MLP-Fixed",
}
model_to_group = {
    model: (
        "width_only" if model.startswith(("LGND", "ADP-W-")) else
        "width_to_depth" if model.startswith("ADP-WD-") else
        "depth_to_width" if model.startswith("ADP-DW-") else
        "depth_only"
    )
    for model in family_to_model.values()
}

class GridAdapter:
    name = "fixture-grid"
    def __init__(self, routes):
        self.routes = routes
    def enumerate_routes(self, _profiles, _settings):
        return list(self.routes)
    def fit_predict(self, *args, **kwargs):
        raise AssertionError("grid differential must not execute a model")

class FixtureProvider:
    name = "fixture"
    def list_tasks(self, suite):
        return [DatasetTaskRef("fixture", suite, "task-1")]
    def load_task(self, ref):
        return SimpleNamespace(
            dataset_id="fixture-data", name="fixture", target_name="label",
            content_sha256="fixture-content", X=np.asarray([[0.0], [1.0], [2.0], [3.0]]),
        )
    def list_splits(self, ref, data, settings):
        train, valid, test = (0, 1), (2,), (3,)
        return [SplitSpec(0, 0, 0, train, valid, test, _sha_indices(train, valid, test))]

def routes_for(width_depths, width_only_width, combined_depth, combined_width, depth_widths, include_vanilla=False):
    routes = []
    for model in ("LGND-W-Greedy", "ADP-W-RR", "ADP-W-All"):
        for depth in width_depths:
            routes.append(ModelRoute(model, label=f"{model}-{depth}", suite_group="width_only", init_width=width_only_width, init_depth=depth))
    for model in (
        "ADP-WD-RR-D1", "ADP-WD-All-Dcur", "ADP-WD-RR-Dcur", "ADP-WD-All-D1",
        "ADP-DW-D1-RR", "ADP-DW-Dcur-All", "ADP-DW-Dcur-RR", "ADP-DW-D1-All",
    ):
        group = model_to_group[model]
        routes.append(ModelRoute(model, label=model, suite_group=group, init_width=combined_width, init_depth=combined_depth))
    for model in ("ADP-D-D1", "ADP-D-Dcur"):
        for width in depth_widths:
            routes.append(ModelRoute(model, label=f"{model}-{width}", suite_group="depth_only", init_width=width, init_depth=1))
    if include_vanilla:
        grid = tuple(tuple(widths) for widths in vanilla_mlp_grid())
        routes.append(ModelRoute("MLP-Fixed", label="MLP-Fixed-aggregate", fixed_grid=grid, suite_group="vanilla"))
        routes.extend(
            ModelRoute(
                "MLP-Fixed", label=f"vanilla-d{len(widths)}-w{widths[0]}",
                fixed_grid=(tuple(widths),), suite_group="vanilla",
            )
            for widths in vanilla_mlp_grid()
        )
    return routes

cases = [
    # Five outer repeats with adaptive routes emitted only once.
    dict(repeats=5, adp_repeats=1, tasks="all", surface="all", flags=(1, 1, 1, 1), start=0, limit=0, vanilla=False),
    # ADP uses the missing/present repeat count, independently of outer count.
    dict(repeats=3, adp_repeats=2, tasks="width_to_depth,depth_only", surface="active", flags=(1, 1, 1, 1), start=2, limit=7, vanilla=False),
    # Repeat zero still creates one outer pass; unknown task names select no ADP route.
    dict(repeats=0, adp_repeats=1, tasks="unknown", surface="all", flags=(1, 1, 1, 1), start=0, limit=0, vanilla=False),
    # ADP repeats can exceed repeat_count; queue slicing applies after generation.
    dict(repeats=2, adp_repeats=4, tasks="width_only,depth_to_width", surface="all", flags=(1, 1, 1, 1), start=5, limit=13, vanilla=False),
    # Backlog clears task routes regardless of include flags.
    dict(repeats=3, adp_repeats=2, tasks="all", surface="backlog", flags=(1, 1, 1, 1), start=0, limit=0, vanilla=False),
    # Vanilla runs at every outer repeat while ADP runs only to its own limit.
    dict(repeats=2, adp_repeats=1, tasks="all", surface="all", flags=(1, 1, 1, 1), start=0, limit=0, vanilla=True),
    # Individual surface flags independently remove their categories.
    dict(repeats=1, adp_repeats=1, tasks="all", surface="all", flags=(1, 0, 1, 0), start=0, limit=0, vanilla=False),
]

for case_index, case in enumerate(cases):
    args = reference.parse_args()
    args.mode = "massive"
    args.cases = "fixture"
    args.surface = case["surface"]
    args.tasks = case["tasks"]
    args.repeat_count = case["repeats"]
    args.missing_present_task_repeats = case["adp_repeats"]
    args.prediction_repeats = 1
    args.include_width_only, args.include_width_to_depth, args.include_depth_to_width, args.include_depth_only = case["flags"]
    args.include_vanilla = case["vanilla"]
    args.width_only_depths = "1,3"
    args.width_only_width = 2
    args.width_to_depth_depth = 2
    args.width_to_depth_width = 4
    args.depth_only_widths = "1,4"
    args.vanilla_depths = "1,2,3,4,5"
    args.vanilla_widths = "1,2,4,8,16,32,64,128,256,512"
    args.job_start_index = case["start"]
    args.job_limit = case["limit"]
    args.head_variants = "single_head"
    reference.build_child_command = lambda *_args, **_kwargs: ("worker",)
    reference_jobs = reference.build_suite_jobs(args, temporary / f"source-run-{case_index}", temporary / f"source-results-{case_index}")
    expected = [
        (job.repeat_index, family_to_model[job.family], job.depth, job.width)
        for job in reference_jobs
    ]

    routes = routes_for((1, 3), 2, 2, 4, (1, 4), case["vanilla"])
    settings = RunnerSettings(
        repeat_count=case["repeats"], missing_present_task_repeats=case["adp_repeats"],
        prediction_repeats=1, model_seeds=(0,), generic_profiles=("faithful",),
        run_hardness_screen=False, tasks=tuple(item for item in case["tasks"].split(",") if item),
        surface=case["surface"], include_vanilla=case["vanilla"],
        include_width_only=bool(case["flags"][0]), include_width_to_depth=bool(case["flags"][1]),
        include_depth_to_width=bool(case["flags"][2]), include_depth_only=bool(case["flags"][3]),
        width_only_depths=(1, 3), width_only_width=2, width_to_depth_depth=2, width_to_depth_width=4,
        depth_only_widths=(1, 4), job_start_index=case["start"], job_limit=case["limit"],
    )
    generic_plan = SuitePlan.build(
        settings=settings, results_dir=temporary / f"generic-results-{case_index}",
        run_root=temporary / f"generic-run-{case_index}", dataset_providers=(FixtureProvider(),),
        dataset_suites=(("fixture", "fixture-suite"),), model_adapters=(GridAdapter(routes),),
    )
    aggregates = [spec for spec in generic_plan.runs if spec.route.label == "MLP-Fixed-aggregate"]
    if case["vanilla"]:
        expected_grid = tuple(tuple(widths) for widths in vanilla_mlp_grid())
        assert len(aggregates) == max(case["repeats"], 1)
        assert all(spec.route.fixed_grid == expected_grid for spec in aggregates)
    else:
        assert not aggregates
    actual = []
    for spec in generic_plan.runs:
        route = spec.route
        if route.model_id == "MLP-Fixed":
            # The generic suite includes one aggregate fixed-grid route in
            # addition to the 50 ledger-style individual vanilla routes.
            if route.label == "MLP-Fixed-aggregate":
                continue
            widths = route.fixed_grid[0]
            actual.append((spec.identity.suite_repeat, route.model_id, len(widths), widths[0]))
        else:
            actual.append((spec.identity.suite_repeat, route.model_id,
                           int(route.init_depth or spec.worker_settings["init_depth"]),
                           int(route.init_width or spec.worker_settings["init_width"])))
    assert actual == expected, (case_index, expected, actual)
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository), str(tmp_path)],
        cwd=repository, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout


def test_scheduler_label_cpu_hold_edge_matches_reference(tmp_path):
    """Pressure-aware requires both lane predicates; fixed/GPU-first do not."""
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import itertools
import os
import sys
from types import SimpleNamespace

repository = os.path.realpath(sys.argv[1])
sys.path.insert(0, repository + "/references/opf_adp_runner_suite")
for key in ("CPU_ONLY", "TABULAR_CPU_ONLY", "TRAINING_CONTROL_CPU_ONLY",
            "TRAINING_CONTROL_BACKEND", "TABULAR_BACKEND",
            "OPF_ADP_DISABLE_GPU_ACCELERATORS", "SIMPLE_ADP_DISABLE_GPU_ACCELERATORS"):
    os.environ.pop(key, None)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
reference = importlib.import_module("utils.opf_massive_suite_runner")
from simple_adp.runner.admission import AdmissionState, LaneCounts, choose_device
from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.resources import GPUMemorySample, HostMemorySample

source_host = reference.MemorySample(8192, 4096, 50.0, 0, 0, 0.0)
source_gpu = reference.GpuMemorySample(1000, 100, 900, 10.0)
generic_host = HostMemorySample(8192, 4096, 50.0, 0, 0, 0.0)
generic_gpu = GPUMemorySample(1000, 100, 900, 10.0)
for label in ("fixed", "gpu_first", "pressure_aware"):
    args = SimpleNamespace(
        scheduler=label, cpu_only=False, host_ram_resume_avail_mib=1280,
        gpu_resume_avail_mib=500, max_active_gpu_jobs=0, max_active_jobs=0,
    )
    source = reference.select_launch_device(
        SimpleNamespace(), args, source_host, source_gpu, 0, 0,
        host_admission_enabled=False, gpu_admission_enabled=True,
    )
    settings = RunnerSettings(scheduler=label)
    state = AdmissionState(cpu_hold_until=2.0)
    generic = choose_device(
        settings, state, host=generic_host, gpu=generic_gpu,
        counts=LaneCounts(0, 0, 0), now=1.0,
    )
    expected = None if label == "pressure_aware" else "gpu"
    assert source == generic == expected, (label, source, generic, expected)

    source_ready = reference.select_launch_device(
        SimpleNamespace(), args, source_host, source_gpu, 0, 0,
        host_admission_enabled=True, gpu_admission_enabled=True,
    )
    generic_ready = choose_device(
        settings, AdmissionState(), host=generic_host, gpu=generic_gpu,
        counts=LaneCounts(0, 0, 0), now=1.0,
    )
    assert source_ready == generic_ready == "gpu", (label, source_ready, generic_ready)

# Exhaust the discrete controller inputs that determine helper admission:
# scheduler label, CPU mask, resource readiness, active caps, gates, flags,
# and independent device holds. Invalid GPU-count > total-count cases are
# excluded because they cannot arise from a live process set.
checked = 0
for label in ("fixed", "gpu_first", "pressure_aware"):
    for cpu_only, parent_masked, host_ready in itertools.product((False, True), repeat=3):
        host_available = 1280 if host_ready else 1279
        for gpu_kind in ("missing", "below", "ready"):
            gpu_sample = None if gpu_kind == "missing" else reference.GpuMemorySample(
                1000, 1000 - (499 if gpu_kind == "below" else 500),
                499 if gpu_kind == "below" else 500,
                50.1 if gpu_kind == "below" else 50.0,
            )
            generic_gpu_sample = None if gpu_sample is None else GPUMemorySample(
                gpu_sample.total_mib, gpu_sample.used_mib, gpu_sample.free_mib,
                gpu_sample.used_pct,
            )
            for max_total, active_total, max_gpu, active_gpu in itertools.product(
                (0, 1), repeat=4,
            ):
                if active_gpu > active_total:
                    continue
                for cpu_enabled, gpu_enabled, host_gate, gpu_gate, cpu_hold_ready, gpu_hold_ready in itertools.product(
                    (False, True), repeat=6,
                ):
                    counts = LaneCounts(active_total, active_gpu, active_total - active_gpu)
                    state = AdmissionState(
                        gpu_gate_closed=gpu_gate, host_gate_closed=host_gate,
                        gpu_launches_enabled=gpu_enabled, cpu_launches_enabled=cpu_enabled,
                        cpu_hold_until=0.0 if cpu_hold_ready else 2.0,
                        gpu_hold_until=0.0 if gpu_hold_ready else 2.0,
                    )
                    settings = RunnerSettings(
                        scheduler=label, cpu_only=cpu_only,
                        max_active_jobs=max_total, max_active_gpu_jobs=max_gpu,
                    )
                    reference._parent_cpu_masked = lambda value=parent_masked: value
                    import simple_adp.runner.admission as admission
                    admission.parent_cpu_masked = lambda value=parent_masked: value
                    source_host_ready = (
                        cpu_enabled and not host_gate and host_ready and cpu_hold_ready
                    )
                    source_gpu_ready = (
                        not cpu_only and gpu_enabled and not gpu_gate and not host_gate
                        and host_ready and gpu_kind == "ready"
                        and (max_gpu <= 0 or active_gpu < max_gpu)
                        and gpu_hold_ready
                    )
                    args = SimpleNamespace(
                        scheduler=label, cpu_only=cpu_only,
                        host_ram_resume_avail_mib=1280, gpu_resume_avail_mib=500,
                        max_active_jobs=max_total, max_active_gpu_jobs=max_gpu,
                    )
                    source = reference.select_launch_device(
                        SimpleNamespace(), args,
                        reference.MemorySample(8192, host_available, 50, 0, 0, 0),
                        gpu_sample, active_gpu, active_total - active_gpu,
                        host_admission_enabled=source_host_ready,
                        gpu_admission_enabled=source_gpu_ready,
                    )
                    generic = choose_device(
                        settings, state,
                        host=HostMemorySample(8192, host_available, 50, 0, 0, 0),
                        gpu=generic_gpu_sample, counts=counts, now=1.0,
                    )
                    assert generic == source, (
                        label, cpu_only, parent_masked, host_available, gpu_kind,
                        max_total, active_total, max_gpu, active_gpu,
                        cpu_enabled, gpu_enabled, host_gate, gpu_gate,
                        cpu_hold_ready, gpu_hold_ready, source, generic,
                    )
                    checked += 1
assert checked == 55296, checked
print("scheduler-label CPU-hold behavior matched copied source")
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository)], cwd=repository,
        check=False, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout
    assert result.stdout.strip().endswith(
        "scheduler-label CPU-hold behavior matched copied source"
    )


def test_source_and_generic_standalone_wrapper_return_natural_child_exit(tmp_path):
    repository = Path(__file__).parents[1]
    script = r'''
import contextlib
import importlib
import io
import sys
from pathlib import Path
from types import SimpleNamespace

repository = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
source = importlib.import_module("utils.pressure_aware_wrapper")
from simple_adp.runner import pressure_wrapper as generic
from simple_adp.runner.config import StandalonePressureSettings

def execute(source_mode, pid_missing=False):
    launches = []
    class Child:
        pid = 7200
        returncode = 7
        def poll(self):
            return 7
        def wait(self, timeout=None):
            return 7
    def popen(command, **kwargs):
        launches.append(tuple(command))
        return Child()

    module = source if source_mode else generic
    psutil_module = module.psutil
    old = (
        module.subprocess.Popen,
        psutil_module.virtual_memory,
        psutil_module.swap_memory,
        psutil_module.Process,
        psutil_module.process_iter,
    )
    module.subprocess.Popen = popen
    psutil_module.virtual_memory = lambda: SimpleNamespace(percent=20.0, used=1000 * 1024 * 1024)
    psutil_module.swap_memory = lambda: SimpleNamespace(percent=0.0)
    def process_lookup(_pid):
        if pid_missing:
            raise psutil_module.NoSuchProcess(7200)
        return SimpleNamespace(pid=7200)
    psutil_module.Process = process_lookup
    psutil_module.process_iter = lambda _attrs=None: []
    output = io.StringIO()
    try:
        if source_mode:
            old_install = source._install_signal_handlers
            old_argv = sys.argv
            source._install_signal_handlers = lambda: None
            source._ACTIVE_CHILD_PROCESS = None
            sys.argv = ["pressure_aware_wrapper.py", "--", "probe"]
            try:
                with contextlib.redirect_stdout(output):
                    try:
                        source.main()
                    except SystemExit as exc:
                        result = int(exc.code)
            finally:
                source._install_signal_handlers = old_install
                source._ACTIVE_CHILD_PROCESS = None
                sys.argv = old_argv
        else:
            wrapper = generic.PressureAwareWrapper(
                ["probe"], StandalonePressureSettings(), sleeper=lambda _seconds: None,
            )
            with contextlib.redirect_stdout(output):
                result = wrapper.run()
    finally:
        (
            module.subprocess.Popen,
            psutil_module.virtual_memory,
            psutil_module.swap_memory,
            psutil_module.Process,
            psutil_module.process_iter,
        ) = old
    return result, launches

for missing_pid in (False, True):
    source_result = execute(True, missing_pid)
    generic_result = execute(False, missing_pid)
    assert source_result == generic_result == (7, [("probe",)]), (
        missing_pid, source_result, generic_result,
    )
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository)],
        cwd=repository, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout


def test_source_and_generic_standalone_wrapper_install_process_lifetime_signal_handlers(tmp_path):
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

repository = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
source = importlib.import_module("utils.pressure_aware_wrapper")
from simple_adp.runner import pressure_wrapper as generic
from simple_adp.runner.config import StandalonePressureSettings

registered = []
signal.signal = lambda signum, handler: registered.append((signum, handler))
signal.getsignal = lambda _signum: (_ for _ in ()).throw(AssertionError("source does not restore handlers"))

class Child:
    pid = 7201
    returncode = 0
    def poll(self):
        return 0
    def wait(self, timeout=None):
        return 0

module_subprocess = source.subprocess
module_psutil = source.psutil
old_popen = module_subprocess.Popen
old_process = module_psutil.Process
old_argv = sys.argv
try:
    module_subprocess.Popen = lambda *_args, **_kwargs: Child()
    module_psutil.Process = lambda _pid: SimpleNamespace(pid=7201)
    source._ACTIVE_CHILD_PROCESS = None
    sys.argv = ["pressure_aware_wrapper.py", "--", "probe"]
    try:
        source.main()
    except SystemExit as exc:
        assert exc.code == 0
    source_signals = [signum for signum, _handler in registered]

    generic_result = generic.PressureAwareWrapper(
        ["probe"], StandalonePressureSettings(), sleeper=lambda _seconds: None,
    ).run()
    generic_signals = [signum for signum, _handler in registered[len(source_signals):]]
finally:
    module_subprocess.Popen = old_popen
    module_psutil.Process = old_process
    source._ACTIVE_CHILD_PROCESS = None
    sys.argv = old_argv

expected = [signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT]
assert source_signals == expected
assert generic_signals == expected
assert generic_result == 0
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository)],
        cwd=repository, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout


def test_source_and_generic_standalone_wrapper_empty_command_error_match(tmp_path):
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import sys
from pathlib import Path

repository = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
source = importlib.import_module("utils.pressure_aware_wrapper")
from simple_adp.runner import pressure_wrapper as generic

sys.argv = ["pressure_aware_wrapper.py"]
try:
    source.main()
except SystemExit as exc:
    source_error = str(exc.code)
else:
    raise AssertionError("source accepted an empty child command")

try:
    generic.main([])
except SystemExit as exc:
    generic_error = str(exc.code)
else:
    raise AssertionError("generic wrapper accepted an empty child command")

assert source_error == generic_error == "No command provided", (source_error, generic_error)
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository)],
        cwd=repository, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout


def test_source_and_generic_standalone_wrapper_cli_validation_errors_match(tmp_path):
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import sys
from pathlib import Path

repository = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
source = importlib.import_module("utils.pressure_aware_wrapper")
from simple_adp.runner import pressure_wrapper as generic

cases = [
    (
        ["--host-ram-pressure-limit-pct", "85", "--host-ram-resume-pct", "90", "--", "probe"],
        ["--host-pressure-pct", "85", "--host-resume-pct", "90", "--", "probe"],
        "host RAM resume/pressure percentages must satisfy 0 <= resume <= pressure <= 100",
    ),
    (
        ["--gpu-memory-pressure-limit-pct", "85", "--gpu-memory-resume-pct", "90", "--", "probe"],
        ["--gpu-pressure-pct", "85", "--gpu-resume-pct", "90", "--", "probe"],
        "GPU resume/pressure percentages must satisfy 0 <= resume <= pressure <= 100",
    ),
    (
        ["--external-memory-drop-reopen-mib", "0", "--", "probe"],
        ["--external-memory-drop-mib", "0", "--", "probe"],
        "external-drop option validation",
    ),
    (
        ["--pressure-poll-interval-sec", "0", "--", "probe"],
        ["--poll-seconds", "0", "--", "probe"],
        "poll option validation",
    ),
]

for source_args, generic_args, expected in cases:
    sys.argv = ["pressure_aware_wrapper.py", *source_args]
    try:
        source.main()
    except SystemExit as exc:
        source_error = str(exc.code)
    else:
        raise AssertionError("source accepted invalid settings")
    try:
        generic.main(generic_args)
    except SystemExit as exc:
        generic_error = str(exc.code)
    else:
        raise AssertionError("generic wrapper accepted invalid settings")
    if expected == "external-drop option validation":
        assert source_error == "--external-memory-drop-reopen-mib must be > 0"
        assert generic_error == "--external-memory-drop-mib must be > 0"
    elif expected == "poll option validation":
        assert source_error == "--pressure-poll-interval-sec must be > 0"
        assert generic_error == "--poll-seconds must be > 0"
    else:
        assert generic_error == source_error == expected, (source_error, generic_error)
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository)],
        cwd=repository, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout


def test_source_and_generic_standalone_wrapper_real_process_group_pause_resume(tmp_path):
    repository = Path(__file__).parents[1]
    marker = tmp_path / "child-finished.txt"
    script = r'''
import contextlib
import importlib
import io
import sys
from pathlib import Path
from types import SimpleNamespace

repository = Path(sys.argv[1]).resolve()
marker = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
source = importlib.import_module("utils.pressure_aware_wrapper")
from simple_adp.runner import pressure_wrapper as generic
from simple_adp.runner.config import StandalonePressureSettings

def execute(source_mode):
    module = source if source_mode else generic
    psutil_module = module.psutil
    samples = 0
    events = []
    child_code = (
        "import time; time.sleep(0.2); "
        f"open({str(marker)!r}, 'w', encoding='utf-8').write('finished')"
    )
    old_virtual = psutil_module.virtual_memory
    old_swap = psutil_module.swap_memory
    old_process_iter = psutil_module.process_iter
    old_check_output = module.subprocess.check_output
    try:
        def virtual_memory():
            nonlocal samples
            samples += 1
            return SimpleNamespace(
                percent=91.0 if samples == 1 else 20.0,
                used=1000 * 1024 * 1024,
            )
        psutil_module.virtual_memory = virtual_memory
        psutil_module.swap_memory = lambda: SimpleNamespace(percent=0.0)
        psutil_module.process_iter = lambda _attrs=None: []
        module.subprocess.check_output = lambda *args, **kwargs: b""
        if source_mode:
            old_install = source._install_signal_handlers
            old_pause = source._pause_active_child_tree
            old_resume = source._resume_active_child_tree
            old_python_host = source.sample_python_host_memory_mib
            old_python_gpu = source.sample_python_gpu_memory_mib
            old_gpu_sample = source.sample_gpu_memory_pressure
            old_argv = sys.argv
            source._install_signal_handlers = lambda: None
            source._ACTIVE_CHILD_PROCESS = None
            source.sample_python_host_memory_mib = lambda: 0.0
            source.sample_python_gpu_memory_mib = lambda _index=0: 0.0
            source.sample_gpu_memory_pressure = lambda _index=0: (0.0, 0)
            def pause_source():
                events.append("pause")
                old_pause()
            def resume_source():
                events.append("resume")
                old_resume()
            source._pause_active_child_tree = pause_source
            source._resume_active_child_tree = resume_source
            sys.argv = [
                "pressure_aware_wrapper.py", "--",
                sys.executable, "-c", child_code,
            ]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    try:
                        source.main()
                    except SystemExit as exc:
                        result = int(exc.code)
            finally:
                source._install_signal_handlers = old_install
                source._pause_active_child_tree = old_pause
                source._resume_active_child_tree = old_resume
                source.sample_python_host_memory_mib = old_python_host
                source.sample_python_gpu_memory_mib = old_python_gpu
                source.sample_gpu_memory_pressure = old_gpu_sample
                source._ACTIVE_CHILD_PROCESS = None
                sys.argv = old_argv
        else:
            old_pause = generic._pause_child
            old_usage = generic._sample_python_usage
            old_gpu_sample = generic._gpu_pressure_sample
            generic._sample_python_usage = lambda _index=0: (0.0, 0.0)
            generic._gpu_pressure_sample = lambda _index=0: (0.0, 0)
            def pause_generic(child, paused):
                events.append("pause" if paused else "resume")
                old_pause(child, paused)
            generic._pause_child = pause_generic
            try:
                wrapper = generic.PressureAwareWrapper(
                    [sys.executable, "-c", child_code],
                    StandalonePressureSettings(poll_seconds=0.1),
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    result = wrapper.run()
            finally:
                generic._pause_child = old_pause
                generic._sample_python_usage = old_usage
                generic._gpu_pressure_sample = old_gpu_sample
    finally:
        psutil_module.virtual_memory = old_virtual
        psutil_module.swap_memory = old_swap
        psutil_module.process_iter = old_process_iter
        module.subprocess.check_output = old_check_output
    return result, events

source_result = execute(True)
generic_result = execute(False)
assert source_result == generic_result == (0, ["pause", "resume"]), (
    source_result, generic_result,
)
assert marker.read_text(encoding="utf-8") == "finished"
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository), str(marker)],
        cwd=repository, check=False, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout


def test_source_and_generic_standalone_wrapper_interrupt_resumes_then_terminates(tmp_path):
    repository = Path(__file__).parents[1]
    script = r'''
import contextlib
import importlib
import io
import sys
from pathlib import Path
from types import SimpleNamespace

repository = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
source = importlib.import_module("utils.pressure_aware_wrapper")
from simple_adp.runner import pressure_wrapper as generic
from simple_adp.runner.config import StandalonePressureSettings

def execute(source_mode):
    events = []
    result = None
    class Child:
        pid = 7300
        returncode = None
        def poll(self):
            return None
        def wait(self, timeout=None):
            return None
    def popen(_command, **_kwargs):
        return Child()
    def interrupt(_seconds):
        raise KeyboardInterrupt

    module = source if source_mode else generic
    psutil_module = module.psutil
    old = (
        module.subprocess.Popen,
        module.time.sleep,
        psutil_module.virtual_memory,
        psutil_module.swap_memory,
        psutil_module.Process,
        psutil_module.process_iter,
        module.subprocess.check_output,
    )
    module.subprocess.Popen = popen
    module.time.sleep = interrupt
    psutil_module.virtual_memory = lambda: SimpleNamespace(percent=91.0, used=1000 * 1024 * 1024)
    psutil_module.swap_memory = lambda: SimpleNamespace(percent=70.0)
    psutil_module.Process = lambda _pid: SimpleNamespace(pid=7300)
    psutil_module.process_iter = lambda _attrs=None: []
    module.subprocess.check_output = lambda *args, **kwargs: b""
    output = io.StringIO()
    try:
        if source_mode:
            old_install = source._install_signal_handlers
            old_pause = source._pause_active_child_tree
            old_resume = source._resume_active_child_tree
            old_terminate = source._terminate_active_child_tree
            old_python_host = source.sample_python_host_memory_mib
            old_python_gpu = source.sample_python_gpu_memory_mib
            old_gpu_sample = source.sample_gpu_memory_pressure
            old_argv = sys.argv
            source._install_signal_handlers = lambda: None
            source._ACTIVE_CHILD_PROCESS = None
            source.sample_python_host_memory_mib = lambda: 0.0
            source.sample_python_gpu_memory_mib = lambda _index=0: 0.0
            source.sample_gpu_memory_pressure = lambda _index=0: (0.0, 0)
            source._pause_active_child_tree = lambda: events.append("pause")
            source._resume_active_child_tree = lambda: events.append("resume")
            source._terminate_active_child_tree = lambda: events.append("terminate")
            sys.argv = ["pressure_aware_wrapper.py", "--", "probe"]
            try:
                with contextlib.redirect_stdout(output):
                    try:
                        source.main()
                    except SystemExit as exc:
                        result = int(exc.code)
            finally:
                source._install_signal_handlers = old_install
                source._pause_active_child_tree = old_pause
                source._resume_active_child_tree = old_resume
                source._terminate_active_child_tree = old_terminate
                source.sample_python_host_memory_mib = old_python_host
                source.sample_python_gpu_memory_mib = old_python_gpu
                source.sample_gpu_memory_pressure = old_gpu_sample
                source._ACTIVE_CHILD_PROCESS = None
                sys.argv = old_argv
        else:
            old_pause = generic._pause_child
            old_terminate = generic.terminate_standalone_pressure_tree
            old_usage = generic._sample_python_usage
            old_gpu_sample = generic._gpu_pressure_sample
            generic._sample_python_usage = lambda _index=0: (0.0, 0.0)
            generic._gpu_pressure_sample = lambda _index=0: (0.0, 0)
            generic._pause_child = lambda _child, paused: events.append(
                "pause" if paused else "resume"
            )
            generic.terminate_standalone_pressure_tree = lambda _child: events.append("terminate")
            wrapper = generic.PressureAwareWrapper(
                ["probe"], StandalonePressureSettings(poll_seconds=0.1), sleeper=interrupt,
            )
            with contextlib.redirect_stdout(output):
                result = wrapper.run()
            generic._pause_child = old_pause
            generic.terminate_standalone_pressure_tree = old_terminate
            generic._sample_python_usage = old_usage
            generic._gpu_pressure_sample = old_gpu_sample
    finally:
        (
            module.subprocess.Popen,
            module.time.sleep,
            psutil_module.virtual_memory,
            psutil_module.swap_memory,
            psutil_module.Process,
            psutil_module.process_iter,
            module.subprocess.check_output,
        ) = old
    assert result is not None, (source_mode, events)
    return result, events

source_result = execute(True)
generic_result = execute(False)
assert source_result == generic_result == (130, ["pause", "resume", "terminate"]), (
    source_result, generic_result,
)
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository)],
        cwd=repository, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout


def test_source_and_generic_emergency_cleanup_markers_are_entrypoint_specific(tmp_path):
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import sys
from pathlib import Path

repository = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
source = importlib.import_module("scripts.kill_all_runners")
from simple_adp.runner.process_cleanup import owns_process

cases = [
    # Source entrypoint argv, generic equivalent argv, source marker result.
    (
        ["python", "opf_suite_launcher.py", "--mode", "massive"],
        ["python", "-m", "simple_adp.runner.launcher", "--output", "results"],
        True,
    ),
    (
        ["python", "opf_suite_launcher.py", "--model", "wrong-marker"],
        ["python", "-m", "simple_adp.runner.launcher", "--model", "wrong-marker"],
        False,
    ),
    (
        ["python", "opf_massive_suite_runner.py", "--mode", "massive"],
        ["python", "-m", "simple_adp.runner.controller", "--plan", "plan.json"],
        True,
    ),
    (
        ["python", "opf_massive_suite_runner.py", "--spec", "wrong-marker"],
        ["python", "-m", "simple_adp.runner.controller", "--spec", "wrong-marker"],
        False,
    ),
    (
        ["python", "universal_main.py", "--model", "ADP-W-All"],
        ["python", "-m", "simple_adp.runner.worker", "--spec", "job.json"],
        True,
    ),
    (
        ["python", "universal_main.py", "--mode", "wrong-marker"],
        ["python", "-m", "simple_adp.runner.worker", "--mode", "wrong-marker"],
        False,
    ),
]
for source_argv, generic_argv, expected in cases:
    source_match = source._is_owned_command(source_argv)
    generic_match = owns_process(generic_argv, str(root), root)
    assert source_match is generic_match is expected, (
        source_argv, generic_argv, expected, source_match, generic_match,
    )
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository), str(tmp_path)],
        cwd=repository, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout


def test_source_and_generic_result_bundle_export_restore_match(tmp_path):
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import json
import sys
import tarfile
from pathlib import Path

repository = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
source = importlib.import_module("utils.opf_resume_bundle")
from simple_adp.runner.bundles import export_results_bundle, restore_results_bundle

source_root = root / "source-repo"
generic_root = root / "generic-repo"
source_results = source_root / "results"
generic_results = generic_root / "results"
for result in (source_results, generic_results):
    (result / "nested").mkdir(parents=True)
    (result / "state.json").write_text('{"state":"saved"}', encoding="utf-8")
    (result / "telemetry.csv").write_text("kind\\nbatch\\n", encoding="utf-8")
    (result / "nested" / "telemetry.csv").write_text("kind\\nepoch\\n", encoding="utf-8")
    (result / "nested" / "kept.csv").write_text("keep\\n", encoding="utf-8")
    (result / "nested" / "telemetry.csv.manifest.json").write_text("{}", encoding="utf-8")

source_bundle = root / "source.tar.gz"
generic_bundle = root / "generic.tar.gz"
assert source.export_results_bundle(source_results, source_bundle) == source_bundle
assert export_results_bundle(generic_results, generic_root, generic_bundle) == generic_bundle

def stable_manifest(bundle):
    path = bundle.with_suffix("").with_suffix(".json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {key: payload[key] for key in ("kind", "bundle_mode", "part_size_bytes", "source_exists", "source_size_bytes")}

assert stable_manifest(source_bundle) == stable_manifest(generic_bundle)

def archived_files(bundle):
    with tarfile.open(bundle, "r:gz") as archive:
        return {
            member.name: archive.extractfile(member).read()
            for member in archive.getmembers()
            if member.isfile()
        }

expected = archived_files(source_bundle)
actual = archived_files(generic_bundle)
assert actual == expected
assert "results/telemetry.csv" not in actual
assert "results/nested/telemetry.csv" not in actual
assert "results/nested/kept.csv" in actual

for result in (source_results, generic_results):
    import shutil
    shutil.rmtree(result)
assert source.restore_results_bundle(source_results, source_bundle)
assert restore_results_bundle(generic_results, generic_root, generic_bundle)

def tree(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*") if path.is_file()
    }

assert tree(source_results) == tree(generic_results)
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository), str(tmp_path)],
        cwd=repository, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout


def test_source_main_and_generic_coordinator_external_gate_reopen_resume_match(tmp_path):
    """Compare two real loops through host pressure and external-PID gate reopen."""
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import json
import sys
from pathlib import Path

repository = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
reference = importlib.import_module("utils.opf_massive_suite_runner")
from simple_adp.runner import processes
from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.datasets import DatasetTaskRef
from simple_adp.runner.models import ModelRoute
from simple_adp.runner.plan import RunIdentity, RunSpec, SuitePlan
from simple_adp.runner.resources import (
    CallbackResourceProvider, HostMemorySample, ProcessMemorySample,
)
from simple_adp.runner.scheduler import SuiteCoordinator
from simple_adp.runner.state import JobStateStore

class CaptureLogger:
    def __init__(self):
        self.info_messages = []
    def info(self, message, *_args, **_kwargs):
        self.info_messages.append(str(message))
    def warning(self, *_args, **_kwargs):
        pass
    def exception(self, *_args, **_kwargs):
        pass

source_logger = CaptureLogger()
reference.logging.getLogger = lambda *_args, **_kwargs: source_logger

current = {"phase": "source", "pass": 0}
popen_calls = []
generic_rss_by_pid = {}
resume_state_at_launch = {}
gate_state_at_pressure_pause = {}
class Child:
    def __init__(self, pid, return_code):
        self.pid = pid
        self.return_code = return_code
    def poll(self):
        if self.return_code is not None:
            return self.return_code
        return 0 if current["pass"] >= 4 else None

def fake_popen(command, **kwargs):
    call_index = len(popen_calls)
    pid = 6000 + call_index
    return_code = 0 if call_index in (2, 5) else None
    if current["phase"] == "source" and call_index == 2:
        resume_state_at_launch["source"] = json.loads(
            (source_root / "run" / "batch_backoff_state.json").read_text()
        )
    if current["phase"] == "generic" and call_index == 5:
        resume_state_at_launch["generic"] = json.loads(
            (root / "generic" / "run" / "batch_backoff_state.json").read_text()
        )
    popen_calls.append((list(command), kwargs, pid))
    if current["phase"] == "generic":
        generic_rss_by_pid[pid] = 100.0 if call_index in (3, 5) else 200.0
    return Child(pid, return_code)

source_root = root / "source"
source_results = source_root / "results"
source_specs = [
    reference.JobSpec(
        job_id=f"synthetic-{index}", phase="probe", family="ADP", case_name="toy",
        repeat_index=0, depth=1, width=index + 1, seed=index + 7,
        results_dir=str(source_results / str(index)), command=("python", "worker.py"),
    )
    for index in range(2)
]
sys.argv = [
    "opf_massive_suite_runner", "--mode", "massive",
    "--results-dir", str(source_results), "--run-root", str(source_root / "run"),
    "--cpu-only", "--concurrency", "2", "--repeat-count", "1", "--max-retries", "1",
    "--pressure-poll-interval-sec", "0.1", "--post-launch-sample-delay-sec", "0",
]
source_args = reference.parse_args()
reference.parse_args = lambda: source_args
reference.validate_args = lambda _args: None
reference.configure_root_logging = lambda: None
reference.set_logging_context_provider = lambda _provider: None
reference.bootstrap_runtime = lambda *_args: None
reference.enable_optional_gpu_accelerators = lambda: None
reference.seed_run_root_from_source = lambda *_args: {"files_copied": 0, "files_skipped": 0}
reference.build_suite_jobs = lambda *_args: source_specs

def source_host_sample():
    current["pass"] += 1
    available = 500 if current["pass"] == 3 else 8192
    return reference.MemorySample(8192, available, 100.0 - 100.0 * available / 8192, 0, 0, 0.0)

reference.sample_host_memory = source_host_sample
reference.sample_gpu_memory = lambda *_args: None
reference.current_pids = lambda _runtimes: []
reference.sample_non_python_host_memory = lambda **_kwargs: (
    (1000.0, [77]) if current["pass"] < 3 else (400.0, [])
)
reference.sample_non_python_gpu_memory = lambda *_args, **_kwargs: (0.0, [], [])
reference.sample_runtime_child_memory = lambda runtime, *_args, **_kwargs: (
    (100.0 if runtime.spec.job_id == "synthetic-0" else 200.0), 0.0, {},
)
reference.make_child_env = lambda *_args, **_kwargs: {"PATH": "/usr/bin"}
reference.subprocess.Popen = fake_popen
def source_terminate(_proc, **_kwargs):
    gate_state_at_pressure_pause["source"] = (
        current["pass"],
        json.loads((source_root / "run" / "batch_backoff_state.json").read_text()),
    )
reference._terminate_process_tree = source_terminate
reference.time.sleep = lambda _seconds: None
reference.REPO_ROOT = repository
reference.main()
source_run_root = source_root / "run"
source_states = [
    reference.load_job_state(reference.job_directory(source_run_root, spec) / "job_state.json")
    for spec in source_specs
]
source_report = json.loads((source_run_root / "final_report.json").read_text())
source_launches = len(popen_calls)
source_passes = current["pass"]

current.update(phase="generic", **{"pass": 0})
generic_root = root / "generic"
generic_results = generic_root / "results"
generic_specs = []
for index in range(2):
    identity = RunIdentity(
        provider="toy", suite="suite", task_id=str(index), split_repeat=0,
        fold=0, sample=0, suite_repeat=1, model_route=f"route-{index}", seed=index + 7,
        split_sha256="split", run_signature="generic-signature",
    )
    generic_specs.append(RunSpec(
        identity, DatasetTaskRef("toy", "suite", str(index)), "probe",
        ModelRoute("ADP-W-All", init_depth=1, init_width=index + 1),
        "split.npz", str(generic_results / str(index)), {}, ("python", "worker.py"),
    ))
plan = SuitePlan(
    "generic-signature", {"run_signature": "generic-signature", "jobs": [s.as_dict() for s in generic_specs]},
    tuple(generic_specs),
)
def generic_host_sample():
    current["pass"] += 1
    available = 500 if current["pass"] == 3 else 8192
    return HostMemorySample(8192, available, 100.0 - 100.0 * available / 8192, 0, 0, 0.0)

provider = CallbackResourceProvider(
    host_sampler=generic_host_sample,
    gpu_sampler=lambda _index: None,
    tree_sampler=lambda pid: frozenset({pid}),
    child_sampler=lambda pid, **_kwargs: ProcessMemorySample(
        host_rss_mib=generic_rss_by_pid.get(pid, 0.0), gpu_vram_mib=0.0,
    ),
    external_host_sampler=lambda _excluded: ProcessMemorySample(
        host_rss_mib=1000.0 if current["pass"] < 3 else 400.0,
        pid_set=frozenset({77}) if current["pass"] < 3 else frozenset(),
    ),
    external_gpu_sampler=lambda _index, _excluded: ProcessMemorySample(),
)
processes.launcher_child_env = lambda *_args, **_kwargs: {"PATH": "/usr/bin"}
import simple_adp.runner.scheduler as scheduler_module
original_reopen_external_gates = scheduler_module.reopen_external_gates
def observe_reopen_external_gates(state, **kwargs):
    before = (state.host_gate_closed, state.cpu_launches_enabled)
    original_reopen_external_gates(state, **kwargs)
    if kwargs.get("host_drop"):
        gate_state_at_pressure_pause["generic_reopen_transition"] = (
            before, (state.host_gate_closed, state.cpu_launches_enabled),
        )
scheduler_module.reopen_external_gates = observe_reopen_external_gates
def generic_terminate(_proc, **_kwargs):
    gate_state_at_pressure_pause["generic"] = (
        current["pass"],
        json.loads((generic_root / "run" / "batch_backoff_state.json").read_text()),
    )
scheduler_module.terminate_process_tree = generic_terminate
coordinator = SuiteCoordinator(
    plan,
    RunnerSettings(
        cpu_only=True, concurrency=2, max_retries=1,
        pressure_poll_seconds=0.1, post_launch_sample_delay_seconds=0.0,
        run_hardness_screen=False,
    ),
    run_root=generic_root / "run", results_dir=generic_results,
    repository_root=repository, resource_provider=provider,
)
coordinator.sleep = lambda _seconds: None
generic_report = coordinator.run()
generic_states = [coordinator.state_store.load(spec.identity.run_id) for spec in generic_specs]
generic_launches = len(popen_calls) - source_launches

assert source_passes >= 4 and current["pass"] >= 4
assert source_report["completed_jobs"] == generic_report["completed"] == 2
assert source_report["failed_attempts"] == generic_report["failed_attempts"] == 0
assert source_launches == generic_launches == 3
assert [state["status"] for state in source_states] == [
    state["status"] for state in generic_states
] == ["completed", "completed"]
assert [state["attempt"] for state in source_states] == [
    state["attempt"] for state in generic_states
] == [1, 2]
assert [state["pause_count"] for state in source_states] == [
    state["pause_count"] for state in generic_states
] == [0, 1]
assert source_states[1]["last_exit_code"] == generic_states[1]["last_exit_code"] == 0
assert gate_state_at_pressure_pause["source"][0] == gate_state_at_pressure_pause["generic"][0] == 3
assert gate_state_at_pressure_pause["source"][1]["host_gate_closed"] is False
assert gate_state_at_pressure_pause["source"][1]["cpu_launches_enabled"] is True
assert gate_state_at_pressure_pause["generic"][1]["host_gate_closed"] is False
assert gate_state_at_pressure_pause["generic"][1]["cpu_launches_enabled"] is True
assert gate_state_at_pressure_pause["generic_reopen_transition"] == ((True, False), (False, True))
assert resume_state_at_launch["source"]["last_external_host_non_python_mib"] == 400.0
assert resume_state_at_launch["source"]["external_host_non_python_pids"] == []
assert resume_state_at_launch["generic"]["external_host_mib"] == 400.0
assert resume_state_at_launch["generic"]["external_host_pids"] == []
assert any("host admission reopened by external non-python memory drop" in item for item in source_logger.info_messages)
assert any("host admission closed" in item for item in source_logger.info_messages)
print("source main and generic coordinator matched external gate reopen and host-pressure resume")
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository), str(tmp_path)],
        cwd=repository, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout
    assert result.stdout.strip().endswith(
        "source main and generic coordinator matched external gate reopen and host-pressure resume"
    )


def test_source_main_and_generic_coordinator_gpu_oom_peer_resume_match(tmp_path):
    """Compare GPU OOM peer pause, retry backoff, and peer resume end to end."""
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import json
import os
import sys
from pathlib import Path

repository = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
for key in (
    "CPU_ONLY", "TABULAR_CPU_ONLY", "TRAINING_CONTROL_CPU_ONLY",
    "TRAINING_CONTROL_BACKEND", "TABULAR_BACKEND",
    "OPF_ADP_DISABLE_GPU_ACCELERATORS", "SIMPLE_ADP_DISABLE_GPU_ACCELERATORS",
):
    os.environ.pop(key, None)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
reference = importlib.import_module("utils.opf_massive_suite_runner")
from simple_adp.runner import processes
from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.datasets import DatasetTaskRef
from simple_adp.runner.models import ModelRoute
from simple_adp.runner.plan import RunIdentity, RunSpec, SuitePlan
from simple_adp.runner.resources import (
    CallbackResourceProvider, GPUMemorySample, HostMemorySample, ProcessMemorySample,
)
from simple_adp.runner.scheduler import SuiteCoordinator
from simple_adp.runner.state import JobStateStore

current = {"phase": "source", "pass": 0}
popen_calls = []
generic_memory_by_pid = {}
class Child:
    def __init__(self, pid, return_code):
        self.pid = pid
        self.return_code = return_code
    def poll(self):
        if self.return_code == 9 and current["pass"] < 3:
            return None
        if self.return_code is None:
            return 0 if current["pass"] >= 4 else None
        return self.return_code

def fake_popen(command, **kwargs):
    call_index = len(popen_calls)
    pid = 7000 + call_index
    return_code = 9 if call_index in (0, 4) else (None if call_index in (1, 5) else 0)
    popen_calls.append((list(command), kwargs, pid))
    if return_code == 9:
        kwargs["stdout"].write("CUDA out of memory\n")
        kwargs["stdout"].flush()
    if current["phase"] == "generic":
        generic_memory_by_pid[pid] = (100.0, 0.0)
    return Child(pid, return_code)

source_root = root / "source"
source_results = source_root / "results"
source_specs = [
    reference.JobSpec(
        job_id=f"oom-{index}", phase="probe", family="ADP", case_name="toy",
        repeat_index=0, depth=1, width=index + 1, seed=index + 11,
        results_dir=str(source_results / str(index)), command=("python", "worker.py"),
    )
    for index in range(2)
]
sys.argv = [
    "opf_massive_suite_runner", "--mode", "massive",
    "--results-dir", str(source_results), "--run-root", str(source_root / "run"),
    "--concurrency", "2", "--repeat-count", "1", "--max-retries", "1",
    "--gpu-resume-avail-mib", "200", "--pressure-poll-interval-sec", "0.1",
    "--post-launch-sample-delay-sec", "0",
]
source_args = reference.parse_args()
reference.parse_args = lambda: source_args
reference.validate_args = lambda _args: None
reference.configure_root_logging = lambda: None
reference.set_logging_context_provider = lambda _provider: None
reference.bootstrap_runtime = lambda *_args: None
reference.enable_optional_gpu_accelerators = lambda: None
reference.seed_run_root_from_source = lambda *_args: {"files_copied": 0, "files_skipped": 0}
reference.build_suite_jobs = lambda *_args: source_specs

def source_host_sample():
    current["pass"] += 1
    return reference.MemorySample(8192, 8192, 0.0, 0, 0, 0.0)

def source_gpu_sample(_index=0):
    if current["pass"] == 3:
        return reference.GpuMemorySample(1000, 800, 200, 80.0)
    if current["pass"] >= 4:
        return reference.GpuMemorySample(1000, 200, 800, 20.0)
    return reference.GpuMemorySample(1000, 100, 900, 10.0)

reference.sample_host_memory = source_host_sample
reference.sample_gpu_memory = source_gpu_sample
reference.current_pids = lambda _runtimes: []
reference.sample_non_python_host_memory = lambda **_kwargs: (0.0, [])
reference.sample_non_python_gpu_memory = lambda *_args, **_kwargs: (0.0, [], [])
reference.sample_runtime_child_memory = lambda *_args, **_kwargs: (100.0, 0.0, {})
reference.make_child_env = lambda *_args, **_kwargs: {"PATH": "/usr/bin"}
reference.subprocess.Popen = fake_popen
reference._terminate_process_tree = lambda *_args, **_kwargs: None
reference.time.sleep = lambda _seconds: None
reference.REPO_ROOT = repository
reference.main()
source_run_root = source_root / "run"
source_states = [
    reference.load_job_state(reference.job_directory(source_run_root, spec) / "job_state.json")
    for spec in source_specs
]
source_report = json.loads((source_run_root / "final_report.json").read_text())
source_launches = len(popen_calls)
source_passes = current["pass"]

current.update(phase="generic", **{"pass": 0})
generic_root = root / "generic"
generic_results = generic_root / "results"
generic_specs = []
for index in range(2):
    identity = RunIdentity(
        provider="toy", suite="suite", task_id=str(index), split_repeat=0,
        fold=0, sample=0, suite_repeat=1, model_route=f"oom-route-{index}",
        seed=index + 11, split_sha256="split", run_signature="generic-signature",
    )
    generic_specs.append(RunSpec(
        identity, DatasetTaskRef("toy", "suite", str(index)), "probe",
        ModelRoute("ADP-W-All", init_depth=1, init_width=index + 1),
        "split.npz", str(generic_results / str(index)), {}, ("python", "worker.py"),
    ))
plan = SuitePlan(
    "generic-signature", {"run_signature": "generic-signature", "jobs": [s.as_dict() for s in generic_specs]},
    tuple(generic_specs),
)
def generic_host_sample():
    current["pass"] += 1
    return HostMemorySample(8192, 8192, 0.0, 0, 0, 0.0)

def generic_gpu_sample(_index):
    if current["pass"] == 3:
        return GPUMemorySample(1000, 800, 200, 80.0)
    if current["pass"] >= 4:
        return GPUMemorySample(1000, 200, 800, 20.0)
    return GPUMemorySample(1000, 100, 900, 10.0)

provider = CallbackResourceProvider(
    host_sampler=generic_host_sample,
    gpu_sampler=generic_gpu_sample,
    tree_sampler=lambda pid: frozenset({pid}),
    child_sampler=lambda pid, **_kwargs: ProcessMemorySample(
        host_rss_mib=generic_memory_by_pid.get(pid, (100.0, 0.0))[0],
        gpu_vram_mib=generic_memory_by_pid.get(pid, (100.0, 0.0))[1],
    ),
    external_host_sampler=lambda _excluded: ProcessMemorySample(),
    external_gpu_sampler=lambda _index, _excluded: ProcessMemorySample(),
)
processes.launcher_child_env = lambda *_args, **_kwargs: {"PATH": "/usr/bin"}
import simple_adp.runner.scheduler as scheduler_module
scheduler_module.terminate_process_tree = lambda *_args, **_kwargs: None
coordinator = SuiteCoordinator(
    plan,
    RunnerSettings(
        concurrency=2, max_retries=1, gpu_resume_avail_mib=200,
        pressure_poll_seconds=0.1, post_launch_sample_delay_seconds=0.0,
        run_hardness_screen=False,
    ),
    run_root=generic_root / "run", results_dir=generic_results,
    repository_root=repository, resource_provider=provider,
)
coordinator.sleep = lambda _seconds: None
generic_report = coordinator.run()
generic_states = [coordinator.state_store.load(spec.identity.run_id) for spec in generic_specs]
generic_launches = len(popen_calls) - source_launches

assert source_passes >= 4 and current["pass"] >= 4
assert source_report["completed_jobs"] == generic_report["completed"] == 2
assert source_report["failed_attempts"] == generic_report["failed_attempts"] == 1
assert source_report["retried_jobs"] == generic_report["retries"] == 1
assert source_launches == generic_launches == 4
assert [state["status"] for state in source_states] == [
    state["status"] for state in generic_states
] == ["completed", "completed"]
assert [state["attempt"] for state in source_states] == [
    state["attempt"] for state in generic_states
] == [2, 2]
assert [state["retry_count"] for state in source_states] == [
    state["retry_count"] for state in generic_states
] == [1, 0]
assert [state["pause_count"] for state in source_states] == [
    state["pause_count"] for state in generic_states
] == [0, 1]
assert [state["gpu_pause_count"] for state in source_states] == [
    state["gpu_pause_count"] for state in generic_states
] == [1, 1]
assert [state["device_mode"] for state in source_states] == [
    state["device_mode"] for state in generic_states
] == ["gpu", "gpu"]
source_backoff = json.loads((source_run_root / "batch_backoff_state.json").read_text())
assert source_backoff["batch_scale"] == coordinator.admission.batch_scale == 1.0
print("source main and generic coordinator matched GPU OOM peer pause, retry, and resume")
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository), str(tmp_path)],
        cwd=repository, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout
    assert result.stdout.strip().endswith(
        "source main and generic coordinator matched GPU OOM peer pause, retry, and resume"
    )


@pytest.mark.parametrize("pressure_kind", ("gpu", "host", "both"))
def test_source_main_and_generic_coordinator_pressure_victim_resume_match(tmp_path, pressure_kind):
    """Compare full-loop lane-specific pressure victim scoring and resume."""
    repository = Path(__file__).parents[1]
    script = r'''
import csv
import importlib
import json
import os
import sys
from pathlib import Path

repository = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
pressure_kind = sys.argv[3]
for key in (
    "CPU_ONLY", "TABULAR_CPU_ONLY", "TRAINING_CONTROL_CPU_ONLY",
    "TRAINING_CONTROL_BACKEND", "TABULAR_BACKEND",
    "OPF_ADP_DISABLE_GPU_ACCELERATORS", "SIMPLE_ADP_DISABLE_GPU_ACCELERATORS",
):
    os.environ.pop(key, None)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
reference = importlib.import_module("utils.opf_massive_suite_runner")
from simple_adp.runner import processes
from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.datasets import DatasetTaskRef
from simple_adp.runner.models import ModelRoute
from simple_adp.runner.plan import RunIdentity, RunSpec, SuitePlan
from simple_adp.runner.resources import (
    CallbackResourceProvider, GPUMemorySample, HostMemorySample, ProcessMemorySample,
)
from simple_adp.runner.scheduler import SuiteCoordinator

current = {"phase": "source", "pass": 0}
popen_calls = []
generic_vram_by_pid = {}
generic_host_by_pid = {}
phase_launches = {"source": 0, "generic": 0}
class Child:
    def __init__(self, pid, return_code):
        self.pid = pid
        self.return_code = return_code
    def poll(self):
        if self.return_code is None:
            return 0 if current["pass"] >= 4 else None
        return self.return_code

def fake_popen(command, **kwargs):
    call_index = len(popen_calls)
    pid = 8000 + call_index
    return_code = None
    popen_calls.append((list(command), kwargs, pid))
    if current["phase"] == "generic":
        local_index = phase_launches["generic"]
        generic_vram_by_pid[pid] = 100.0 if local_index == 0 else 400.0
        generic_host_by_pid[pid] = 500.0 if local_index == 0 else 100.0
    phase_launches[current["phase"]] += 1
    return Child(pid, return_code)

source_root = root / "source"
source_results = source_root / "results"
source_specs = [
    reference.JobSpec(
        job_id=f"gpu-pressure-{index}", phase="probe", family="ADP", case_name="toy",
        repeat_index=0, depth=1, width=index + 1, seed=index + 21,
        results_dir=str(source_results / str(index)), command=("python", "worker.py"),
    )
    for index in range(2)
]
sys.argv = [
    "opf_massive_suite_runner", "--mode", "massive",
    "--results-dir", str(source_results), "--run-root", str(source_root / "run"),
    "--concurrency", "2", "--repeat-count", "1", "--max-retries", "1",
    "--pressure-poll-interval-sec", "0.1", "--post-launch-sample-delay-sec", "0",
]
source_args = reference.parse_args()
reference.parse_args = lambda: source_args
reference.validate_args = lambda _args: None
reference.configure_root_logging = lambda: None
reference.set_logging_context_provider = lambda _provider: None
reference.bootstrap_runtime = lambda *_args: None
reference.enable_optional_gpu_accelerators = lambda: None
reference.seed_run_root_from_source = lambda *_args: {"files_copied": 0, "files_skipped": 0}
reference.build_suite_jobs = lambda *_args: source_specs

def source_host_sample():
    current["pass"] += 1
    available = 1000 if current["pass"] == 3 and pressure_kind in {"host", "both"} else 8192
    return reference.MemorySample(8192, available, 0.0, 0, 0, 0.0)

def source_gpu_sample(_index=0):
    if current["pass"] == 3 and pressure_kind in {"gpu", "both"}:
        return reference.GpuMemorySample(1000, 950, 50, 95.0)
    if current["pass"] >= 4:
        return reference.GpuMemorySample(1000, 200, 800, 20.0)
    return reference.GpuMemorySample(1000, 100, 900, 10.0)

reference.sample_host_memory = source_host_sample
reference.sample_gpu_memory = source_gpu_sample
reference.current_pids = lambda _runtimes: []
reference.sample_non_python_host_memory = lambda **_kwargs: (0.0, [])
reference.sample_non_python_gpu_memory = lambda *_args, **_kwargs: (0.0, [], [])
reference.sample_runtime_child_memory = lambda runtime, *_args, **_kwargs: (
    500.0 if runtime.spec.job_id == "gpu-pressure-0" else 100.0,
    100.0 if runtime.spec.job_id == "gpu-pressure-0" else 400.0, {},
)
reference.make_child_env = lambda *_args, **_kwargs: {"PATH": "/usr/bin"}
reference.subprocess.Popen = fake_popen
reference._terminate_process_tree = lambda *_args, **_kwargs: None
reference.time.sleep = lambda _seconds: None
reference.REPO_ROOT = repository
reference.main()
source_run_root = source_root / "run"
source_states = [
    reference.load_job_state(reference.job_directory(source_run_root, spec) / "job_state.json")
    for spec in source_specs
]
source_report = json.loads((source_run_root / "final_report.json").read_text())
source_launches = len(popen_calls)
source_passes = current["pass"]

current.update(phase="generic", **{"pass": 0})
generic_root = root / "generic"
generic_results = generic_root / "results"
generic_specs = []
for index in range(2):
    identity = RunIdentity(
        provider="toy", suite="suite", task_id=str(index), split_repeat=0,
        fold=0, sample=0, suite_repeat=1, model_route=f"gpu-pressure-{index}",
        seed=index + 21, split_sha256="split", run_signature="generic-signature",
    )
    generic_specs.append(RunSpec(
        identity, DatasetTaskRef("toy", "suite", str(index)), "probe",
        ModelRoute("ADP-W-All", init_depth=1, init_width=index + 1),
        "split.npz", str(generic_results / str(index)), {}, ("python", "worker.py"),
    ))
plan = SuitePlan(
    "generic-signature", {"run_signature": "generic-signature", "jobs": [s.as_dict() for s in generic_specs]},
    tuple(generic_specs),
)
def generic_host_sample():
    current["pass"] += 1
    available = 1000 if current["pass"] == 3 and pressure_kind in {"host", "both"} else 8192
    return HostMemorySample(8192, available, 0.0, 0, 0, 0.0)

def generic_gpu_sample(_index):
    if current["pass"] == 3 and pressure_kind in {"gpu", "both"}:
        return GPUMemorySample(1000, 950, 50, 95.0)
    if current["pass"] >= 4:
        return GPUMemorySample(1000, 200, 800, 20.0)
    return GPUMemorySample(1000, 100, 900, 10.0)

provider = CallbackResourceProvider(
    host_sampler=generic_host_sample,
    gpu_sampler=generic_gpu_sample,
    tree_sampler=lambda pid: frozenset({pid}),
    child_sampler=lambda pid, **_kwargs: ProcessMemorySample(
        host_rss_mib=generic_host_by_pid.get(pid, 100.0),
        gpu_vram_mib=generic_vram_by_pid.get(pid, 400.0),
    ),
    external_host_sampler=lambda _excluded: ProcessMemorySample(),
    external_gpu_sampler=lambda _index, _excluded: ProcessMemorySample(),
)
processes.launcher_child_env = lambda *_args, **_kwargs: {"PATH": "/usr/bin"}
import simple_adp.runner.scheduler as scheduler_module
scheduler_module.terminate_process_tree = lambda *_args, **_kwargs: None
coordinator = SuiteCoordinator(
    plan,
    RunnerSettings(
        concurrency=2, max_retries=1, gpu_resume_avail_mib=500,
        pressure_poll_seconds=0.1, post_launch_sample_delay_seconds=0.0,
        run_hardness_screen=False,
    ),
    run_root=generic_root / "run", results_dir=generic_results,
    repository_root=repository, resource_provider=provider,
)
coordinator.sleep = lambda _seconds: None
generic_report = coordinator.run()
generic_states = [coordinator.state_store.load(spec.identity.run_id) for spec in generic_specs]
generic_launches = len(popen_calls) - source_launches

assert source_passes >= 4 and current["pass"] >= 4
assert source_report["completed_jobs"] == generic_report["completed"] == 2
assert source_report["failed_attempts"] == generic_report["failed_attempts"] == 0
assert source_launches == generic_launches == 3
assert [state["status"] for state in source_states] == [
    state["status"] for state in generic_states
] == ["completed", "completed"]
expected_attempts = [2, 1] if pressure_kind == "host" else [1, 2]
assert [state["attempt"] for state in source_states] == [
    state["attempt"] for state in generic_states
] == expected_attempts
expected_pauses = [1, 0] if pressure_kind == "host" else [0, 1]
assert [state["pause_count"] for state in source_states] == [
    state["pause_count"] for state in generic_states
] == expected_pauses
expected_gpu_pauses = [0, 0] if pressure_kind == "host" else expected_pauses
assert [state["gpu_pause_count"] for state in source_states] == [
    state["gpu_pause_count"] for state in generic_states
] == expected_gpu_pauses
expected_host_pauses = [0, 0] if pressure_kind == "gpu" else expected_pauses
assert [state["host_pause_count"] for state in source_states] == [
    state["host_pause_count"] for state in generic_states
] == expected_host_pauses
assert [state["device_mode"] for state in source_states] == [
    state["device_mode"] for state in generic_states
] == ["gpu", "gpu"]
source_events = list(csv.DictReader((source_run_root / "run_progress.csv").open(newline="")))
generic_events = list(csv.DictReader((generic_root / "run" / "events.csv").open(newline="")))
expected_reason = {
    "gpu": "gpu_memory_pressure",
    "host": "host_ram_pressure",
    "both": "gpu_memory_pressure+host_ram_pressure",
}[pressure_kind]
expected_victim = "gpu-pressure-0" if pressure_kind == "host" else "gpu-pressure-1"
assert any(
    row["event"] == "pause"
    and row["reason"] == expected_reason
    and row["job_id"] == expected_victim
    for row in source_events
)
assert any(
    row["event"] == "pause"
    and row["reason"] == expected_reason
    and row["job_id"] == generic_specs[0 if pressure_kind == "host" else 1].identity.run_id
    for row in generic_events
)
print(f"source main and generic coordinator matched {pressure_kind} pressure victim and resume")
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository), str(tmp_path), pressure_kind],
        cwd=repository, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout
    assert result.stdout.strip().endswith(
        f"source main and generic coordinator matched {pressure_kind} pressure victim and resume"
    )


def test_combined_pressure_without_gpu_child_matches_source_victim_filter(tmp_path):
    """Combined pressure selects the GPU victim pool even when it is empty."""
    repository = Path(__file__).parents[1]
    script = r'''
import csv
import importlib
import os
import sys
from pathlib import Path

repository = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
for name in ("CPU_ONLY", "TRAINING_CONTROL_CPU_ONLY", "TRAINING_CONTROL_BACKEND",
             "TABULAR_CPU_ONLY", "TABULAR_BACKEND", "OPF_ADP_DISABLE_GPU_ACCELERATORS",
             "SIMPLE_ADP_DISABLE_GPU_ACCELERATORS"):
    os.environ.pop(name, None)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
reference = importlib.import_module("utils.opf_massive_suite_runner")
from simple_adp.runner import processes
from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.datasets import DatasetTaskRef
from simple_adp.runner.models import ModelRoute
from simple_adp.runner.plan import RunIdentity, RunSpec, SuitePlan
from simple_adp.runner.resources import CallbackResourceProvider, GPUMemorySample, HostMemorySample, ProcessMemorySample
from simple_adp.runner.scheduler import SuiteCoordinator

current = {"phase": "source", "sample": 0}
popen_calls = []
class Child:
    pid = 54321
    def poll(self):
        return 0 if current["sample"] >= 3 else None

def fake_popen(command, **kwargs):
    popen_calls.append((list(command), kwargs))
    return Child()

source_root = root / "source"
source_results = source_root / "results"
source_spec = reference.JobSpec(
    job_id="cpu-only-child", phase="probe", family="ADP", case_name="toy",
    repeat_index=0, depth=1, width=1, seed=9,
    results_dir=str(source_results / "task"), command=("python", "worker.py"),
)
sys.argv = ["opf_massive_suite_runner", "--mode", "massive",
    "--results-dir", str(source_results), "--run-root", str(source_root / "run"),
    "--cpu-only", "--concurrency", "1", "--repeat-count", "1",
    "--pressure-poll-interval-sec", "0.1", "--post-launch-sample-delay-sec", "0"]
args = reference.parse_args()
reference.parse_args = lambda: args
reference.validate_args = lambda _args: None
reference.configure_root_logging = lambda: None
reference.set_logging_context_provider = lambda _provider: None
reference.bootstrap_runtime = lambda *_args: None
reference.enable_optional_gpu_accelerators = lambda: None
reference.seed_run_root_from_source = lambda *_args: {"files_copied": 0, "files_skipped": 0}
reference.build_suite_jobs = lambda *_args: [source_spec]
def source_host():
    current["sample"] += 1
    available = 500 if current["sample"] == 2 else 8192
    return reference.MemorySample(8192, available, 0.0, 0, 0, 0.0)
def source_gpu(_index=0):
    return reference.GpuMemorySample(1000, 950, 50, 95.0) if current["sample"] == 2 else reference.GpuMemorySample(1000, 100, 900, 10.0)
reference.sample_host_memory = source_host
reference.sample_gpu_memory = source_gpu
reference.current_pids = lambda _runtimes: []
reference.sample_non_python_host_memory = lambda **_kwargs: (0.0, [])
reference.sample_non_python_gpu_memory = lambda *_args, **_kwargs: (0.0, [], [])
reference.sample_runtime_child_memory = lambda *_args, **_kwargs: (100.0, 0.0, {})
reference.make_child_env = lambda *_args, **_kwargs: {"PATH": "/usr/bin"}
reference.subprocess.Popen = fake_popen
reference._terminate_process_tree = lambda *_args, **_kwargs: None
reference.time.sleep = lambda _seconds: None
reference.REPO_ROOT = repository
reference.main()
source_state = reference.load_job_state(reference.job_directory(source_root / "run", source_spec) / "job_state.json")
source_events = list(csv.DictReader((source_root / "run" / "run_progress.csv").open(newline="")))
source_samples = current["sample"]

current.update(phase="generic", sample=0)
generic_root = root / "generic"
generic_results = generic_root / "results"
identity = RunIdentity(provider="toy", suite="suite", task_id="toy", split_repeat=0,
    fold=0, sample=0, suite_repeat=1, model_route="cpu-only-child", seed=9,
    split_sha256="split", run_signature="generic-signature")
spec = RunSpec(identity, DatasetTaskRef("toy", "suite", "toy"), "probe",
    ModelRoute("ADP-W-All", init_depth=1, init_width=1), "split.npz",
    str(generic_results / "task"), {}, ("python", "worker.py"))
plan = SuitePlan("generic-signature", {"run_signature": "generic-signature", "jobs": [spec.as_dict()]}, (spec,))
def generic_host():
    current["sample"] += 1
    available = 500 if current["sample"] == 2 else 8192
    return HostMemorySample(8192, available, 0.0, 0, 0, 0.0)
def generic_gpu(_index):
    return GPUMemorySample(1000, 950, 50, 95.0) if current["sample"] == 2 else GPUMemorySample(1000, 100, 900, 10.0)
provider = CallbackResourceProvider(host_sampler=generic_host, gpu_sampler=generic_gpu,
    tree_sampler=lambda _pid: frozenset(), child_sampler=lambda _pid, **_kwargs: ProcessMemorySample(host_rss_mib=100),
    external_host_sampler=lambda _excluded: ProcessMemorySample(),
    external_gpu_sampler=lambda _index, _excluded: ProcessMemorySample())
processes.launcher_child_env = lambda *_args, **_kwargs: {"PATH": "/usr/bin"}
import simple_adp.runner.scheduler as scheduler
scheduler.terminate_process_tree = lambda _process: None
coordinator = SuiteCoordinator(plan, RunnerSettings(cpu_only=True, concurrency=1,
    pressure_poll_seconds=0.1, post_launch_sample_delay_seconds=0,
    run_hardness_screen=False), run_root=generic_root / "run", results_dir=generic_results,
    repository_root=repository, resource_provider=provider)
coordinator.sleep = lambda _seconds: None
generic_report = coordinator.run()
generic_state = coordinator.state_store.load(spec.identity.run_id)
generic_events = list(csv.DictReader((generic_root / "run" / "events.csv").open(newline="")))

assert source_samples >= 3 and current["sample"] >= 3
assert source_state["status"] == generic_state["status"] == "completed"
assert source_state["pause_count"] == generic_state["pause_count"] == 0
assert source_state["attempt"] == generic_state["attempt"] == 1
assert not any(row["event"] == "pause" for row in source_events)
assert not any(row["event"] == "pause" for row in generic_events)
assert generic_report["completed"] == 1
print("combined pressure with no GPU child matched source; CPU child was not paused")
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository), str(tmp_path)],
        cwd=repository, check=False, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout
    assert result.stdout.strip().endswith(
        "combined pressure with no GPU child matched source; CPU child was not paused"
    )


def test_cpu_completion_lane_reenable_with_closed_gpu_gate_matches_source(tmp_path):
    """CPU completion leaves GPU launch enablement unchanged while gate stays closed."""
    repository = Path(__file__).parents[1]
    script = r'''
import csv
import importlib
import os
import sys
from pathlib import Path

repository = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
for name in ("CPU_ONLY", "TABULAR_CPU_ONLY", "TRAINING_CONTROL_CPU_ONLY",
             "TRAINING_CONTROL_BACKEND", "TABULAR_BACKEND",
             "OPF_ADP_DISABLE_GPU_ACCELERATORS", "SIMPLE_ADP_DISABLE_GPU_ACCELERATORS"):
    os.environ.pop(name, None)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
reference = importlib.import_module("utils.opf_massive_suite_runner")
from simple_adp.runner import processes
from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.datasets import DatasetTaskRef
from simple_adp.runner.models import ModelRoute
from simple_adp.runner.plan import RunIdentity, RunSpec, SuitePlan
from simple_adp.runner.resources import CallbackResourceProvider, GPUMemorySample, HostMemorySample, ProcessMemorySample
from simple_adp.runner.scheduler import SuiteCoordinator

current = {"phase": "source", "sample": 0}
launches = []
generic_pid_memory = {}
class Child:
    def __init__(self, pid, success_sample):
        self.pid, self.success_sample = pid, success_sample
    def poll(self):
        return 0 if current["sample"] >= self.success_sample else None

def fake_popen(command, **kwargs):
    pid = 6000 + len(launches)
    local_index = sum(row[0] == current["phase"] for row in launches)
    success_sample = 3 if local_index == 0 else 5
    generic_pid_memory[pid] = 600.0 if current["phase"] == "generic" and local_index == 1 else 0.0
    launches.append((current["phase"], list(command), pid, success_sample))
    return Child(pid, success_sample)

source_root = root / "source"
source_results = source_root / "results"
source_specs = [
    reference.JobSpec(job_id="a-cpu-finished", phase="probe", family="ADP", case_name="toy",
        repeat_index=0, depth=1, width=1, seed=1, results_dir=str(source_results / "0"), command=("python", "worker.py")),
    reference.JobSpec(job_id="z-gpu-live", phase="probe", family="ADP", case_name="toy",
        repeat_index=0, depth=1, width=2, seed=2, results_dir=str(source_results / "1"), command=("python", "worker.py")),
    reference.JobSpec(job_id="zz-next", phase="probe", family="ADP", case_name="toy",
        repeat_index=0, depth=1, width=3, seed=3, results_dir=str(source_results / "2"), command=("python", "worker.py")),
]
sys.argv = ["opf_massive_suite_runner", "--mode", "massive", "--results-dir", str(source_results),
    "--run-root", str(source_root / "run"), "--concurrency", "2", "--max-active-gpu-jobs", "1",
    "--repeat-count", "1", "--gpu-child-gate-close-buffer-mib", "100",
    "--pressure-poll-interval-sec", "0.1", "--post-launch-sample-delay-sec", "0"]
args = reference.parse_args()
reference.parse_args = lambda: args
reference.validate_args = lambda _args: None
reference.configure_root_logging = lambda: None
reference.set_logging_context_provider = lambda _provider: None
reference.bootstrap_runtime = lambda *_args: None
reference.enable_optional_gpu_accelerators = lambda: None
reference.seed_run_root_from_source = lambda *_args: {"files_copied": 0, "files_skipped": 0}
reference.build_suite_jobs = lambda *_args: source_specs
def source_host():
    current["sample"] += 1
    return reference.MemorySample(8192, 8192, 0.0, 0, 0, 0.0)
def source_gpu(_index=0):
    if current["sample"] == 1:
        return None
    free = 650 if current["sample"] == 3 else 900
    return reference.GpuMemorySample(1000, 1000 - free, free, 100.0 * (1000 - free) / 1000)
reference.sample_host_memory = source_host
reference.sample_gpu_memory = source_gpu
reference.current_pids = lambda _runtimes: []
reference.sample_non_python_host_memory = lambda **_kwargs: (0.0, [])
reference.sample_non_python_gpu_memory = lambda *_args, **_kwargs: (0.0, [], [])
reference.sample_runtime_child_memory = lambda runtime, *_args, **_kwargs: (
    100.0, 600.0 if runtime.spec.job_id == "z-gpu-live" else 0.0, {})
reference.make_child_env = lambda *_args, **_kwargs: {"PATH": "/usr/bin"}
reference.subprocess.Popen = fake_popen
reference._terminate_process_tree = lambda *_args, **_kwargs: None
reference.time.sleep = lambda _seconds: None
reference.REPO_ROOT = repository
source_completion_states = []
original_update_backoff = reference.update_batch_backoff_state
def capture_source_backoff(path, state):
    frame = sys._getframe(1)
    while frame is not None and "completed_jobs" not in frame.f_locals:
        frame = frame.f_back
    completing_runtime = None if frame is None else frame.f_locals.get("runtime")
    if (state.get("gpu_gate_closed") and state.get("last_child_job_id") == "z-gpu-live"
            and frame is not None and frame.f_locals.get("completed_jobs", 0) > 0
            and getattr(completing_runtime, "device_mode", None) == "cpu"):
        source_completion_states.append(dict(state))
    return original_update_backoff(path, state)
reference.update_batch_backoff_state = capture_source_backoff
launch_contexts = []
source_launch = reference.launch_job
def capture_source_launch(*args, **kwargs):
    launch_contexts.append((args[1].spec.job_id, args[3], dict(kwargs.get("launch_context") or {})))
    return source_launch(*args, **kwargs)
reference.launch_job = capture_source_launch
reference.main()
source_run = source_root / "run"
source_states = [reference.load_job_state(reference.job_directory(source_run, spec) / "job_state.json") for spec in source_specs]
source_third_launch = launch_contexts[2]
source_samples = current["sample"]

current.update(phase="generic", sample=0)
generic_root = root / "generic"
generic_results = generic_root / "results"
generic_specs = []
for index, name in enumerate(("a-cpu-finished", "z-gpu-live", "zz-next")):
    identity = RunIdentity(provider="toy", suite="suite", task_id=str(index), split_repeat=0,
        fold=0, sample=0, suite_repeat=1, model_route=name, seed=index + 1,
        split_sha256="split", run_signature="generic-signature")
    generic_specs.append(RunSpec(identity, DatasetTaskRef("toy", "suite", str(index)), "probe",
        ModelRoute("ADP-W-All", init_depth=1, init_width=index + 1), "split.npz",
        str(generic_results / str(index)), {}, ("python", "worker.py")))
plan = SuitePlan("generic-signature", {"run_signature": "generic-signature", "jobs": [s.as_dict() for s in generic_specs]}, tuple(generic_specs))
def generic_host():
    current["sample"] += 1
    return HostMemorySample(8192, 8192, 0.0, 0, 0, 0.0)
def generic_gpu(_index):
    if current["sample"] == 1:
        return None
    free = 650 if current["sample"] == 3 else 900
    return GPUMemorySample(1000, 1000 - free, free, 100.0 * (1000 - free) / 1000)
provider = CallbackResourceProvider(host_sampler=generic_host, gpu_sampler=generic_gpu,
    tree_sampler=lambda pid: frozenset({pid}),
    child_sampler=lambda pid, **_kwargs: ProcessMemorySample(host_rss_mib=100,
        gpu_vram_mib=generic_pid_memory.get(pid, 0.0)),
    external_host_sampler=lambda _excluded: ProcessMemorySample(),
    external_gpu_sampler=lambda _index, _excluded: ProcessMemorySample())
processes.launcher_child_env = lambda *_args, **_kwargs: {"PATH": "/usr/bin"}
import simple_adp.runner.scheduler as scheduler
scheduler.terminate_process_tree = lambda _process: None
coordinator = SuiteCoordinator(plan, RunnerSettings(concurrency=2, max_active_gpu_jobs=1,
    child_gpu_gate_buffer_mib=100, pressure_poll_seconds=0.1, post_launch_sample_delay_seconds=0,
    run_hardness_screen=False), run_root=generic_root / "run", results_dir=generic_results,
    repository_root=repository, resource_provider=provider)
coordinator.sleep = lambda _seconds: None
generic_contexts = []
generic_launch = scheduler.launch_job
def capture_generic_launch(runtime, **kwargs):
    generic_contexts.append((runtime.job_id, kwargs.get("device"), dict(kwargs.get("launch_context") or {})))
    return generic_launch(runtime, **kwargs)
scheduler.launch_job = capture_generic_launch
generic_completion_states = []
original_save_controller_state = coordinator._save_controller_state
def capture_generic_state():
    original_save_controller_state()
    if coordinator.counters.completed and coordinator.admission.gpu_gate_closed:
        generic_completion_states.append({
            "gpu_gate_closed": coordinator.admission.gpu_gate_closed,
            "gpu_launches_enabled": coordinator.admission.gpu_launches_enabled,
            "cpu_launches_enabled": coordinator.admission.cpu_launches_enabled,
        })
coordinator._save_controller_state = capture_generic_state
# Make the CPU child sort before the GPU child; source IDs also put GPU last.
generic_report = coordinator.run()
generic_states = [coordinator.state_store.load(spec.identity.run_id) for spec in generic_specs]
generic_third_launch = generic_contexts[2]

assert source_samples >= 5 and current["sample"] >= 5
assert len(launch_contexts) == len(generic_contexts) == 3
assert [row[1] for row in launch_contexts] == [row[1] for row in generic_contexts] == ["cpu", "gpu", "cpu"]
assert source_third_launch[2]["gpu_gate_closed"] is True
assert generic_third_launch[2]["gates"]["gpu_closed"] is True
assert source_completion_states and source_completion_states[0]["gpu_gate_closed"] is True, source_completion_states
assert source_completion_states[0]["gpu_launches_enabled"] is False
assert source_completion_states[0]["cpu_launches_enabled"] is True
assert generic_completion_states and all(
    row == {"gpu_gate_closed": True, "gpu_launches_enabled": False, "cpu_launches_enabled": True}
    for row in generic_completion_states
), generic_completion_states
assert [row["status"] for row in source_states] == [row["status"] for row in generic_states] == ["completed"] * 3
assert [row["attempt"] for row in source_states] == [row["attempt"] for row in generic_states] == [1, 1, 1]
assert generic_report["completed"] == 3
print("CPU completion retained the GPU gate and both source/generic controllers fell back to CPU")
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository), str(tmp_path)],
        cwd=repository, check=False, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout
    assert result.stdout.strip().endswith(
        "CPU completion retained the GPU gate and both source/generic controllers fell back to CPU"
    )
def test_source_main_and_generic_coordinator_restart_a_real_child_after_host_pressure(tmp_path):
    """Compare real process-tree termination and same-lane pressure restart."""
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

repository = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
reference = importlib.import_module("utils.opf_massive_suite_runner")
from simple_adp.runner import processes
from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.datasets import DatasetTaskRef
from simple_adp.runner.models import ModelRoute
from simple_adp.runner.plan import RunIdentity, RunSpec, SuitePlan
from simple_adp.runner.resources import (
    CallbackResourceProvider, HostMemorySample, ProcessMemorySample,
)
from simple_adp.runner.scheduler import SuiteCoordinator

real_popen = subprocess.Popen
launches = {"source": [], "generic": []}
current = {"phase": "source"}
child_observed = {"source": False, "generic": False}
worker_code = (
    "import os, time; "
    "time.sleep(8.0) if os.environ.get('PRESSURE_FIDELITY_ATTEMPT') == '1' else None"
)
worker = [sys.executable, "-c", worker_code]

class ObservedPopen(real_popen):
    def __init__(self, command, *args, **kwargs):
        env = dict(kwargs.get("env") or os.environ)
        attempt = len(launches[current["phase"]]) + 1
        env["PRESSURE_FIDELITY_ATTEMPT"] = str(attempt)
        kwargs["env"] = env
        super().__init__(command, *args, **kwargs)
        launches[current["phase"]].append(self)

def host_sample(phase):
    active = launches[phase][-1] if launches[phase] else None
    pressure = (
        child_observed[phase]
        and len(launches[phase]) == 1
        and active is not None
        and active.poll() is None
    )
    available = 500 if pressure else 8192
    used_pct = 100.0 - 100.0 * available / 8192
    if phase == "source":
        return reference.MemorySample(8192, available, used_pct, 0, 0, 0.0)
    return HostMemorySample(8192, available, used_pct, 0, 0, 0.0)

source_root = root / "source"
source_results = source_root / "results"
source_spec = reference.JobSpec(
    job_id="real-pressure-child", phase="probe", family="ADP", case_name="toy",
    repeat_index=0, depth=1, width=2, seed=17,
    results_dir=str(source_results / "one"), command=tuple(worker),
)
sys.argv = [
    "opf_massive_suite_runner", "--mode", "massive",
    "--results-dir", str(source_results), "--run-root", str(source_root / "run"),
    "--cpu-only", "--concurrency", "1", "--repeat-count", "1",
    "--host-ram-stop-avail-mib", "1024", "--host-ram-resume-avail-mib", "1280",
    "--pressure-poll-interval-sec", "0.05", "--post-launch-sample-delay-sec", "0",
]
source_args = reference.parse_args()
reference.parse_args = lambda: source_args
reference.validate_args = lambda _args: None
reference.configure_root_logging = lambda: None
reference.set_logging_context_provider = lambda _provider: None
reference.bootstrap_runtime = lambda *_args: None
reference.enable_optional_gpu_accelerators = lambda: None
reference.seed_run_root_from_source = lambda *_args: {"files_copied": 0, "files_skipped": 0}
reference.build_suite_jobs = lambda *_args: [source_spec]
reference.sample_host_memory = lambda: host_sample("source")
reference.sample_gpu_memory = lambda *_args: None
reference.current_pids = lambda _runtimes: []
reference.sample_non_python_host_memory = lambda **_kwargs: (0.0, [])
reference.sample_non_python_gpu_memory = lambda *_args, **_kwargs: (0.0, [], [])
def source_child_sample(*_args, **_kwargs):
    child_observed["source"] = True
    return 100.0, 0.0, {}
reference.sample_runtime_child_memory = source_child_sample
reference.make_child_env = lambda *_args, **_kwargs: os.environ.copy()
reference.subprocess.Popen = ObservedPopen
reference.REPO_ROOT = repository
reference.main()
source_state = reference.load_job_state(
    reference.job_directory(source_root / "run", source_spec) / "job_state.json",
)
source_report = json.loads((source_root / "run" / "final_report.json").read_text())

current["phase"] = "generic"
generic_root = root / "generic"
generic_results = generic_root / "results"
identity = RunIdentity(
    provider="toy", suite="suite", task_id="one", split_repeat=0,
    fold=0, sample=0, suite_repeat=1, model_route="real-pressure-child",
    seed=17, split_sha256="split", run_signature="generic-signature",
)
generic_spec = RunSpec(
    identity, DatasetTaskRef("toy", "suite", "one"), "probe",
    ModelRoute("ADP-W-All", init_depth=1, init_width=2), "split.npz",
    str(generic_results / "one"), {}, tuple(worker),
)
plan = SuitePlan(
    "generic-signature", {"run_signature": "generic-signature", "jobs": [generic_spec.as_dict()]},
    (generic_spec,),
)
provider = CallbackResourceProvider(
    host_sampler=lambda: host_sample("generic"),
    gpu_sampler=lambda _index: None,
    tree_sampler=lambda pid: frozenset({pid}),
    child_sampler=lambda _pid, **_kwargs: (
        child_observed.__setitem__("generic", True)
        or ProcessMemorySample(host_rss_mib=100.0)
    ),
    external_host_sampler=lambda _excluded: ProcessMemorySample(),
    external_gpu_sampler=lambda _index, _excluded: ProcessMemorySample(),
)
processes.launcher_child_env = lambda *_args, **_kwargs: os.environ.copy()
processes.subprocess.Popen = ObservedPopen
coordinator = SuiteCoordinator(
    plan,
    RunnerSettings(
        cpu_only=True, concurrency=1, max_retries=1,
        host_ram_stop_avail_mib=1024, host_ram_resume_avail_mib=1280,
        pressure_poll_seconds=0.05, post_launch_sample_delay_seconds=0.0,
        run_hardness_screen=False,
    ),
    run_root=generic_root / "run", results_dir=generic_results,
    repository_root=repository, resource_provider=provider,
)
generic_report = coordinator.run()
generic_state = coordinator.state_store.load(identity.run_id)

assert source_report["completed_jobs"] == generic_report["completed"] == 1
assert source_report["failed_attempts"] == generic_report["failed_attempts"] == 0
assert len(launches["source"]) == len(launches["generic"]) == 2
assert all(
    process.poll() is not None
    for group in launches.values()
    for process in group
), [[process.pid, process.poll()] for group in launches.values() for process in group]
assert source_state["status"] == generic_state["status"] == "completed"
assert source_state["attempt"] == generic_state["attempt"] == 2
assert source_state["pause_count"] == generic_state["pause_count"] == 1
assert source_state["host_pause_count"] == generic_state["host_pause_count"] == 1
assert source_state["gpu_pause_count"] == generic_state["gpu_pause_count"] == 0
assert source_state["device_mode"] == generic_state["device_mode"] == "cpu"
print("source main and generic coordinator matched real-child pressure termination and CPU relaunch")
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository), str(tmp_path)],
        cwd=repository, check=False, capture_output=True, text=True, timeout=25,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout
    assert result.stdout.strip().endswith(
        "source main and generic coordinator matched real-child pressure termination and CPU relaunch"
    )


def test_source_main_and_generic_coordinator_resume_all_paused_jobs_past_active_caps(tmp_path):
    """Compare the source's same-pass resume bypass of launch admission limits."""
    repository = Path(__file__).parents[1]
    script = r'''
import csv
import importlib
import json
import sys
from collections import deque
from pathlib import Path

repository = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
reference = importlib.import_module("utils.opf_massive_suite_runner")
from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.datasets import DatasetTaskRef
from simple_adp.runner.models import ModelRoute
from simple_adp.runner.plan import RunIdentity, RunSpec, SuitePlan
from simple_adp.runner.resources import (
    CallbackResourceProvider, HostMemorySample, ProcessMemorySample,
)
from simple_adp.runner.scheduler import SuiteCoordinator
from simple_adp.runner.state import JobRuntime as GenericRuntime, RunStatus

class ExitChild:
    def __init__(self, pid):
        self.pid = pid
    def poll(self):
        return 0

current = {"phase": "source"}
source_launches = []
generic_launches = []
source_root = root / "source"
source_results = source_root / "results"
source_specs = [
    reference.JobSpec(
        job_id=f"paused-{index}", phase="probe", family="ADP", case_name="toy",
        repeat_index=0, depth=1, width=index + 1, seed=index + 10,
        results_dir=str(source_results / str(index)), command=("python", "worker.py"),
    )
    for index in range(2)
]
source_runtimes = {
    spec.job_id: reference.JobRuntime(
        spec=spec,
        state_path=reference.job_directory(source_root / "run", spec) / "job_state.json",
        log_path=reference.job_directory(source_root / "run", spec) / "job.log",
        device_mode="cpu", attempt=1, paused=True, pause_reason="host_ram_pressure",
    )
    for spec in source_specs
}

def source_load(_run_root, spec):
    return source_runtimes[spec.job_id]

def source_launch(_args, runtime, _concurrency, device_mode, _run_root, launch_context=None):
    runtime.attempt += 1
    runtime.device_mode = device_mode
    runtime.paused = False
    runtime.pause_reason = ""
    runtime.proc = ExitChild(7100 + len(source_launches))
    source_launches.append((runtime.spec.job_id, device_mode, dict(launch_context or {})))
    return runtime.proc

sys.argv = [
    "opf_massive_suite_runner", "--mode", "massive",
    "--results-dir", str(source_results), "--run-root", str(source_root / "run"),
    "--cpu-only", "--concurrency", "1", "--max-active-jobs", "1",
    "--repeat-count", "1", "--pressure-poll-interval-sec", "0.1",
    "--post-launch-sample-delay-sec", "0",
]
source_args = reference.parse_args()
reference.parse_args = lambda: source_args
reference.validate_args = lambda _args: None
reference.configure_root_logging = lambda: None
reference.set_logging_context_provider = lambda _provider: None
reference.bootstrap_runtime = lambda *_args: None
reference.enable_optional_gpu_accelerators = lambda: None
reference.seed_run_root_from_source = lambda *_args: {"files_copied": 0, "files_skipped": 0}
reference.build_suite_jobs = lambda *_args: source_specs
reference.load_existing_runtime = source_load
reference.launch_job = source_launch
reference.sample_host_memory = lambda: reference.MemorySample(8192, 8192, 0.0, 0, 0, 0.0)
reference.sample_gpu_memory = lambda *_args: None
reference.current_pids = lambda _runtimes: []
reference.sample_non_python_host_memory = lambda **_kwargs: (0.0, [])
reference.sample_non_python_gpu_memory = lambda *_args, **_kwargs: (0.0, [], [])
reference.sample_runtime_child_memory = lambda *_args, **_kwargs: (0.0, 0.0, {})
reference.time.sleep = lambda _seconds: None
reference.REPO_ROOT = repository
reference.main()
source_report = json.loads((source_root / "run" / "final_report.json").read_text())
source_states = [
    reference.load_job_state(reference.job_directory(source_root / "run", spec) / "job_state.json")
    for spec in source_specs
]

current["phase"] = "generic"
generic_root = root / "generic"
generic_results = generic_root / "results"
generic_specs = []
for index in range(2):
    identity = RunIdentity(
        provider="toy", suite="suite", task_id=str(index), split_repeat=0,
        fold=0, sample=0, suite_repeat=1, model_route=f"paused-{index}",
        seed=index + 10, split_sha256="split", run_signature="generic-signature",
    )
    generic_specs.append(RunSpec(
        identity, DatasetTaskRef("toy", "suite", str(index)), "probe",
        ModelRoute("ADP-W-All", init_depth=1, init_width=index + 1),
        "split.npz", str(generic_results / str(index)), {}, ("python", "worker.py"),
    ))
plan = SuitePlan(
    "generic-signature", {"run_signature": "generic-signature", "jobs": [s.as_dict() for s in generic_specs]},
    tuple(generic_specs),
)
provider = CallbackResourceProvider(
    host_sampler=lambda: HostMemorySample(8192, 8192, 0.0, 0, 0, 0.0),
    gpu_sampler=lambda _index: None,
    tree_sampler=lambda pid: frozenset({pid}),
    child_sampler=lambda _pid, **_kwargs: ProcessMemorySample(),
    external_host_sampler=lambda _excluded: ProcessMemorySample(),
    external_gpu_sampler=lambda _index, _excluded: ProcessMemorySample(),
)
coordinator = SuiteCoordinator(
    plan,
    RunnerSettings(
        cpu_only=True, concurrency=1, max_active_jobs=1,
        pressure_poll_seconds=0.1, post_launch_sample_delay_seconds=0.0,
        run_hardness_screen=False,
    ),
    run_root=generic_root / "run", results_dir=generic_results,
    repository_root=repository, resource_provider=provider,
)
generic_runtimes = [
    GenericRuntime(
        spec, status=RunStatus.PAUSED, attempt=1, device_mode="cpu",
        pause_reason="host_ram_pressure",
    )
    for spec in generic_specs
]
coordinator._build_runtimes = lambda: ([], list(generic_runtimes), deque())
import simple_adp.runner.scheduler as scheduler_module
def generic_launch(runtime, *, device, launch_context, **_kwargs):
    runtime.attempt += 1
    runtime.device_mode = device
    runtime.status = RunStatus.RUNNING
    runtime.process = ExitChild(8100 + len(generic_launches))
    runtime.pid = runtime.process.pid
    generic_launches.append((runtime.job_id, device, dict(launch_context)))
    return runtime.process
scheduler_module.launch_job = generic_launch
summary = coordinator.run()
generic_states = [coordinator.state_store.load(spec.identity.run_id) for spec in generic_specs]

assert len(source_launches) == len(generic_launches) == 2
assert source_report["completed_jobs"] == summary["completed"] == 2
assert [item[1] for item in source_launches] == [item[1] for item in generic_launches] == ["cpu", "cpu"]
assert all(item[2].get("resumed_from_pause") for item in source_launches + generic_launches)
assert [state["status"] for state in source_states] == [state["status"] for state in generic_states] == ["completed", "completed"]
assert [state["attempt"] for state in source_states] == [state["attempt"] for state in generic_states] == [2, 2]
assert [state["pause_count"] for state in source_states] == [state["pause_count"] for state in generic_states] == [0, 0]
with (source_root / "run" / "run_progress.csv").open(newline="", encoding="utf-8") as stream:
    source_resume_events = [row for row in csv.DictReader(stream) if row["event"] == "resume"]
with coordinator.state_store.events_path.open(newline="", encoding="utf-8") as stream:
    generic_resume_events = [row for row in csv.DictReader(stream) if row["event"] == "resume"]
assert len(source_resume_events) == len(generic_resume_events) == 2
assert [row["reason"] for row in source_resume_events] == [row["reason"] for row in generic_resume_events] == ["gate_reopened"] * 2
print("source main and generic coordinator matched same-pass paused-job resume beyond active caps")
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository), str(tmp_path)],
        cwd=repository, check=False, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout
    assert result.stdout.strip().endswith(
        "source main and generic coordinator matched same-pass paused-job resume beyond active caps"
    )


def test_source_and_generic_recovered_child_unknown_exit_requeue_match(tmp_path):
    """Compare the restart-recovery transition and FIFO placement directly."""
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import sys
from collections import deque
from pathlib import Path

repository = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
reference = importlib.import_module("utils.opf_massive_suite_runner")
from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.datasets import DatasetTaskRef
from simple_adp.runner.models import ModelRoute
from simple_adp.runner.plan import RunIdentity, RunSpec, SuitePlan
from simple_adp.runner.resources import HostMemorySample, ResourceSnapshot
from simple_adp.runner.scheduler import SuiteCoordinator
from simple_adp.runner.state import JobRuntime as GenericRuntime, RunStatus

def source_runtime(name):
    spec = reference.JobSpec(
        job_id=name, phase="probe", family="ADP", case_name="toy",
        repeat_index=0, depth=1, width=8, seed=3,
        results_dir=str(root / name), command=("python", "worker.py"),
    )
    state_path = root / "source" / name / "job_state.json"
    runtime = reference.JobRuntime(spec, state_path, state_path.parent / "job.log")
    runtime.attempt, runtime.retry_count = 3, 2
    runtime.proc = reference.RecoveredProcessHandle(777)
    return runtime

source_root = root / "source"
recovered = source_runtime("recovered")
queued = source_runtime("queued")
source_active = [recovered]
source_queue = deque([queued])
reference._is_live_pid = lambda _pid: False
assert reference._requeue_recovered_runtime_after_exit(
    recovered, source_active, source_queue,
)
source_state = reference.load_job_state(recovered.state_path)

def generic_runtime(name):
    identity = RunIdentity(
        provider="toy", suite="suite", task_id=name, split_repeat=0, fold=0,
        sample=0, suite_repeat=1, model_route=name, seed=3,
        split_sha256="split", run_signature="signature",
    )
    spec = RunSpec(
        identity, DatasetTaskRef("toy", "suite", name), "probe",
        ModelRoute("ADP-W-All", init_depth=1, init_width=8), "split.npz",
        str(root / "generic" / name), {}, ("python", "worker.py"),
    )
    return GenericRuntime(
        spec, status=RunStatus.RUNNING, attempt=3, retry_count=2,
        device_mode="gpu", pid=777, recovered_process=True,
        process=reference.RecoveredProcessHandle(777),
    )

generic_recovered = generic_runtime("recovered")
generic_queued = generic_runtime("queued")
plan = SuitePlan("signature", {"run_signature": "signature", "jobs": []},
                 (generic_recovered.spec, generic_queued.spec))
coordinator = SuiteCoordinator(
    plan, RunnerSettings(cpu_only=True), run_root=root / "generic-run",
    results_dir=root / "generic-results", repository_root=repository,
)
generic_active = [generic_recovered]
generic_queue = deque([generic_queued])
snapshot = ResourceSnapshot(
    HostMemorySample(8192, 4096, 50, 0, 0, 0), None,
    1, 0, 1, 0, 0, 0, 0, 0, frozenset(), frozenset(), False, False, 1,
)
coordinator._handle_exit(
    generic_recovered, generic_active, [], generic_queue, snapshot,
)
generic_state = coordinator.state_store.load(generic_recovered.job_id)

assert [item.spec.job_id for item in source_queue] == ["queued", "recovered"]
assert [item.spec.identity.task_id for item in generic_queue] == ["queued", "recovered"]
for state in (source_state, generic_state):
    assert state["status"] == "queued"
    assert state["attempt"] == 3
    assert state["retry_count"] == 2
assert source_state["last_exit_code"] is None
assert generic_state["exit_code"] is None
assert generic_state["last_exit_code"] is None
assert source_state["reason"] == "recovered_child_exit_status_unknown"
assert generic_state["last_error"] == "recovered_child_exit_status_unknown"
assert coordinator.counters.failed_attempts == coordinator.counters.retries == 0
assert coordinator.counters.recovered_unknown_exit == 1
events = coordinator.state_store.events_path.read_text(encoding="utf-8")
assert "recovered_child_exit_status_unknown" in events
print("source and generic recovered-child unknown-exit transitions match")
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository), str(tmp_path)],
        cwd=repository, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout
    assert result.stdout.strip().endswith(
        "source and generic recovered-child unknown-exit transitions match"
    )


def test_source_and_generic_startup_reconciliation_matrix_match(tmp_path):
    """Compare persisted-status/PID/argv recovery outcomes with the source."""
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

repository = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
reference = importlib.import_module("utils.opf_massive_suite_runner")
from simple_adp.runner.datasets import DatasetTaskRef
from simple_adp.runner.models import ModelRoute
from simple_adp.runner.plan import RunIdentity, RunSpec
from simple_adp.runner.processes import reconcile_runtime
from simple_adp.runner.state import JobRuntime as GenericRuntime

command = ("python", "worker.py", "--case", "recovery")
matched = SimpleNamespace(pid=9001)
source_spec = reference.JobSpec(
    job_id="recovery", phase="probe", family="ADP", case_name="toy",
    repeat_index=0, depth=1, width=2, seed=7,
    results_dir=str(root / "results"), command=command,
)
identity = RunIdentity(
    provider="toy", suite="suite", task_id="recovery", split_repeat=0,
    fold=0, sample=0, suite_repeat=1, model_route="recovery", seed=7,
    split_sha256="split", run_signature="signature",
)
generic_spec = RunSpec(
    identity, DatasetTaskRef("toy", "suite", "recovery"), "probe",
    ModelRoute("ADP-W-All", init_depth=1, init_width=2), "split.npz",
    str(root / "results"), {}, command,
)

source_dir = reference.job_directory(root / "source-run", source_spec)
source_dir.mkdir(parents=True, exist_ok=True)
source_state_path = source_dir / "job_state.json"
observed = 0
for status in ("running", "launching", "retrying", "paused", "queued", "failed", "completed", "unknown"):
    for live_pid in (False, True):
        for exact_match in (False, True):
            for pid in (4242, None):
                record = {
                    "job_id": "recovery", "status": status, "pid": pid,
                    "attempt": 2, "retry_count": 1, "device_mode": "gpu",
                    "command": list(command), "pause_reason": "host_ram_pressure",
                    "last_exit_code": 0 if status == "completed" else 7,
                    "exit_code": None, "ended_at": None,
                }
                source_state_path.write_text(json.dumps(record), encoding="utf-8")
                reference._is_live_pid = lambda value, live=live_pid, found=exact_match: (
                    bool(value) and (live or (found and int(value) == matched.pid))
                )
                reference._find_live_pid_for_command = lambda _value, found=exact_match: (
                    matched.pid if found else None
                )
                source_runtime = reference.load_existing_runtime(root / "source-run", source_spec)
                source_state = reference.load_job_state(source_state_path)
                if source_state.get("status") == "completed" and int(source_state.get("last_exit_code", 0)) == 0:
                    source_location = "completed"
                elif reference.is_terminal_inexact_resume_failure(source_state):
                    source_location = "terminal"
                elif source_runtime.status == "running":
                    source_location = "active"
                elif source_runtime.status == "paused":
                    source_location = "suspended"
                else:
                    source_location = "queued"

                from simple_adp.runner import processes
                processes.process_is_live = lambda value, live=live_pid, found=exact_match: (
                    bool(value) and (live or (found and int(value) == matched.pid))
                )
                processes.find_exact_argv = lambda _value, found=exact_match: matched if found else None
                generic_runtime = GenericRuntime(generic_spec)
                generic_location = reconcile_runtime(generic_runtime, record)
                assert generic_location == source_location, (
                    status, live_pid, exact_match, pid,
                    source_location, generic_location,
                )
                observed += 1
assert observed == 64
terminal_cases = 0
for status in ("failed", "fail"):
    source_record = {
            "job_id": "recovery", "status": status, "pid": 4242,
            "attempt": 2, "retry_count": 1, "device_mode": "gpu",
            "command": list(command), "reason": "inexact_checkpoint_resume",
            "last_exit_code": 9,
    }
    source_state_path.write_text(json.dumps(source_record), encoding="utf-8")
    reference._is_live_pid = lambda _value: True
    reference._find_live_pid_for_command = lambda _value: matched.pid
    source_runtime = reference.load_existing_runtime(root / "source-run", source_spec)
    source_state = reference.load_job_state(source_state_path)
    source_location = (
        "terminal" if reference.is_terminal_inexact_resume_failure(source_state)
        else "active" if source_runtime.status == "running" else "queued"
    )
    generic_record = {**source_record, "last_error": source_record.pop("reason")}
    from simple_adp.runner import processes
    processes.process_is_live = lambda _value: True
    processes.find_exact_argv = lambda _value: matched
    generic_location = reconcile_runtime(GenericRuntime(generic_spec), generic_record)
    assert source_location == generic_location == "terminal", (
        status, source_location, generic_location,
    )
    terminal_cases += 1
assert terminal_cases == 2
print("source and generic startup reconciliation matched 66 persisted-state cases")
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository), str(tmp_path)],
        cwd=repository, check=False, capture_output=True, text=True, timeout=25,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout
    assert result.stdout.strip().endswith(
        "source and generic startup reconciliation matched 66 persisted-state cases"
    )


def test_source_and_generic_resume_launch_failure_preserve_paused_state(tmp_path):
    """Compare failed pressure-resume persistence at both runner boundaries."""
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

repository = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
reference = importlib.import_module("utils.opf_massive_suite_runner")
from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.datasets import DatasetTaskRef
from simple_adp.runner.models import ModelRoute
from simple_adp.runner.plan import RunIdentity, RunSpec, SuitePlan
from simple_adp.runner.resources import ResourceSnapshot, HostMemorySample
from simple_adp.runner.scheduler import SuiteCoordinator
from simple_adp.runner.state import JobRuntime as GenericRuntime, RunStatus

source_root = root / "source"
source_spec = reference.JobSpec(
    job_id="paused", phase="probe", family="ADP", case_name="toy",
    repeat_index=0, depth=1, width=2, seed=7,
    results_dir=str(root / "source-results"), command=("python", "worker.py"),
)
source_path = reference.job_directory(source_root, source_spec) / "job_state.json"
source_runtime = reference.JobRuntime(
    source_spec, source_path, source_path.parent / "job.log",
    device_mode="gpu", paused=True, pause_reason="gpu_memory_pressure",
)
reference.launch_job = lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("cannot restart"))
try:
    reference._resume_runtime_from_pause(
        source_runtime, SimpleNamespace(), 1, source_root, device_mode="gpu",
        active=[], launch_context={"resumed_from_pause": True},
    )
except OSError:
    pass
else:
    raise AssertionError("source resume launch error did not propagate")
source_state = reference.load_job_state(source_path)

identity = RunIdentity(
    provider="toy", suite="suite", task_id="1", split_repeat=0, fold=0,
    sample=0, suite_repeat=1, model_route="paused", seed=7,
    split_sha256="split", run_signature="signature",
)
spec = RunSpec(
    identity, DatasetTaskRef("toy", "suite", "1"), "probe",
    ModelRoute("ADP-W-All"), "split.npz", str(root / "generic-results"),
    {}, ("python", "worker.py"),
)
coordinator = SuiteCoordinator(
    SuitePlan("signature", {"run_signature": "signature", "jobs": []}, (spec,)),
    RunnerSettings(cpu_only=True), run_root=root / "generic-run",
    results_dir=root / "generic-results", repository_root=repository,
)
generic_runtime = GenericRuntime(
    spec, status=RunStatus.PAUSED, device_mode="gpu",
    pause_reason="gpu_memory_pressure",
)
snapshot = ResourceSnapshot(
    HostMemorySample(8192, 4096, 50, 0, 0, 0), None,
    0, 0, 0, 0, 0, 0, 0, 0, frozenset(), frozenset(), False, False, 1,
)
import simple_adp.runner.scheduler as scheduler
scheduler.launch_job = lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("cannot restart"))
assert coordinator._resume(generic_runtime, [], snapshot, 1) is False
generic_state = coordinator.state_store.load(spec.identity.run_id)

assert source_state["status"] == generic_state["status"] == "paused"
assert source_state["pause_reason"] == generic_state["pause_reason"] == "gpu_memory_pressure"
assert source_state["reason"] == generic_state["last_error"] == "resume_launch_error"
assert source_state["pid"] is generic_state["pid"] is None
assert source_state["last_exit_code"] is generic_state["exit_code"] is None
assert generic_runtime.status == RunStatus.PAUSED and generic_runtime.last_error == "resume_launch_error"
assert not generic_runtime.pause_reason == ""
print("source and generic resume-launch failure state matched")
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository), str(tmp_path)],
        cwd=repository, check=False, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout
    assert result.stdout.strip().endswith("source and generic resume-launch failure state matched")


def test_source_main_and_generic_resume_failure_still_launch_queued_route(tmp_path):
    """Compare both controller loops across resume failure, FIFO launch, and interrupt."""
    repository = Path(__file__).parents[1]
    script = r'''
import csv
import importlib
import json
import sys
from collections import deque
from pathlib import Path

repository = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
reference = importlib.import_module("utils.opf_massive_suite_runner")
from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.datasets import DatasetTaskRef
from simple_adp.runner.models import ModelRoute
from simple_adp.runner.plan import RunIdentity, RunSpec, SuitePlan
from simple_adp.runner.resources import CallbackResourceProvider, HostMemorySample, ProcessMemorySample
from simple_adp.runner.scheduler import SuiteCoordinator
from simple_adp.runner.state import JobRuntime as GenericRuntime, RunStatus

class Child:
    def __init__(self, pid):
        self.pid = pid
    def poll(self):
        return 0

class CaptureLogger:
    def info(self, *_args, **_kwargs): pass
    def warning(self, *_args, **_kwargs): pass
    def exception(self, *_args, **_kwargs): pass

reference.logging.getLogger = lambda *_args, **_kwargs: CaptureLogger()
source_root = root / "source" / "run"
source_results = root / "source" / "results"
source_specs = []
source_runtimes = {}
for key in ("paused", "queued"):
    spec = reference.JobSpec(
        job_id=key, phase="probe", family="ADP", case_name="toy",
        repeat_index=0, depth=1, width=2, seed=7,
        results_dir=str(source_results / key), command=("python", "worker.py"),
    )
    state_path = reference.job_directory(source_root, spec) / "job_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    runtime = reference.JobRuntime(spec, state_path, state_path.parent / "job.log")
    if key == "paused":
        runtime.paused = True
        runtime.device_mode = "cpu"
        runtime.pause_reason = "host_ram_pressure"
        reference.save_job_state(state_path, {**runtime.to_state(), "status": "paused"})
    source_specs.append(spec)
    source_runtimes[key] = runtime

source_launches = []
def source_launch(_args, runtime, _concurrency, device, _run_root, launch_context=None):
    resumed = bool((launch_context or {}).get("resumed_from_pause"))
    source_launches.append((runtime.spec.job_id, resumed))
    runtime.attempt += 1
    if resumed:
        raise OSError("resume launch failed")
    runtime.proc = Child(9100)
    runtime.device_mode = device
    runtime.paused = False
    runtime.pause_reason = ""
    return runtime.proc

source_sleep_count = {"value": 0}
def source_sleep(_seconds):
    source_sleep_count["value"] += 1
    if source_sleep_count["value"] == 2:
        raise KeyboardInterrupt

sys.argv = [
    "opf_massive_suite_runner", "--mode", "massive",
    "--results-dir", str(source_results), "--run-root", str(source_root),
    "--cpu-only", "--concurrency", "2", "--repeat-count", "1",
    "--pressure-poll-interval-sec", "0.1", "--post-launch-sample-delay-sec", "0",
]
source_args = reference.parse_args()
reference.parse_args = lambda: source_args
reference.validate_args = lambda _args: None
reference.configure_root_logging = lambda: None
reference.set_logging_context_provider = lambda *_args: None
reference.bootstrap_runtime = lambda *_args: None
reference.enable_optional_gpu_accelerators = lambda: None
reference.seed_run_root_from_source = lambda *_args: {"files_copied": 0, "files_skipped": 0}
reference.build_suite_jobs = lambda *_args: source_specs
reference.load_existing_runtime = lambda _run_root, spec: source_runtimes[spec.job_id]
reference.launch_job = source_launch
reference.sample_host_memory = lambda: reference.MemorySample(8192, 8192, 0, 0, 0, 0)
reference.sample_gpu_memory = lambda *_args: None
reference.current_pids = lambda _runtimes: []
reference.sample_non_python_host_memory = lambda **_kwargs: (0.0, [])
reference.sample_non_python_gpu_memory = lambda *_args, **_kwargs: (0.0, [], [])
reference.sample_runtime_child_memory = lambda *_args, **_kwargs: (0.0, 0.0, {})
reference.time.sleep = source_sleep
reference.REPO_ROOT = repository
try:
    reference.main()
except KeyboardInterrupt:
    # main() has already run its own shutdown path; the __main__ wrapper
    # converts this to exit 130 in production.
    pass
else:
    raise AssertionError("source controller did not take the expected interrupt shutdown")
source_states = [
    reference.load_job_state(reference.job_directory(source_root, spec) / "job_state.json")
    for spec in source_specs
]

generic_root = root / "generic"
generic_results = generic_root / "results"
generic_specs = []
for key in ("paused", "queued"):
    identity = RunIdentity(
        provider="toy", suite="suite", task_id=key, split_repeat=0,
        fold=0, sample=0, suite_repeat=1, model_route=key, seed=7,
        split_sha256="split", run_signature="signature",
    )
    generic_specs.append(RunSpec(
        identity, DatasetTaskRef("toy", "suite", key), "probe",
        ModelRoute("ADP-W-All"), "split.npz", str(generic_results / key),
        {}, ("python", "worker.py"),
    ))
plan = SuitePlan("signature", {"run_signature": "signature", "jobs": []}, tuple(generic_specs))
provider = CallbackResourceProvider(
    host_sampler=lambda: HostMemorySample(8192, 8192, 0, 0, 0, 0),
    gpu_sampler=lambda _index: None,
    tree_sampler=lambda pid: frozenset({pid}),
    child_sampler=lambda _pid, **_kwargs: ProcessMemorySample(),
    external_host_sampler=lambda _excluded: ProcessMemorySample(),
    external_gpu_sampler=lambda _index, _excluded: ProcessMemorySample(),
)
coordinator = SuiteCoordinator(
    plan,
    RunnerSettings(
        cpu_only=True, concurrency=2, pressure_poll_seconds=0.1,
        post_launch_sample_delay_seconds=0, run_hardness_screen=False,
    ),
    run_root=generic_root / "run", results_dir=generic_results,
    repository_root=repository, resource_provider=provider,
)
paused = GenericRuntime(
    generic_specs[0], status=RunStatus.PAUSED, device_mode="cpu",
    pause_reason="host_ram_pressure",
)
queued = GenericRuntime(generic_specs[1])
coordinator._build_runtimes = lambda: ([], [paused], deque([queued]))
generic_launches = []
def generic_launch(runtime, *, device, launch_context, **_kwargs):
    resumed = bool(launch_context.get("resumed_from_pause"))
    generic_launches.append((runtime.spec.task_ref.task_id, resumed))
    runtime.attempt += 1
    if resumed:
        raise OSError("resume launch failed")
    runtime.process = Child(9200)
    runtime.pid = runtime.process.pid
    runtime.status = RunStatus.RUNNING
    runtime.device_mode = device
    return runtime.process

import simple_adp.runner.scheduler as scheduler
scheduler.launch_job = generic_launch
generic_sleep_count = {"value": 0}
def generic_sleep(_seconds):
    generic_sleep_count["value"] += 1
    if generic_sleep_count["value"] == 2:
        raise KeyboardInterrupt
coordinator.sleep = generic_sleep
try:
    coordinator.run()
except SystemExit as exc:
    assert exc.code == 130
else:
    raise AssertionError("generic controller did not take the expected interrupt shutdown")
generic_states = [coordinator.state_store.load(spec.identity.run_id) for spec in generic_specs]

assert source_launches == generic_launches == [
    ("paused", True), ("queued", False), ("paused", True),
], (source_launches, generic_launches)
assert [item["status"] for item in source_states] == ["paused", "completed"]
assert [item["status"] for item in generic_states] == ["paused", "completed"]
assert [item["attempt"] for item in source_states] == [2, 1]
assert [item["attempt"] for item in generic_states] == [2, 1]
assert source_states[0]["pause_reason"] == generic_states[0]["pause_reason"] == "host_ram_pressure"
assert source_states[0]["reason"] == generic_states[0]["last_error"] == "interrupted:KeyboardInterrupt"
assert source_states[1]["last_exit_code"] == generic_states[1]["last_exit_code"] == 0
print("source main and generic coordinator matched failed resume, queued launch, completion, and paused shutdown")
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository), str(tmp_path)],
        cwd=repository, check=False, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout
    assert result.stdout.strip().endswith(
        "source main and generic coordinator matched failed resume, queued launch, completion, and paused shutdown"
    )


def test_source_and_generic_hard_pressure_thresholds_match():
    """Execute the copied controller's pressure predicates at strict boundaries."""
    from pathlib import Path
    from types import SimpleNamespace

    repository = Path(__file__).parents[1]
    from simple_adp.runner.admission import hard_pressure
    from simple_adp.runner.config import RunnerSettings
    from simple_adp.runner.resources import GPUMemorySample, HostMemorySample

    source = (
        repository / "references" / "opf_adp_runner_suite" / "utils"
        / "opf_massive_suite_runner.py"
    ).read_text(encoding="utf-8").splitlines()
    source_lines = [
        line.strip() for line in source
        if line.strip().startswith("gpu_pressure = gpu_free_effective is not None")
        or line.strip().startswith("host_pressure = host_available_effective <")
    ]
    assert len(source_lines) == 2
    source_code = compile("\n".join(source_lines), "copied-runner-pressure.py", "exec")
    settings = RunnerSettings(host_ram_stop_avail_mib=1024, gpu_stop_avail_mib=100)
    host_values = (1023.999999, 1024.0, 1024.000001)
    gpu_values = (None, 99.999999, 100.0, 100.000001)

    checked = 0
    for host_available in host_values:
        for gpu_free in gpu_values:
            args = SimpleNamespace(gpu_stop_avail_mib=100, host_ram_stop_avail_mib=1024)
            source_values = {
                "args": args,
                "gpu_free_effective": gpu_free,
                "host_available_effective": host_available,
            }
            exec(source_code, {}, source_values)
            host = HostMemorySample(8192, host_available, 50, 0, 0, 0)
            gpu = None if gpu_free is None else GPUMemorySample(1000, 1000 - gpu_free, gpu_free, 0)
            assert (source_values["host_pressure"], source_values["gpu_pressure"]) == hard_pressure(
                settings, host, gpu,
            ), (host_available, gpu_free, source_values)
            checked += 1
    assert checked == 12


def test_source_and_generic_capacity_reread_timing_match(tmp_path):
    """Changing concurrency between capacity and active-count reads stays aligned."""
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

repository = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
reference = importlib.import_module("utils.opf_massive_suite_runner")
from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.datasets import DatasetTaskRef
from simple_adp.runner.models import ModelRoute
from simple_adp.runner.plan import RunIdentity, RunSpec
from simple_adp.runner.resources import (
    CallbackResourceProvider, HostMemorySample, ProcessMemorySample,
)
from simple_adp.runner.scheduler import SuiteCoordinator
from simple_adp.runner.state import JobRuntime, RunStatus

identity = RunIdentity(
    provider="toy", suite="suite", task_id="capacity", split_repeat=0,
    fold=0, sample=0, suite_repeat=1, model_route="capacity", seed=1,
    split_sha256="split", run_signature="signature",
)
spec = RunSpec(
    identity, DatasetTaskRef("toy", "suite", "capacity"), "probe",
    ModelRoute("ADP-W-All"), "split.npz", str(root / "result"), {},
    ("python", "worker.py"),
)
runtime = JobRuntime(
    spec, status=RunStatus.RUNNING, device_mode="cpu", attempt=1,
    started_at=1.0, pid=50, process=SimpleNamespace(pid=50, poll=lambda: None),
)
plan = SimpleNamespace(run_signature="signature", runs=(spec,))
provider = CallbackResourceProvider(
    host_sampler=lambda: HostMemorySample(8192, 4096, 50, 0, 0, 0),
    gpu_sampler=lambda _index: None,
    tree_sampler=lambda pid: frozenset({pid}),
    child_sampler=lambda _pid, **_kwargs: ProcessMemorySample(host_rss_mib=100),
    external_host_sampler=lambda _excluded: ProcessMemorySample(),
    external_gpu_sampler=lambda _index, _excluded: ProcessMemorySample(),
)

# These are the exact source caller expressions after the helper's own
# concurrency read; the generic side runs the corresponding _sample_snapshot.
source_lines = (
    "launch_capacity = reference.resolve_launch_capacity(args, 4096.0, None)",
    "if active_count > 0 and reference.resolve_concurrency(args) != reference.UNBOUNDED_CONCURRENCY:",
    "    launch_capacity = max(1, min(launch_capacity, active_count + 1))",
)
source_code = compile("\n".join(source_lines), "source-capacity-caller.py", "exec")
unbounded = reference.UNBOUNDED_CONCURRENCY
for sequence, expected in (((5, 5), 2), ((5, unbounded), 5), ((1, 1), 1)):
    source_reads = iter(sequence)
    reference.resolve_concurrency = lambda _args: next(source_reads)
    source_ns = {
        "reference": reference,
        "args": SimpleNamespace(
            concurrency=1, concurrency_file=None,
            launch_host_reserve_mib=0, launch_gpu_reserve_mib=0,
            max_active_jobs=0,
        ),
        "active_count": 1,
    }
    exec(source_code, source_ns)
    assert source_ns["launch_capacity"] == expected

    settings = RunnerSettings(concurrency=1, max_active_jobs=0)
    generic_reads = iter(sequence)
    RunnerSettings.resolve_concurrency = lambda _self: next(generic_reads)
    coordinator = SuiteCoordinator(
        plan, settings, run_root=root / f"generic-{sequence[0]}-{sequence[1]}",
        results_dir=root / "results", repository_root=repository,
        resource_provider=provider,
    )
    snapshot, _, _ = coordinator._sample_snapshot([runtime], [], concurrency=1)
    assert snapshot.launch_capacity == source_ns["launch_capacity"] == expected
    assert list(source_reads) == []
    assert list(generic_reads) == []
print("source and generic concurrency reread timing matches")
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository), str(tmp_path)],
        cwd=repository, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout
    assert result.stdout.strip().endswith("source and generic concurrency reread timing matches")


def test_source_main_and_generic_reread_concurrency_file_between_passes(tmp_path):
    """A live limit increase admits the next FIFO job in both real loops."""
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

repository = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
reference = importlib.import_module("utils.opf_massive_suite_runner")
from simple_adp.runner import processes
from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.datasets import DatasetTaskRef
from simple_adp.runner.models import ModelRoute
from simple_adp.runner.plan import RunIdentity, RunSpec, SuitePlan
from simple_adp.runner.resources import CallbackResourceProvider, HostMemorySample, ProcessMemorySample
from simple_adp.runner.scheduler import SuiteCoordinator

control = root / "concurrency.txt"
control.parent.mkdir(parents=True, exist_ok=True)
current = {"phase": "source", "pass": 0}
launches = {"source": [], "generic": []}

class Child:
    def __init__(self, pid):
        self.pid = pid
    def poll(self):
        return 0 if current["pass"] >= 3 else None

def fake_popen(command, **kwargs):
    phase = current["phase"]
    pid = 8000 + len(launches[phase])
    launches[phase].append((list(command), pid))
    if len(launches[phase]) == 1:
        control.write_text("2\n", encoding="utf-8")
    return Child(pid)

def source_host():
    current["pass"] += 1
    return reference.MemorySample(8192, 8192, 0.0, 0, 0, 0.0)

source_root = root / "source"
source_results = source_root / "results"
source_specs = [
    reference.JobSpec(
        job_id=f"dynamic-{index}", phase="probe", family="ADP", case_name="toy",
        repeat_index=0, depth=1, width=index + 1, seed=4,
        results_dir=str(source_results / str(index)), command=("python", f"worker-{index}.py"),
    )
    for index in range(2)
]
control.write_text("1\n", encoding="utf-8")
sys.argv = [
    "opf_massive_suite_runner", "--mode", "massive",
    "--results-dir", str(source_results), "--run-root", str(source_root / "run"),
    "--cpu-only", "--concurrency", "1", "--concurrency-file", str(control),
    "--repeat-count", "1", "--max-retries", "1",
    "--pressure-poll-interval-sec", "0.1", "--post-launch-sample-delay-sec", "0",
]
source_args = reference.parse_args()
reference.parse_args = lambda: source_args
reference.validate_args = lambda _args: None
reference.configure_root_logging = lambda: None
reference.set_logging_context_provider = lambda _provider: None
reference.bootstrap_runtime = lambda *_args: None
reference.enable_optional_gpu_accelerators = lambda: None
reference.seed_run_root_from_source = lambda *_args: {"files_copied": 0, "files_skipped": 0}
reference.build_suite_jobs = lambda *_args: source_specs
reference.sample_host_memory = source_host
reference.sample_gpu_memory = lambda *_args: None
reference.current_pids = lambda _runtimes: []
reference.sample_non_python_host_memory = lambda **_kwargs: (0.0, [])
reference.sample_non_python_gpu_memory = lambda *_args, **_kwargs: (0.0, [], [])
reference.sample_runtime_child_memory = lambda *_args, **_kwargs: (0.0, 0.0, [])
reference.make_child_env = lambda *_args, **_kwargs: {"PATH": "/usr/bin"}
reference.subprocess.Popen = fake_popen
reference._terminate_process_tree = lambda *_args, **_kwargs: None
reference.time.sleep = lambda _seconds: None
reference.REPO_ROOT = repository
reference.main()
source_report = json.loads((source_root / "run" / "final_report.json").read_text())
assert source_report["completed_jobs"] == 2
assert len(launches["source"]) == 2

current.update(phase="generic", **{"pass": 0})
control.write_text("1\n", encoding="utf-8")
generic_root = root / "generic"
generic_results = generic_root / "results"
generic_specs = []
for index in range(2):
    identity = RunIdentity(
        provider="toy", suite="suite", task_id=str(index), split_repeat=0,
        fold=0, sample=0, suite_repeat=1, model_route=f"route-{index}", seed=4,
        split_sha256="split", run_signature="generic-signature",
    )
    generic_specs.append(RunSpec(
        identity, DatasetTaskRef("toy", "suite", str(index)), "probe",
        ModelRoute("ADP-W-All", init_depth=1, init_width=index + 1),
        "split.npz", str(generic_results / str(index)), {},
        ("python", f"worker-{index}.py"),
    ))
plan = SuitePlan(
    "generic-signature", {"run_signature": "generic-signature", "jobs": []},
    tuple(generic_specs),
)
def generic_host():
    current["pass"] += 1
    return HostMemorySample(8192, 8192, 0.0, 0, 0, 0.0)
provider = CallbackResourceProvider(
    host_sampler=generic_host, gpu_sampler=lambda _index: None,
    tree_sampler=lambda pid: frozenset({pid}),
    child_sampler=lambda _pid, **_kwargs: ProcessMemorySample(),
    external_host_sampler=lambda _excluded: ProcessMemorySample(),
    external_gpu_sampler=lambda _index, _excluded: ProcessMemorySample(),
)
processes.launcher_child_env = lambda *_args, **_kwargs: {"PATH": "/usr/bin"}
coordinator = SuiteCoordinator(
    plan,
    RunnerSettings(
        cpu_only=True, concurrency=1, concurrency_file=str(control), max_retries=1,
        pressure_poll_seconds=0.1, post_launch_sample_delay_seconds=0.0,
        run_hardness_screen=False,
    ),
    run_root=generic_root / "run", results_dir=generic_results,
    repository_root=repository, resource_provider=provider,
)
coordinator.sleep = lambda _seconds: None
generic_report = coordinator.run()
assert generic_report["completed"] == 2
assert len(launches["generic"]) == 2
assert [item[0] for item in launches["source"]] == [
    item[0] for item in launches["generic"]
] == [["python", "worker-0.py"], ["python", "worker-1.py"]]
print("source main and generic coordinator reread changed concurrency between passes")
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository), str(tmp_path)],
        cwd=repository, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout
    assert result.stdout.strip().endswith(
        "source main and generic coordinator reread changed concurrency between passes"
    )


def test_source_main_and_generic_retry_requeue_to_fifo_tail(tmp_path):
    """A failed first route runs again only after the already-queued route."""
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

repository = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
reference = importlib.import_module("utils.opf_massive_suite_runner")
from simple_adp.runner import processes
from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.datasets import DatasetTaskRef
from simple_adp.runner.models import ModelRoute
from simple_adp.runner.plan import RunIdentity, RunSpec, SuitePlan
from simple_adp.runner.resources import CallbackResourceProvider, HostMemorySample, ProcessMemorySample
from simple_adp.runner.scheduler import SuiteCoordinator
from simple_adp.runner.state import JobStateStore

current = {"phase": "source", "launch": 0}
commands = {"source": [], "generic": []}
outcomes = (9, 0, 0)

class Child:
    def __init__(self, pid, return_code):
        self.pid, self.return_code = pid, return_code
    def poll(self):
        return self.return_code

def fake_popen(command, **kwargs):
    index = current["launch"]
    current["launch"] += 1
    return_code = outcomes[index]
    commands[current["phase"]].append(list(command))
    if return_code:
        kwargs["stdout"].write("ordinary nonzero exit\n")
        kwargs["stdout"].flush()
    return Child(9000 + index, return_code)

source_root = root / "source"
source_results = source_root / "results"
source_specs = [
    reference.JobSpec(
        job_id=f"route-{name}", phase="probe", family="ADP", case_name="toy",
        repeat_index=0, depth=1, width=4, seed=index + 1,
        results_dir=str(source_results / name), command=("python", f"worker-{name}.py"),
    )
    for index, name in enumerate(("A", "B"))
]
sys.argv = [
    "opf_massive_suite_runner", "--mode", "massive",
    "--results-dir", str(source_results), "--run-root", str(source_root / "run"),
    "--cpu-only", "--concurrency", "2", "--repeat-count", "1", "--max-retries", "1",
    "--pressure-poll-interval-sec", "0.1", "--post-launch-sample-delay-sec", "0",
]
source_args = reference.parse_args()
reference.parse_args = lambda: source_args
reference.validate_args = lambda _args: None
reference.configure_root_logging = lambda: None
reference.set_logging_context_provider = lambda _provider: None
reference.bootstrap_runtime = lambda *_args: None
reference.enable_optional_gpu_accelerators = lambda: None
reference.seed_run_root_from_source = lambda *_args: {"files_copied": 0, "files_skipped": 0}
reference.build_suite_jobs = lambda *_args: source_specs
reference.sample_host_memory = lambda: reference.MemorySample(8192, 8192, 0.0, 0, 0, 0.0)
reference.sample_gpu_memory = lambda *_args: None
reference.current_pids = lambda _runtimes: []
reference.sample_non_python_host_memory = lambda **_kwargs: (0.0, [])
reference.sample_non_python_gpu_memory = lambda *_args, **_kwargs: (0.0, [], [])
reference.sample_runtime_child_memory = lambda *_args, **_kwargs: (0.0, 0.0, [])
reference.make_child_env = lambda *_args, **_kwargs: {"PATH": "/usr/bin"}
reference.subprocess.Popen = fake_popen
reference._terminate_process_tree = lambda *_args, **_kwargs: None
reference.time.sleep = lambda _seconds: None
reference.REPO_ROOT = repository
reference.main()
source_report = json.loads((source_root / "run" / "final_report.json").read_text())
source_states = [
    reference.load_job_state(reference.job_directory(source_root / "run", spec) / "job_state.json")
    for spec in source_specs
]
assert [command[1] for command in commands["source"]] == [
    "worker-A.py", "worker-B.py", "worker-A.py",
]

current.update(phase="generic", launch=0)
generic_root = root / "generic"
generic_results = generic_root / "results"
generic_specs = []
for index, name in enumerate(("A", "B")):
    identity = RunIdentity(
        provider="toy", suite="suite", task_id=name, split_repeat=0, fold=index,
        sample=0, suite_repeat=1, model_route=name, seed=index + 1,
        split_sha256="split", run_signature="generic-signature",
    )
    generic_specs.append(RunSpec(
        identity, DatasetTaskRef("toy", "suite", name), "probe",
        ModelRoute("ADP-W-All", init_depth=1, init_width=4), "split.npz",
        str(generic_results / name), {}, ("python", f"worker-{name}.py"),
    ))
plan = SuitePlan(
    "generic-signature", {"run_signature": "generic-signature", "jobs": []},
    tuple(generic_specs),
)
provider = CallbackResourceProvider(
    host_sampler=lambda: HostMemorySample(8192, 8192, 0.0, 0, 0, 0.0),
    gpu_sampler=lambda _index: None,
    tree_sampler=lambda pid: frozenset({pid}),
    child_sampler=lambda _pid, **_kwargs: ProcessMemorySample(),
    external_host_sampler=lambda _excluded: ProcessMemorySample(),
    external_gpu_sampler=lambda _index, _excluded: ProcessMemorySample(),
)
processes.launcher_child_env = lambda *_args, **_kwargs: {"PATH": "/usr/bin"}
coordinator = SuiteCoordinator(
    plan,
    RunnerSettings(
        cpu_only=True, concurrency=2, max_retries=1, pressure_poll_seconds=0.1,
        post_launch_sample_delay_seconds=0.0, run_hardness_screen=False,
    ),
    run_root=generic_root / "run", results_dir=generic_results,
    repository_root=repository, resource_provider=provider,
)
coordinator.sleep = lambda _seconds: None
generic_report = coordinator.run()
store = coordinator.state_store
generic_states = [store.load(spec.identity.run_id) for spec in generic_specs]
assert [command[1] for command in commands["generic"]] == [
    "worker-A.py", "worker-B.py", "worker-A.py",
]

assert source_report["completed_jobs"] == generic_report["completed"] == 2
assert source_report["failed_attempts"] == generic_report["failed_attempts"] == 1
assert source_report["retried_jobs"] == generic_report["retries"] == 1
assert [state["status"] for state in source_states] == [
    state["status"] for state in generic_states
] == ["completed", "completed"]
assert [state["attempt"] for state in source_states] == [
    state["attempt"] for state in generic_states
] == [2, 1]
assert [state["retry_count"] for state in source_states] == [
    state["retry_count"] for state in generic_states
] == [1, 0]
print("source main and generic coordinator matched retry-to-FIFO-tail trajectory")
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository), str(tmp_path)],
        cwd=repository, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout
    assert result.stdout.strip().endswith(
        "source main and generic coordinator matched retry-to-FIFO-tail trajectory"
    )


def test_source_and_generic_mixed_device_latest_child_sampling_match(tmp_path):
    """Compare the copied source selector and generic snapshot on the ledger quirk."""
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

repository = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
reference = importlib.import_module("utils.opf_massive_suite_runner")
from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.datasets import DatasetTaskRef
from simple_adp.runner.models import ModelRoute
from simple_adp.runner.plan import RunIdentity, RunSpec
from simple_adp.runner.resources import (
    HostMemorySample, ProcessMemorySample, GPUMemorySample,
)
from simple_adp.runner.scheduler import SuiteCoordinator
from simple_adp.runner.state import JobRuntime as GenericRuntime
import simple_adp.runner.scheduler as scheduler_module

def source_runtime(name, pid, mode, started):
    spec = reference.JobSpec(
        job_id=name, phase="probe", family="ADP", case_name="toy",
        repeat_index=0, depth=1, width=4, seed=2,
        results_dir=str(root / name), command=("python", "worker.py"),
    )
    runtime = reference.JobRuntime(spec, root / name / "state.json", root / name / "job.log")
    runtime.proc = SimpleNamespace(pid=pid, poll=lambda: None)
    runtime.device_mode, runtime.started_at = mode, started
    runtime.attempt = 1
    return runtime

source_gpu = source_runtime("gpu-job", 11, "gpu", "1")
source_cpu = source_runtime("cpu-job", 22, "cpu", "2")
source_memory = {"gpu-job": (100.0, 900.0, []), "cpu-job": (200.0, 0.0, [])}
reference.sample_runtime_child_memory = lambda runtime, *_args: source_memory[runtime.spec.job_id]
source_latest = reference._latest_active_runtime([source_gpu, source_cpu])
source_latest_gpu = reference._latest_active_runtime([source_gpu, source_cpu], device_mode="gpu")
source_sample = reference.sample_runtime_child_memory(source_latest, 0)
assert source_latest.spec.job_id == "cpu-job"
assert source_latest_gpu.spec.job_id == "gpu-job"

memory_by_pid = {
    11: ProcessMemorySample(100.0, 900.0, frozenset({11})),
    22: ProcessMemorySample(200.0, 0.0, frozenset({22})),
}
scheduler_module.process_tree_pids = lambda pid: frozenset({pid})
scheduler_module.sample_child_memory = lambda pid, **_kwargs: memory_by_pid[pid]
scheduler_module.sample_external_host_memory = lambda _pids: ProcessMemorySample()
scheduler_module.sample_external_gpu_memory = lambda *_args: ProcessMemorySample()

def generic_runtime(name, pid, mode, started):
    identity = RunIdentity(
        provider="toy", suite="suite", task_id=name, split_repeat=0, fold=0,
        sample=0, suite_repeat=1, model_route=name, seed=2,
        split_sha256="split", run_signature="signature",
    )
    spec = RunSpec(
        identity, DatasetTaskRef("toy", "suite", name), "probe",
        ModelRoute("ADP-W-All", init_depth=1, init_width=4), "split.npz",
        str(root / name), {}, ("python", "worker.py"),
    )
    runtime = GenericRuntime(spec, device_mode=mode, started_at=float(started), attempt=1)
    runtime.pid = pid
    runtime.process = SimpleNamespace(poll=lambda: None)
    return runtime

generic_gpu = generic_runtime("gpu-job", 11, "gpu", "1")
generic_cpu = generic_runtime("cpu-job", 22, "cpu", "2")
coordinator = SuiteCoordinator(
    SimpleNamespace(), RunnerSettings(child_gpu_gate_buffer_mib=100),
    run_root=root / "run", results_dir=root / "results", repository_root=repository,
    host_sampler=lambda: HostMemorySample(8192, 4096, 50, 0, 0, 0),
    gpu_sampler=lambda _index: GPUMemorySample(1000, 500, 500, 50),
)
snapshot, latest_all, latest_gpu = coordinator._sample_snapshot(
    [generic_gpu, generic_cpu], [], 10,
)
source_latest_host, source_latest_vram, _ = source_sample
assert latest_all is generic_cpu and latest_gpu is generic_gpu
assert snapshot.latest_child_rss_mib == source_latest_host == 200.0
assert snapshot.latest_child_vram_mib == source_latest_vram == 0.0
assert coordinator.last_resource_snapshot["latest_child_job_id"] == generic_cpu.job_id
assert generic_cpu.spec.identity.task_id == source_latest.spec.job_id
assert coordinator.last_resource_snapshot["latest_child_gpu_remaining_mib"] == 500.0
print("source and generic mixed-device latest-child sampling match")
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository), str(tmp_path)],
        cwd=repository, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout
    assert result.stdout.strip().endswith(
        "source and generic mixed-device latest-child sampling match"
    )


def test_source_main_and_generic_coordinator_match_uncapped_launch_error_spin(tmp_path):
    """Compare immediate uncapped launch-error retries until interruption."""
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import sys
from pathlib import Path

repository = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
reference = importlib.import_module("utils.opf_massive_suite_runner")
from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.datasets import DatasetTaskRef
from simple_adp.runner.models import ModelRoute
from simple_adp.runner.plan import RunIdentity, RunSpec, SuitePlan
from simple_adp.runner.resources import (
    CallbackResourceProvider, HostMemorySample, ProcessMemorySample,
)
from simple_adp.runner.scheduler import SuiteCoordinator

source_attempts = []
source_sleeps = []
source_root = root / "source"
source_results = source_root / "results"
source_spec = reference.JobSpec(
    job_id="uncapped-launch-error", phase="probe", family="ADP", case_name="toy",
    repeat_index=0, depth=1, width=2, seed=11,
    results_dir=str(source_results / "one"), command=("python", "worker.py"),
)
def source_launch(_args, _runtime, *_args2, **_kwargs):
    source_attempts.append(1)
    if len(source_attempts) == 4:
        raise KeyboardInterrupt
    raise OSError("synthetic spawn error")

sys.argv = [
    "opf_massive_suite_runner", "--mode", "massive",
    "--results-dir", str(source_results), "--run-root", str(source_root / "run"),
    "--cpu-only", "--concurrency", "1", "--max-retries", "0",
    "--repeat-count", "1", "--pressure-poll-interval-sec", "0.5",
    "--post-launch-sample-delay-sec", "0",
]
source_args = reference.parse_args()
reference.parse_args = lambda: source_args
reference.validate_args = lambda _args: None
reference.configure_root_logging = lambda: None
reference.set_logging_context_provider = lambda _provider: None
reference.bootstrap_runtime = lambda *_args: None
reference.enable_optional_gpu_accelerators = lambda: None
reference.seed_run_root_from_source = lambda *_args: {"files_copied": 0, "files_skipped": 0}
reference.build_suite_jobs = lambda *_args: [source_spec]
reference.launch_job = source_launch
reference.sample_host_memory = lambda: reference.MemorySample(8192, 8192, 0.0, 0, 0, 0.0)
reference.sample_gpu_memory = lambda *_args: None
reference.current_pids = lambda _runtimes: []
reference.sample_non_python_host_memory = lambda **_kwargs: (0.0, [])
reference.sample_non_python_gpu_memory = lambda *_args, **_kwargs: (0.0, [], [])
reference.sample_runtime_child_memory = lambda *_args, **_kwargs: (0.0, 0.0, {})
reference.time.sleep = lambda seconds: source_sleeps.append(seconds)
reference.REPO_ROOT = repository
try:
    reference.main()
except KeyboardInterrupt:
    pass

current = {"phase": "generic"}
generic_attempts = []
generic_sleeps = []
generic_root = root / "generic"
generic_results = generic_root / "results"
identity = RunIdentity(
    provider="toy", suite="suite", task_id="one", split_repeat=0,
    fold=0, sample=0, suite_repeat=1, model_route="uncapped-launch-error",
    seed=11, split_sha256="split", run_signature="generic-signature",
)
generic_spec = RunSpec(
    identity, DatasetTaskRef("toy", "suite", "one"), "probe",
    ModelRoute("ADP-W-All", init_depth=1, init_width=2), "split.npz",
    str(generic_results / "one"), {}, ("python", "worker.py"),
)
plan = SuitePlan(
    "generic-signature", {"run_signature": "generic-signature", "jobs": [generic_spec.as_dict()]},
    (generic_spec,),
)
provider = CallbackResourceProvider(
    host_sampler=lambda: HostMemorySample(8192, 8192, 0.0, 0, 0, 0.0),
    gpu_sampler=lambda _index: None,
    tree_sampler=lambda pid: frozenset({pid}),
    child_sampler=lambda _pid, **_kwargs: ProcessMemorySample(),
    external_host_sampler=lambda _excluded: ProcessMemorySample(),
    external_gpu_sampler=lambda _index, _excluded: ProcessMemorySample(),
)
coordinator = SuiteCoordinator(
    plan,
    RunnerSettings(
        cpu_only=True, concurrency=1, max_retries=0,
        pressure_poll_seconds=0.5, post_launch_sample_delay_seconds=0.0,
        run_hardness_screen=False,
    ),
    run_root=generic_root / "run", results_dir=generic_results,
    repository_root=repository, resource_provider=provider,
    poll_sleep=lambda seconds: generic_sleeps.append(seconds),
)
import simple_adp.runner.scheduler as scheduler_module
def generic_launch(_runtime, **_kwargs):
    generic_attempts.append(1)
    if len(generic_attempts) == 4:
        raise KeyboardInterrupt
    raise OSError("synthetic spawn error")
scheduler_module.launch_job = generic_launch
try:
    coordinator.run()
except SystemExit as exc:
    assert exc.code == 130

assert len(source_attempts) == len(generic_attempts) == 4
assert source_sleeps == generic_sleeps == []
assert coordinator.counters.failed_attempts == coordinator.counters.retries == 3
print("source main and generic coordinator matched uncapped immediate launch-error retries")
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository), str(tmp_path)],
        cwd=repository, check=False, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout
    assert result.stdout.strip().endswith(
        "source main and generic coordinator matched uncapped immediate launch-error retries"
    )


def test_source_and_generic_final_report_fields_have_explicit_semantic_mapping(tmp_path):
    """Pin report-field equivalence while allowing documented generic fields."""
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

repository = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
reference = importlib.import_module("utils.opf_massive_suite_runner")
from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.datasets import DatasetTaskRef
from simple_adp.runner.models import ModelRoute
from simple_adp.runner.plan import RunIdentity, RunSpec, SuitePlan
from simple_adp.runner.scheduler import SuiteCoordinator

source_root = root / "source-run"
source_results = root / "source-results"
source_jobs = [SimpleNamespace(job_id="source-one")]
source = reference.render_summary(
    source_root, source_results, source_jobs,
    completed=1, failed=1, retried=3,
)
assert set(source) == {
    "run_root", "results_dir", "total_jobs", "completed_jobs",
    "failed_jobs", "retried_jobs", "timestamp",
}

identity = RunIdentity(
    provider="toy", suite="suite", task_id="task", split_repeat=0,
    fold=0, sample=0, suite_repeat=1, model_route="ADP-W-All", seed=7,
    split_sha256="split", run_signature="generic-signature",
)
spec = RunSpec(
    identity, DatasetTaskRef("toy", "suite", "task"), "probe",
    ModelRoute("ADP-W-All"), "split.npz", str(root / "generic-results" / "task"),
    {}, ("python", "worker.py"),
)
plan = SuitePlan(
    "generic-signature", {"run_signature": "generic-signature", "jobs": [spec.as_dict()]},
    (spec,),
)
coordinator = SuiteCoordinator(
    plan, RunnerSettings(run_hardness_screen=False),
    run_root=root / "generic-run", results_dir=root / "generic-results",
    repository_root=repository,
)
coordinator.counters.completed = 1
coordinator.counters.terminal_failures = 1
coordinator.counters.failed_attempts = 4
coordinator.counters.retries = 3
generic = coordinator._write_final_reports()

assert source["run_root"] == str(source_root)
assert source["results_dir"] == str(source_results)
assert source["total_jobs"] == generic["manifest_jobs"] == 1
assert source["completed_jobs"] == generic["completed"] == 1
assert source["failed_jobs"] == generic["terminal_failures"] == 1
assert source["retried_jobs"] == generic["retries"] == 3
assert set(generic) == {
    "run_root", "results_dir", "run_signature", "manifest_jobs", "completed",
    "terminal_failures", "failed_attempts", "retries", "recovered_unknown_exit",
    "timestamp", "resource_snapshot", "hardness_screening", "batches_done",
    "epochs_done", "live_batches_done", "live_epochs_done", "telemetry_jobs",
    "telemetry_dirs",
}
assert generic["failed_attempts"] == 4
assert generic["recovered_unknown_exit"] == 0
print("source and generic final report fields map by process-outcome semantics")
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository), str(tmp_path)],
        cwd=repository, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout
    assert result.stdout.strip().endswith(
        "source and generic final report fields map by process-outcome semantics"
    )


def test_source_and_generic_finalization_write_same_finish_marker(tmp_path):
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

repository = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
reference = importlib.import_module("utils.opf_massive_suite_runner")
from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.datasets import DatasetTaskRef
from simple_adp.runner.models import ModelRoute
from simple_adp.runner.plan import RunIdentity, RunSpec, SuitePlan
from simple_adp.runner.scheduler import SuiteCoordinator
from simple_adp.runner.state import JobRuntime, RunStatus

class CaptureHandle:
    def __init__(self):
        self.parts = []
        self.closed = False
    def write(self, value):
        self.parts.append(value)
    def flush(self):
        pass
    def close(self):
        self.closed = True
    def value(self):
        return "".join(self.parts)

source_root = root / "source-run"
source_spec = reference.JobSpec(
    "finish-probe", "probe", "ADP", "toy", 0, 1, 2, 7,
    str(root / "source-results"), ("python", "worker.py"),
)
source_handle = CaptureHandle()
source_runtime = reference.JobRuntime(
    source_spec,
    state_path=source_root / "jobs" / "finish-probe" / "job_state.json",
    log_path=source_root / "jobs" / "finish-probe" / "job.log",
    log_handle=source_handle,
    attempt=2,
)
reference.finalize_job(
    SimpleNamespace(), source_runtime, 7, Path("unused.log"),
    source_root / "run_progress.csv", source_root, {}, "fail", "nonzero_exit:7",
)

identity = RunIdentity(
    provider="toy", suite="suite", task_id="task", split_repeat=0,
    fold=0, sample=0, suite_repeat=1, model_route="ADP-W-All", seed=7,
    split_sha256="split", run_signature="sig",
)
generic_spec = RunSpec(
    identity, DatasetTaskRef("toy", "suite", "task"), "probe",
    ModelRoute("ADP-W-All"), "split.npz", str(root / "generic-results"),
    {}, ("python", "worker.py"),
)
generic_plan = SuitePlan("sig", {"run_signature": "sig", "jobs": []}, (generic_spec,))
coordinator = SuiteCoordinator(
    generic_plan, RunnerSettings(run_hardness_screen=False),
    run_root=root / "generic-run", results_dir=root / "generic-results",
    repository_root=repository,
)
generic_handle = CaptureHandle()
generic_runtime = JobRuntime(
    generic_spec, status=RunStatus.RUNNING, attempt=2,
    exit_code=7, last_exit_code=7, log_handle=generic_handle,
)
coordinator._finalize(
    generic_runtime, success=False, reason="nonzero_exit:7",
)
assert source_handle.closed and generic_handle.closed
source_marker = source_handle.value()
generic_marker = generic_handle.value()
assert source_marker == "=== FINISH finish-probe exit=7 reason=nonzero_exit:7 ===\n"
assert generic_marker == (
    f"=== FINISH {generic_runtime.job_id} exit=7 reason=nonzero_exit:7 ===\n"
), (source_marker, generic_marker)
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository), str(tmp_path)],
        cwd=repository, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout


def test_source_and_generic_launch_log_prefix_matches_exactly(tmp_path):
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

repository = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
reference = importlib.import_module("utils.opf_massive_suite_runner")
from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.datasets import DatasetTaskRef
from simple_adp.runner.models import ModelRoute
from simple_adp.runner.plan import RunIdentity, RunSpec, SuitePlan
from simple_adp.runner.processes import launch_job as generic_launch_job
from simple_adp.runner.state import JobRuntime, JobStateStore

class Child:
    pid = 9123
    def poll(self):
        return None

popen_observations = []
def fake_popen(command, **kwargs):
    handle = kwargs["stdout"]
    popen_observations.append((list(command), handle.tell()))
    return Child()

reference.subprocess.Popen = fake_popen
reference.make_child_env = lambda *_args, **_kwargs: {"PATH": "/usr/bin"}
reference.REPO_ROOT = repository
source_root = root / "source-run"
source_spec = reference.JobSpec(
    "launch-probe", "probe", "ADP", "toy", 0, 1, 2, 7,
    str(root / "source-results"), ("python", "worker.py"),
)
source_log = source_root / "jobs" / "launch-probe" / "job.log"
source_runtime = reference.JobRuntime(
    source_spec,
    state_path=source_root / "jobs" / "launch-probe" / "job_state.json",
    log_path=source_log,
)
context = {"z": 2, "a": 1}
reference.launch_job(
    SimpleNamespace(cpu_only=True), source_runtime, 1, "cpu", source_root,
    launch_context=context,
)
source_prefix = source_log.read_text()
source_position = popen_observations[-1][1]

generic_root = root / "generic-run"
identity = RunIdentity(
    provider="toy", suite="suite", task_id="task", split_repeat=0,
    fold=0, sample=0, suite_repeat=1, model_route="ADP-W-All", seed=7,
    split_sha256="split", run_signature="sig",
)
generic_spec = RunSpec(
    identity, DatasetTaskRef("toy", "suite", "task"), "probe",
    ModelRoute("ADP-W-All"), "split.npz", str(root / "generic-results"),
    {}, ("python", "worker.py"),
)
generic_store = JobStateStore(generic_root)
generic_runtime = JobRuntime(generic_spec, device_mode="cpu")
generic_launch_job(
    generic_runtime, settings=RunnerSettings(cpu_only=True),
    state_store=generic_store, repository_root=repository, launch_budget=1,
    device="cpu", launch_context=context,
)
generic_prefix = Path(generic_runtime.log_path).read_text()
generic_position = popen_observations[-1][1]
expected_source = (
    "\n=== LAUNCH launch-probe attempt=1 device=cpu ===\n"
    "COMMAND: python worker.py\n"
    'LAUNCH_CONTEXT: {"a": 1, "z": 2}\n'
)
expected_generic = expected_source.replace("launch-probe", generic_runtime.job_id)
assert source_prefix == expected_source, repr(source_prefix)
assert generic_prefix == expected_generic, repr(generic_prefix)
assert source_position == len(expected_source.encode())
assert generic_position == len(expected_generic.encode())
source_runtime.log_handle.close()
generic_runtime.log_handle.close()
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository), str(tmp_path)],
        cwd=repository, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout


def test_source_and_generic_process_tree_termination_order_matches(tmp_path):
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import sys
from pathlib import Path

repository = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
reference = importlib.import_module("utils.opf_massive_suite_runner")
from simple_adp.runner import processes
import subprocess

class FakePopen:
    def __init__(self, events):
        self.pid = 10
        self.events = events
    def poll(self):
        return None
    def terminate(self):
        self.events.append(("popen_terminate", 10))
    def wait(self, *, timeout):
        self.events.append(("popen_wait", timeout))

class Node:
    def __init__(self, pid, events, children=()):
        self.pid = pid
        self.events = events
        self.descendants = list(children)
    def children(self, *, recursive):
        self.events.append(("children", self.pid, recursive))
        return self.descendants
    def terminate(self):
        self.events.append(("terminate", self.pid))
    def kill(self):
        self.events.append(("kill", self.pid))

class FakePsutil:
    def __init__(self, events, survivor_rounds):
        self.events = events
        self.root = Node(10, events)
        children = [Node(11, events), Node(12, events)]
        self.root.descendants = children
        self.nodes = {10: self.root, 11: children[0], 12: children[1]}
        self.survivor_rounds = iter(survivor_rounds)
    def Process(self, pid):
        return self.nodes[pid]
    def wait_procs(self, items, *, timeout):
        self.events.append(("wait_procs", [item.pid for item in items], timeout))
        pids = next(self.survivor_rounds)
        return [], [self.nodes[pid] for pid in pids]

original_popen = subprocess.Popen
subprocess.Popen = FakePopen
try:
    for survivor_rounds in (((),), ((12, 10), ()), ((12, 10), (10,))):
        traces = []
        for terminate in (reference._terminate_process_tree, processes.terminate_process_tree):
            events = []
            psutil = FakePsutil(events, survivor_rounds)
            if terminate is reference._terminate_process_tree:
                reference.psutil = psutil
            else:
                processes.psutil = psutil
            child = FakePopen(events)
            terminate(child, 0.25) if terminate is reference._terminate_process_tree else terminate(child, timeout_seconds=0.25)
            traces.append(events)
        assert traces[0] == traces[1], (survivor_rounds, traces)
        assert traces[0][:4] == [
            ("children", 10, True), ("terminate", 12),
            ("terminate", 11), ("terminate", 10),
        ]
        assert traces[0][-1] == ("popen_wait", 0.1 if survivor_rounds == ((),) else 0.25)
finally:
    subprocess.Popen = original_popen
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository)],
        cwd=repository, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout


def test_source_and_generic_launch_failure_close_log_without_extra_flush(tmp_path):
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

repository = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
reference = importlib.import_module("utils.opf_massive_suite_runner")
from simple_adp.runner import processes
from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.datasets import DatasetTaskRef
from simple_adp.runner.models import ModelRoute
from simple_adp.runner.plan import RunIdentity, RunSpec, SuitePlan
from simple_adp.runner.state import JobRuntime, JobStateStore
import subprocess

class CaptureHandle:
    def __init__(self):
        self.events = []
        self.parts = []
    def write(self, value):
        self.events.append("write")
        self.parts.append(value)
    def flush(self):
        self.events.append("flush")
    def close(self):
        self.events.append("close")
    def tell(self):
        return sum(map(len, self.parts))

handles = []
original_path_open = Path.open
def capture_log_open(path, *args, **kwargs):
    if path.name == "job.log":
        handle = CaptureHandle()
        handles.append(handle)
        return handle
    return original_path_open(path, *args, **kwargs)
Path.open = capture_log_open
original_popen = subprocess.Popen
def failing_popen(_command, **kwargs):
    kwargs["stdout"].events.append("popen")
    raise OSError("injected launch failure")
subprocess.Popen = failing_popen
reference.make_child_env = lambda *_args, **_kwargs: {"PATH": "/usr/bin"}
reference.REPO_ROOT = repository

try:
    source_root = root / "source-run"
    source_spec = reference.JobSpec(
        "launch-error", "probe", "ADP", "toy", 0, 1, 2, 7,
        str(root / "source-results"), ("python", "worker.py"),
    )
    source_runtime = reference.JobRuntime(
        source_spec,
        state_path=source_root / "jobs" / "launch-error" / "job_state.json",
        log_path=source_root / "jobs" / "launch-error" / "job.log",
    )
    try:
        reference.launch_job(SimpleNamespace(cpu_only=True), source_runtime, 1, "cpu", source_root)
    except OSError as exc:
        assert str(exc) == "injected launch failure"
    else:
        raise AssertionError("source launch failure was not propagated")
    source_handle = handles[-1]
    assert source_runtime.log_handle is None

    identity = RunIdentity(
        provider="toy", suite="suite", task_id="task", split_repeat=0,
        fold=0, sample=0, suite_repeat=1, model_route="ADP-W-All", seed=7,
        split_sha256="split", run_signature="sig",
    )
    spec = RunSpec(
        identity, DatasetTaskRef("toy", "suite", "task"), "probe",
        ModelRoute("ADP-W-All"), "split.npz", str(root / "generic-results"),
        {}, ("python", "worker.py"),
    )
    store = JobStateStore(root / "generic-run")
    runtime = JobRuntime(spec, device_mode="cpu")
    processes.launcher_child_env = lambda *_args, **_kwargs: {"PATH": "/usr/bin"}
    try:
        processes.launch_job(
            runtime, settings=RunnerSettings(cpu_only=True), state_store=store,
            repository_root=repository, launch_budget=1, device="cpu",
        )
    except OSError as exc:
        assert str(exc) == "injected launch failure"
    else:
        raise AssertionError("generic launch failure was not propagated")
    generic_handle = handles[-1]
    assert runtime.log_handle is None

    assert source_handle.events[-2:] == generic_handle.events[-2:] == ["popen", "close"], (
        source_handle.events, generic_handle.events,
    )
    source_text = "".join(source_handle.parts).replace(
        "launch-error", runtime.job_id,
    )
    assert source_text == "".join(generic_handle.parts)
finally:
    Path.open = original_path_open
    subprocess.Popen = original_popen
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository), str(tmp_path)],
        cwd=repository, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout


def test_source_run_root_seeding_skips_reserved_name_subtrees_like_reference(tmp_path):
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import sys
from pathlib import Path

repository = Path(sys.argv[1]).resolve()
case_root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
reference = importlib.import_module("utils.opf_massive_suite_runner")
from simple_adp.runner.state import seed_run_root_from_source

source = case_root / "source"
reserved = source / "manifest.json"
reserved.mkdir(parents=True)
(reserved / "nested.json").write_text("reserved subtree", encoding="utf-8")
(source / "ordinary.json").write_text("ordinary file", encoding="utf-8")
source_destination = case_root / "reference-destination"
generic_destination = case_root / "generic-destination"
for destination in (source_destination, generic_destination):
    destination.mkdir()
    # Named artifacts are copied first; an existing target is counted skipped.
    # Recursive traversal must also skip the entire source subtree with this
    # reserved first path component.
    (destination / "manifest.json").write_text("keep", encoding="utf-8")

expected = reference.seed_run_root_from_source(source_destination, str(source))
actual = seed_run_root_from_source(source, generic_destination)
assert actual == {
    "copied": expected["files_copied"],
    "skipped": expected["files_skipped"],
}
expected_files = sorted(
    path.relative_to(source_destination).as_posix()
    for path in source_destination.rglob("*") if path.is_file()
)
actual_files = sorted(
    path.relative_to(generic_destination).as_posix()
    for path in generic_destination.rglob("*") if path.is_file()
)
assert actual_files == expected_files == ["manifest.json", "ordinary.json"]
assert (generic_destination / "manifest.json").read_text(encoding="utf-8") == "keep"

# If the reserved source path is a directory and there is no destination with
# that name, the named-first copy attempts copy2(directory, target) and raises.
# Both implementations must raise the same exception after creating only the
# destination parent directory.
error_source = case_root / "error-source"
(error_source / "manifest.json").mkdir(parents=True)
(error_source / "manifest.json" / "nested.json").write_text("nested", encoding="utf-8")
source_error_destination = case_root / "reference-error-destination"
generic_error_destination = case_root / "generic-error-destination"

def capture(operation):
    try:
        operation()
    except Exception as error:
        return type(error), str(error)
    return None

source_error = capture(
    lambda: reference.seed_run_root_from_source(source_error_destination, str(error_source))
)
generic_error = capture(
    lambda: seed_run_root_from_source(error_source, generic_error_destination)
)
assert generic_error == source_error
assert source_error is not None and source_error[0] is IsADirectoryError
assert source_error_destination.exists() == generic_error_destination.exists()
assert not (source_error_destination / "manifest.json" / "nested.json").exists()
assert not (generic_error_destination / "manifest.json" / "nested.json").exists()

# The source helper creates destination parents only as part of an actual
# copy. Missing, empty-directory, and regular-file roots therefore leave the
# destination absent; the generic helper must match those side effects too.
for index, source_path in enumerate((case_root / "missing", case_root / "empty", case_root / "source-file")):
    reference_destination = case_root / f"reference-empty-{index}"
    adapter_destination = case_root / f"adapter-empty-{index}"
    if source_path.name == "empty":
        source_path.mkdir()
    elif source_path.name == "source-file":
        source_path.write_text("not a directory", encoding="utf-8")
    expected = reference.seed_run_root_from_source(reference_destination, str(source_path))
    actual = seed_run_root_from_source(source_path, adapter_destination)
    assert actual == {
        "copied": expected["files_copied"],
        "skipped": expected["files_skipped"],
    }
    assert reference_destination.exists() == adapter_destination.exists()
    assert not reference_destination.exists()
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository), str(tmp_path)],
        cwd=repository, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout


def test_source_and_generic_multipart_bundle_round_trip_match(tmp_path):
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

repository = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
source = importlib.import_module("utils.opf_resume_bundle")
import simple_adp.runner.bundles as generic

# Reduce the chunk size in both implementations so this contract exercises
# the actual multi-part branch without creating a 64 MiB test artifact.
source.BUNDLE_PART_SIZE_BYTES = 128
generic.PART_SIZE_BYTES = 128
source_root = root / "source-repo"
generic_root = root / "generic-repo"
source_results = source_root / "results"
generic_results = generic_root / "results"
payload = os.urandom(4096)
for result in (source_results, generic_results):
    (result / "nested").mkdir(parents=True)
    (result / "state.bin").write_bytes(payload)
    (result / "nested" / "telemetry.csv").write_text("kind\\nbatch\\n", encoding="utf-8")

source_bundle = root / "source.tar.gz"
generic_bundle = root / "generic.tar.gz"
assert source.export_results_bundle(source_results, source_bundle) == source_bundle
assert generic.export_results_bundle(generic_results, generic_root, generic_bundle) == generic_bundle

def part_paths(bundle):
    return sorted(bundle.parent.glob(bundle.name + ".part*"))

source_parts = part_paths(source_bundle)
generic_parts = part_paths(generic_bundle)
assert len(source_parts) > 1
assert len(generic_parts) == len(source_parts)
assert not source_bundle.exists() and not generic_bundle.exists()
assert all(len(path.read_bytes()) <= 128 for path in source_parts + generic_parts)
assert [path.name.split(".part")[-1] for path in source_parts] == [path.name.split(".part")[-1] for path in generic_parts]
assert not list(source_root.rglob("*.tmp"))
assert not list(generic_root.rglob("*.tmp"))

def stable_manifest(bundle):
    path = bundle.with_suffix("").with_suffix(".json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {key: payload[key] for key in ("kind", "bundle_mode", "part_size_bytes", "source_exists", "source_size_bytes")}

assert stable_manifest(source_bundle) == stable_manifest(generic_bundle)
for result in (source_results, generic_results):
    shutil.rmtree(result)
assert source.restore_results_bundle(source_results, source_bundle)
assert generic.restore_results_bundle(generic_results, generic_root, generic_bundle)
assert (source_results / "state.bin").read_bytes() == (generic_results / "state.bin").read_bytes() == payload
assert not (source_results / "nested" / "telemetry.csv").exists()
assert not (generic_results / "nested" / "telemetry.csv").exists()
assert not source_bundle.with_suffix(source_bundle.suffix + ".restore").exists()
assert not generic_bundle.with_suffix(generic_bundle.suffix + ".restore").exists()

# A corrupt multipart archive leaves the reassembly artifact behind in both
# implementations, and creates the destination parent before reading parts.
bad_source_target = source_root / "failed" / "results"
bad_generic_target = generic_root / "failed" / "results"
bad_source_bundle = root / "bad-source.tar.gz"
bad_generic_bundle = root / "bad-generic.tar.gz"
bad_source_bundle.with_suffix(bad_source_bundle.suffix + ".part000").write_bytes(b"bad archive")
bad_generic_bundle.with_suffix(bad_generic_bundle.suffix + ".part000").write_bytes(b"bad archive")
errors = []
for restore, target, bundle, repo in (
    (source.restore_results_bundle, bad_source_target, bad_source_bundle, None),
    (generic.restore_results_bundle, bad_generic_target, bad_generic_bundle, generic_root),
):
    try:
        if repo is None:
            restore(target, bundle)
        else:
            restore(target, repo, bundle)
    except Exception as exc:
        errors.append(type(exc))
    else:
        raise AssertionError("corrupt multipart archive unexpectedly restored")
assert errors[0] is errors[1]
for target, bundle in ((bad_source_target, bad_source_bundle), (bad_generic_target, bad_generic_bundle)):
    assert target.parent.is_dir()
    assert bundle.with_suffix(bundle.suffix + ".restore").exists()
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository), str(tmp_path)],
        cwd=repository, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout


def test_source_and_generic_bundle_manifest_failure_preserves_old_archive(tmp_path):
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import sys
from pathlib import Path

repository = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
source = importlib.import_module("utils.opf_resume_bundle")
from simple_adp.runner.bundles import export_results_bundle

outcomes = []
for name, export, source_signature in (
    ("source", source.export_results_bundle, True),
    ("generic", export_results_bundle, False),
):
    repository_root = root / f"{name}-repo"
    results = repository_root / "results"
    results.mkdir(parents=True)
    (results / "state.json").write_text("{}", encoding="utf-8")
    bundle = root / f"{name}.tar.gz"
    bundle.write_bytes(b"previous archive")
    part = bundle.with_suffix(bundle.suffix + ".part000")
    part.write_bytes(b"previous part")
    manifest = bundle.with_suffix("").with_suffix(".json")
    manifest.mkdir()
    try:
        if source_signature:
            export(results, bundle)
        else:
            export(results, repository_root, bundle)
    except Exception as exc:
        outcomes.append(type(exc))
    else:
        raise AssertionError("manifest write to a directory unexpectedly succeeded")
    assert bundle.read_bytes() == b"previous archive"
    assert part.read_bytes() == b"previous part"
    assert manifest.is_dir()

assert outcomes[0] is outcomes[1]
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository), str(tmp_path)],
        cwd=repository, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout


def test_source_and_generic_bundle_tar_failure_cleanup_order_matches(tmp_path):
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import sys
from pathlib import Path

repository = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
source = importlib.import_module("utils.opf_resume_bundle")
import simple_adp.runner.bundles as generic

snapshots = []
for name, module, source_signature in (
    ("source", source, True), ("generic", generic, False),
):
    repository_root = root / f"{name}-repo"
    results = repository_root / "results"
    results.mkdir(parents=True)
    (results / "state.json").write_text("{}", encoding="utf-8")
    bundle = root / f"{name}.tar.gz"
    bundle.write_bytes(b"previous archive")
    part = bundle.with_suffix(bundle.suffix + ".part000")
    part.write_bytes(b"previous part")
    original_open = module.tarfile.open
    def fail_tar_open(*args, **kwargs):
        raise OSError("injected tar creation failure")
    module.tarfile.open = fail_tar_open
    try:
        try:
            if source_signature:
                module.export_results_bundle(results, bundle)
            else:
                module.export_results_bundle(results, repository_root, bundle)
        except OSError as exc:
            assert str(exc) == "injected tar creation failure"
        else:
            raise AssertionError("tar creation failure was not propagated")
    finally:
        module.tarfile.open = original_open
    manifest = bundle.with_suffix("").with_suffix(".json")
    snapshots.append({
        "archive": bundle.exists(), "old_part": part.exists(),
        "manifest": manifest.is_file(),
        "temporary": bundle.with_suffix(bundle.suffix + ".tmp").exists(),
        "parts": sorted(path.name for path in bundle.parent.glob(bundle.name + ".part*")),
    })
assert snapshots[0] == snapshots[1], snapshots
assert snapshots[0] == {
    "archive": False, "old_part": False, "manifest": True,
    "temporary": False, "parts": [],
}
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository), str(tmp_path)],
        cwd=repository, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout


def test_source_and_generic_bundle_part_write_failure_residue_matches(tmp_path):
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import sys
from pathlib import Path

repository = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
source = importlib.import_module("utils.opf_resume_bundle")
import simple_adp.runner.bundles as generic
source.BUNDLE_PART_SIZE_BYTES = 8
generic.PART_SIZE_BYTES = 8

snapshots = []
for name, module, source_signature in (
    ("source", source, True), ("generic", generic, False),
):
    repository_root = root / f"{name}-repo"
    results = repository_root / "results"
    results.mkdir(parents=True)
    (results / "state.json").write_bytes(b"x" * 512)
    bundle = root / f"{name}.tar.gz"
    bundle.write_bytes(b"previous archive")
    bundle.with_suffix(bundle.suffix + ".part000").write_bytes(b"previous part")
    original_open = Path.open
    def fail_second_part(path, mode="r", *args, **kwargs):
        if path.name.endswith(".part001") and "w" in mode:
            raise OSError("injected second-part write failure")
        return original_open(path, mode, *args, **kwargs)
    Path.open = fail_second_part
    try:
        try:
            if source_signature:
                module.export_results_bundle(results, bundle)
            else:
                module.export_results_bundle(results, repository_root, bundle)
        except OSError as exc:
            assert str(exc) == "injected second-part write failure"
        else:
            raise AssertionError("second part write failure was not propagated")
    finally:
        Path.open = original_open
    snapshots.append({
        "archive": bundle.exists(),
        "temporary_exists": bundle.with_suffix(bundle.suffix + ".tmp").exists(),
        "parts": [(path.name.removeprefix(bundle.name), path.stat().st_size)
                  for path in sorted(bundle.parent.glob(bundle.name + ".part*"))],
    })
assert snapshots[0] == snapshots[1], snapshots
assert snapshots[0]["archive"] is False
assert snapshots[0]["temporary_exists"] is True
assert snapshots[0]["parts"] == [(".part000", 8)]
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository), str(tmp_path)],
        cwd=repository, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout


def test_source_and_generic_sharded_telemetry_rehydration_match(tmp_path):
    repository = Path(__file__).parents[1]
    script = r'''
import hashlib
import importlib
import json
import sys
from pathlib import Path

repository = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
source = importlib.import_module("utils.opf_resume_bundle")
import simple_adp.runner.bundles as generic

source_root = root / "source-root"
generic_root = root / "generic-root"
source_root.mkdir()
generic_root.mkdir()
source.REPO_ROOT = source_root

def rehydrate_source_tree(base, rehydrate, generic_mode):
    run = base / "run"
    telemetry = run / "telemetry"
    telemetry.mkdir(parents=True)
    target = run / "telemetry.csv"
    payload = b"kind,value\\nbatch,1\\nepoch,2\\n"
    (telemetry / "part000").write_bytes(payload[:12])
    (telemetry / "part001").write_bytes(payload[12:])
    manifest = telemetry / "metrics.csv.manifest.json"
    manifest.write_text(json.dumps({
        "original_path": str(target),
        "parts": ["part000", "part001"],
        "sha256": hashlib.sha256(payload).hexdigest(),
    }), encoding="utf-8")
    count = rehydrate(telemetry, base) if generic_mode else rehydrate(telemetry)
    assert count == 1
    assert target.read_bytes() == payload
    assert (rehydrate(telemetry, base) if generic_mode else rehydrate(telemetry)) == 0

    # An existing target wins before missing shard validation.
    (telemetry / "existing.csv.manifest.json").write_text(json.dumps({
        "original_path": str(target), "parts": ["absent.part"],
    }), encoding="utf-8")
    assert (rehydrate(telemetry, base) if generic_mode else rehydrate(telemetry)) == 0

def error_case(base, rehydrate, generic_mode, case, manifest_payload, files):
    run = base / case
    telemetry = run / "telemetry"
    telemetry.mkdir(parents=True)
    target = run / "metrics.csv"
    for name, data in files.items():
        path = (telemetry / name).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    payload = dict(manifest_payload)
    payload["original_path"] = str(target)
    manifest = telemetry / f"{case}.csv.manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    try:
        rehydrate(telemetry, base) if generic_mode else rehydrate(telemetry)
    except Exception as exc:
        error_type = type(exc)
    else:
        raise AssertionError(f"{case} unexpectedly rehydrated")
    return error_type, target.with_suffix(target.suffix + ".rehydrate").exists()

rehydrate_source_tree(source_root, source.rehydrate_sharded_telemetry, False)
rehydrate_source_tree(generic_root, generic.rehydrate_sharded_telemetry, True)

missing = {"parts": ["missing.part"]}
source_missing = error_case(source_root, source.rehydrate_sharded_telemetry, False, "missing", missing, {})
generic_missing = error_case(generic_root, generic.rehydrate_sharded_telemetry, True, "missing", missing, {})
assert source_missing == generic_missing
assert source_missing[0] is FileNotFoundError and source_missing[1]

checksum_payload = b"wrong content"
checksum = {"parts": ["part000"], "sha256": "0" * 64}
source_checksum = error_case(source_root, source.rehydrate_sharded_telemetry, False, "checksum", checksum, {"part000": checksum_payload})
generic_checksum = error_case(generic_root, generic.rehydrate_sharded_telemetry, True, "checksum", checksum, {"part000": checksum_payload})
assert source_checksum == generic_checksum
assert source_checksum[0] is ValueError and not source_checksum[1]

escape = {"parts": ["../../escape.part"]}
source_escape = error_case(source_root, source.rehydrate_sharded_telemetry, False, "escape", escape, {})
generic_escape = error_case(generic_root, generic.rehydrate_sharded_telemetry, True, "escape", escape, {})
assert source_escape == generic_escape
assert source_escape[0] is ValueError and source_escape[1]
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository), str(tmp_path)],
        cwd=repository, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout


def test_source_and_generic_standalone_wrapper_relaunch_exited_paused_child(tmp_path):
    repository = Path(__file__).parents[1]
    script = r'''
import contextlib
import importlib
import io
import sys
from pathlib import Path
from types import SimpleNamespace

repository = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
source = importlib.import_module("utils.pressure_aware_wrapper")
from simple_adp.runner import pressure_wrapper as generic
from simple_adp.runner.config import StandalonePressureSettings
from simple_adp.runner.resources import MIB

def execute(source_mode):
    events = []
    children = []
    memory_samples = 0

    class FakeChild:
        def __init__(self, index):
            self.pid = 7000 + index
            self.index = index
            self.done = False
            self.returncode = 0
        def poll(self):
            if self.index == 0 and self.done:
                return 1
            return None if self.index == 0 else 0
        def wait(self, timeout=None):
            return 1 if self.index == 0 else 0

    def popen(command, **kwargs):
        child = FakeChild(len(children))
        children.append(child)
        events.append(("launch", tuple(command)))
        return child

    def virtual_memory():
        nonlocal memory_samples
        memory_samples += 1
        if memory_samples == 2 and children:
            children[0].done = True
        return SimpleNamespace(percent=91.0 if memory_samples == 1 else 80.0, used=(1000 if memory_samples == 1 else 900) * MIB)

    psutil_module = source.psutil if source_mode else generic.psutil
    old = {
        "virtual_memory": psutil_module.virtual_memory,
        "swap_memory": psutil_module.swap_memory,
        "process": psutil_module.Process,
        "process_iter": psutil_module.process_iter,
    }
    module = source if source_mode else generic
    old_popen = module.subprocess.Popen
    old_check_output = module.subprocess.check_output
    old_sleep = module.time.sleep
    try:
        psutil_module.virtual_memory = virtual_memory
        psutil_module.swap_memory = lambda: SimpleNamespace(percent=70.0)
        psutil_module.Process = lambda pid: SimpleNamespace(pid=pid)
        psutil_module.process_iter = lambda _attrs=None: []
        module.subprocess.Popen = popen
        module.subprocess.check_output = lambda *args, **kwargs: b""
        module.time.sleep = lambda _seconds: None
        output = io.StringIO()
        if source_mode:
            old_install = source._install_signal_handlers
            old_pause = source._pause_active_child_tree
            old_resume = source._resume_active_child_tree
            source._install_signal_handlers = lambda: None
            source._ACTIVE_CHILD_PROCESS = None
            def pause_source():
                events.append(("pause", source._ACTIVE_CHILD_PROCESS.pid))
            def resume_source():
                child = source._ACTIVE_CHILD_PROCESS
                if child is not None and child.poll() is None:
                    events.append(("resume", child.pid))
            source._pause_active_child_tree = pause_source
            source._resume_active_child_tree = resume_source
            old_argv = sys.argv
            sys.argv = ["pressure_aware_wrapper.py", "--", "probe"]
            try:
                with contextlib.redirect_stdout(output):
                    try:
                        source.main()
                    except SystemExit as exc:
                        result = int(exc.code)
            finally:
                sys.argv = old_argv
                source._install_signal_handlers = old_install
                source._pause_active_child_tree = old_pause
                source._resume_active_child_tree = old_resume
                source._ACTIVE_CHILD_PROCESS = None
        else:
            old_pause = generic._pause_child
            generic._pause_child = lambda child, paused: events.append(("pause" if paused else "resume", child.pid))
            wrapper = generic.PressureAwareWrapper(
                ["probe"], StandalonePressureSettings(), sleeper=lambda _seconds: None,
            )
            with contextlib.redirect_stdout(output):
                result = wrapper.run()
            generic._pause_child = old_pause
    finally:
        psutil_module.virtual_memory = old["virtual_memory"]
        psutil_module.swap_memory = old["swap_memory"]
        psutil_module.Process = old["process"]
        psutil_module.process_iter = old["process_iter"]
        module.subprocess.Popen = old_popen
        module.subprocess.check_output = old_check_output
        module.time.sleep = old_sleep
    return result, events, output.getvalue()

source_result = execute(True)
generic_result = execute(False)
assert source_result[0] == generic_result[0] == 0, (source_result[0], generic_result[0])
assert source_result[1] == generic_result[1] == [("launch", ("probe",)), ("pause", 7000), ("launch", ("probe",))], (source_result[1], generic_result[1])
assert source_result[2] == generic_result[2], (source_result[2], generic_result[2])
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository)],
        cwd=repository, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout


def test_outer_watchdog_nested_signal_order_matches_executed_shell_source(monkeypatch):
    import os
    import signal

    if os.name != "posix":
        return

    repository = Path(__file__).parents[1]
    shell_script = r'''
source "$1"
sleep() { command sleep 0.01; }
# The source defines its process-tree helper inside this wrapper. Let a
# successful child return so the nested helper is installed without invoking
# the watchdog's unrelated long-running master-monitor branch.
_run_with_watchdog true
phase=1
pgrep() {
    local parent="$2"
    case "$phase:$parent" in
        1:10) printf '11\n13\n' ;;
        2:10) printf '11\n13\n14\n' ;;
        1:11|2:11) printf '12\n' ;;
        *) return 1 ;;
    esac
}
kill() { printf '%s:%s\n' "$1" "$2"; }
_watchdog_kill_process_tree 10 TERM
phase=2
_watchdog_kill_process_tree 10 KILL
'''
    source = subprocess.run(
        ["bash", "-c", shell_script, "watchdog-source",
         str(repository / "references/opf_adp_runner_suite/_orphan_watchdog.sh")],
        cwd=repository, check=True, capture_output=True, text=True,
    )
    source_order = source.stdout.splitlines()

    from simple_adp.runner import processes

    events = []
    late = None

    class Process:
        def __init__(self, pid, children=()):
            self.pid = pid
            self.children_list = list(children)
            self.scan_count = 0

        def children(self, recursive=False):
            assert recursive is False
            self.scan_count += 1
            if self is root:
                return [branch, sibling] if self.scan_count == 1 else [branch, sibling, late]
            return self.children_list

        def send_signal(self, sig):
            signal_name = signal.Signals(sig).name.removeprefix("SIG")
            events.append(f"-{signal_name}:{self.pid}")

    leaf = Process(12)
    branch = Process(11, [leaf])
    sibling = Process(13)
    late = Process(14)
    root = Process(10)
    monkeypatch.setattr(processes.psutil, "Process", lambda _pid: root)
    monkeypatch.setattr(processes.time, "sleep", lambda _seconds: None)
    processes.terminate_process_tree_after_grace(10, grace_seconds=0.5)

    assert source_order == events



def test_source_main_and_generic_drop_active_route_with_missing_process_handle(tmp_path):
    """Compare the source loop's handle-less active-runtime edge directly."""
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import json
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

repository = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
reference = importlib.import_module("utils.opf_massive_suite_runner")
from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.datasets import DatasetTaskRef
from simple_adp.runner.models import ModelRoute
from simple_adp.runner.plan import RunIdentity, RunSpec, SuitePlan
from simple_adp.runner.resources import (
    CallbackResourceProvider, HostMemorySample, ProcessMemorySample,
)
from simple_adp.runner.scheduler import SuiteCoordinator
from simple_adp.runner.state import JobRuntime, RunStatus

source_root = root / "source"
source_results = source_root / "results"
source_spec = reference.JobSpec(
    job_id="handle-disappears", phase="probe", family="ADP", case_name="toy",
    repeat_index=0, depth=1, width=2, seed=5,
    results_dir=str(source_results / "one"), command=("python", "worker.py"),
)
source_job_dir = reference.job_directory(source_root / "run", source_spec)
source_job_dir.mkdir(parents=True, exist_ok=True)
(source_job_dir / "job_state.json").write_text(
    json.dumps({"status": "running", "last_exit_code": None}), encoding="utf-8",
)
class HandlelessSourceRuntime(reference.JobRuntime):
    @property
    def status(self):
        return "running"
source_runtime = HandlelessSourceRuntime(
    source_spec, source_job_dir / "job_state.json", source_job_dir / "job.log",
    proc=None, attempt=1, started_at="20261023_120000",
)
sys.argv = [
    "opf_massive_suite_runner", "--mode", "massive", "--results-dir",
    str(source_results), "--run-root", str(source_root / "run"), "--cpu-only",
    "--concurrency", "1", "--repeat-count", "1", "--pressure-poll-interval-sec",
    "0.1", "--post-launch-sample-delay-sec", "0",
]
source_args = reference.parse_args()
reference.parse_args = lambda: source_args
reference.validate_args = lambda _args: None
reference.configure_root_logging = lambda: None
reference.set_logging_context_provider = lambda _provider: None
reference.bootstrap_runtime = lambda *_args: None
reference.enable_optional_gpu_accelerators = lambda: None
reference.seed_run_root_from_source = lambda *_args: {"files_copied": 0, "files_skipped": 0}
reference.build_suite_jobs = lambda *_args: [source_spec]
reference.load_existing_runtime = lambda *_args: source_runtime
reference.sample_host_memory = lambda: reference.MemorySample(8192, 8192, 0.0, 0, 0, 0.0)
reference.sample_gpu_memory = lambda *_args: None
reference.current_pids = lambda _runtimes: []
reference.sample_non_python_host_memory = lambda **_kwargs: (0.0, [])
reference.sample_non_python_gpu_memory = lambda *_args, **_kwargs: (0.0, [], [])
reference.sample_runtime_child_memory = lambda *_args, **_kwargs: (0.0, 0.0, {})
reference.time.sleep = lambda _seconds: None
reference.REPO_ROOT = repository
reference.main()
source_report = json.loads((source_root / "run" / "final_report.json").read_text())
source_state = reference.load_job_state(source_job_dir / "job_state.json")

generic_root = root / "generic"
generic_results = generic_root / "results"
identity = RunIdentity(
    provider="toy", suite="suite", task_id="one", split_repeat=0,
    fold=0, sample=0, suite_repeat=1, model_route="ADP-W-All", seed=5,
    split_sha256="split", run_signature="generic-signature",
)
generic_spec = RunSpec(
    identity, DatasetTaskRef("toy", "suite", "one"), "probe",
    ModelRoute("ADP-W-All", init_depth=1, init_width=2), "split.npz",
    str(generic_results / "one"), {}, ("python", "worker.py"),
)
plan = SuitePlan(
    "generic-signature", {"run_signature": "generic-signature", "jobs": [generic_spec.as_dict()]},
    (generic_spec,),
)
provider = CallbackResourceProvider(
    host_sampler=lambda: HostMemorySample(8192, 8192, 0.0, 0, 0, 0.0),
    gpu_sampler=lambda _index: None,
    tree_sampler=lambda pid: frozenset({pid}),
    child_sampler=lambda _pid, **_kwargs: ProcessMemorySample(),
    external_host_sampler=lambda _excluded: ProcessMemorySample(),
    external_gpu_sampler=lambda _index, _excluded: ProcessMemorySample(),
)
generic_runtime = JobRuntime(
    generic_spec, status=RunStatus.RUNNING, attempt=1, started_at=1.0,
    device_mode="cpu", process=None,
)
coordinator = SuiteCoordinator(
    plan,
    RunnerSettings(
        cpu_only=True, concurrency=1, pressure_poll_seconds=0.1,
        post_launch_sample_delay_seconds=0.0, run_hardness_screen=False,
    ),
    run_root=generic_root / "run", results_dir=generic_results,
    repository_root=repository, resource_provider=provider,
)
coordinator.state_store.save(generic_runtime)
coordinator._build_runtimes = lambda: ([generic_runtime], [], deque())
generic_report = coordinator.run()
generic_state = coordinator.state_store.load(identity.run_id)

assert source_state["status"] == generic_state["status"] == "running"
assert source_state.get("last_exit_code") == generic_state["last_exit_code"] is None
assert source_report["completed_jobs"] == generic_report["completed"] == 0
assert source_report["failed_attempts"] == generic_report["failed_attempts"] == 0
assert source_report["retried_jobs"] == generic_report["retries"] == 0
print("source and generic loops both dropped the active route without inventing an exit")
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository), str(tmp_path)],
        cwd=repository, check=False, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout
    assert result.stdout.strip().endswith(
        "source and generic loops both dropped the active route without inventing an exit"
    )


def test_source_and_generic_process_tree_termination_trace_match(tmp_path):
    """Compare central termination ordering and escalation with the copied source."""
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

repository = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
source = importlib.import_module("utils.opf_massive_suite_runner")
from simple_adp.runner import processes

def run(source_mode):
    events = []
    class Node:
        def __init__(self, name, children=()):
            self.name, self.children_list = name, list(children)
        def children(self, recursive=False):
            assert recursive is True
            return [branch, leaf, sibling]
        def terminate(self):
            events.append("TERM:" + self.name)
        def kill(self):
            events.append("KILL:" + self.name)
    leaf = Node("leaf")
    branch = Node("branch")
    sibling = Node("sibling")
    root = Node("root", [branch, leaf, sibling])
    class FakePopen:
        pid = 4567
        def poll(self):
            return None
        def wait(self, timeout=None):
            events.append(("ROOT_WAIT", timeout))
    process_handle = FakePopen()
    prior_process = processes.subprocess.Popen
    processes.subprocess.Popen = FakePopen
    prior_psutil_process = source.psutil.Process
    prior_psutil_wait = source.psutil.wait_procs
    source.psutil.Process = lambda _pid: root
    waits = iter(([leaf, sibling], [leaf]))
    def wait_procs(items, timeout):
        events.append(("WAIT", [item.name for item in items], timeout))
        return [], next(waits)
    source.psutil.wait_procs = wait_procs
    try:
        if source_mode:
            source._terminate_process_tree(process_handle, timeout_sec=2.5)
        else:
            processes.terminate_process_tree(process_handle, timeout_seconds=2.5)
    finally:
        source.psutil.Process = prior_psutil_process
        source.psutil.wait_procs = prior_psutil_wait
        processes.subprocess.Popen = prior_process
    return events

source_trace = run(True)
generic_trace = run(False)
assert source_trace == generic_trace, (source_trace, generic_trace)
assert source_trace == [
    "TERM:sibling", "TERM:leaf", "TERM:branch", "TERM:root",
    ("WAIT", ["root", "branch", "leaf", "sibling"], 2.5),
    "KILL:leaf", "KILL:sibling",
    ("WAIT", ["leaf", "sibling"], 2.5),
    "KILL:leaf", ("ROOT_WAIT", 2.5),
], source_trace
print("central source and generic process-tree termination traces match")
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository)],
        cwd=repository, check=False, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout


def test_source_and_generic_process_tree_termination_failure_boundaries_match(tmp_path):
    """Compare early-return and exception boundaries for central tree cleanup."""
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import sys
from pathlib import Path

repository = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
source = importlib.import_module("utils.opf_massive_suite_runner")
from simple_adp.runner import processes

def execute(source_mode, case):
    events = []
    class Parent:
        def children(self, recursive=False):
            events.append(("CHILDREN", recursive))
            if case == "enumeration-error":
                raise RuntimeError("tree enumeration failed")
            return []
        def terminate(self):
            events.append("PARENT_TERM")
        def kill(self):
            events.append("PARENT_KILL")
    parent = Parent()
    class FakePopen:
        pid = 7890
        def poll(self):
            return 0 if case == "already-exited" else None
        def terminate(self):
            events.append("HANDLE_TERM")
        def wait(self, timeout=None):
            events.append(("HANDLE_WAIT", timeout))
    handle = FakePopen()
    prior_popen = processes.subprocess.Popen
    prior_process = source.psutil.Process
    processes.subprocess.Popen = FakePopen
    if case == "root-open-error":
        def open_root(_pid):
            raise RuntimeError("root lookup failed")
    else:
        open_root = lambda _pid: parent
    source.psutil.Process = open_root
    try:
        try:
            if source_mode:
                source._terminate_process_tree(handle, timeout_sec=1.25)
            else:
                processes.terminate_process_tree(handle, timeout_seconds=1.25)
        except RuntimeError as exc:
            events.append(("RAISED", str(exc)))
    finally:
        source.psutil.Process = prior_process
        processes.subprocess.Popen = prior_popen
    return events

for case, expected in (
    ("already-exited", []),
    ("root-open-error", ["HANDLE_TERM"]),
    ("enumeration-error", [("CHILDREN", True), ("RAISED", "tree enumeration failed")]),
):
    source_trace = execute(True, case)
    generic_trace = execute(False, case)
    assert source_trace == generic_trace == expected, (case, source_trace, generic_trace)
print("central source and generic termination failure boundaries match")
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository)],
        cwd=repository, check=False, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout


def test_source_and_generic_tree_cleanup_swallow_signal_failures_and_continue(tmp_path):
    """Best-effort TERM/KILL errors must not skip later cleanup steps."""
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import sys
from pathlib import Path

repository = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
source = importlib.import_module("utils.opf_massive_suite_runner")
from simple_adp.runner import processes

def execute(source_mode):
    events = []
    class Node:
        def __init__(self, name):
            self.name = name
        def children(self, recursive=False):
            assert recursive is True
            return [child]
        def terminate(self):
            events.append("TERM:" + self.name)
            raise RuntimeError("term denied")
        def kill(self):
            events.append("KILL:" + self.name)
            raise RuntimeError("kill denied")
    root = Node("root")
    child = Node("child")
    class FakePopen:
        pid = 5678
        def poll(self):
            return None
        def terminate(self):
            events.append("HANDLE_TERM")
        def wait(self, timeout=None):
            events.append(("ROOT_WAIT", timeout))
    handle = FakePopen()
    prior_popen = processes.subprocess.Popen
    prior_process = source.psutil.Process
    prior_wait = source.psutil.wait_procs
    rounds = iter(([root, child], []))
    def wait_procs(items, timeout):
        events.append(("WAIT", [item.name for item in items], timeout))
        return [], next(rounds)
    processes.subprocess.Popen = FakePopen
    source.psutil.Process = lambda _pid: root
    source.psutil.wait_procs = wait_procs
    try:
        if source_mode:
            source._terminate_process_tree(handle, timeout_sec=1.75)
        else:
            processes.terminate_process_tree(handle, timeout_seconds=1.75)
    finally:
        processes.subprocess.Popen = prior_popen
        source.psutil.Process = prior_process
        source.psutil.wait_procs = prior_wait
    return events

source_trace = execute(True)
generic_trace = execute(False)
assert source_trace == generic_trace, (source_trace, generic_trace)
assert source_trace == [
    "TERM:child", "TERM:root",
    ("WAIT", ["root", "child"], 1.75),
    "KILL:root", "KILL:child",
    ("WAIT", ["root", "child"], 1.75),
    ("ROOT_WAIT", 1.75),
], source_trace
print("central source and generic cleanup continue after signal failures")
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository)],
        cwd=repository, check=False, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout


def test_source_and_generic_reap_success_before_hard_pressure_victim_selection(tmp_path):
    """A successful exit on the pressure sample is reaped before victim selection."""
    repository = Path(__file__).parents[1]
    script = r'''
import csv
import importlib
import json
import sys
from pathlib import Path

repository = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
reference = importlib.import_module("utils.opf_massive_suite_runner")
from simple_adp.runner import processes
from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.datasets import DatasetTaskRef
from simple_adp.runner.models import ModelRoute
from simple_adp.runner.plan import RunIdentity, RunSpec, SuitePlan
from simple_adp.runner.resources import (
    CallbackResourceProvider, HostMemorySample, ProcessMemorySample,
)
from simple_adp.runner.scheduler import SuiteCoordinator

sample = {"phase": "source", "count": 0}
hard_pressure_samples = []
class Child:
    pid = 7777
    def poll(self):
        return 0 if sample["count"] >= 2 else None

def host_sample():
    sample["count"] += 1
    available = 100.0 if sample["count"] >= 2 else 8192.0
    if available < 1024.0:
        hard_pressure_samples.append(sample["phase"])
    if sample["phase"] == "source":
        return reference.MemorySample(8192.0, available, 0.0, 0, 0, 0.0)
    return HostMemorySample(8192.0, available, 100.0, 0.0, 0.0, 0.0)

def fake_popen(_command, **_kwargs):
    return Child()

source_root = root / "source"
source_results = source_root / "results"
source_spec = reference.JobSpec(
    job_id="complete-on-pressure", phase="probe", family="ADP", case_name="toy",
    repeat_index=0, depth=1, width=1, seed=3,
    results_dir=str(source_results / "one"), command=("python", "worker.py"),
)
sys.argv = [
    "opf_massive_suite_runner", "--mode", "massive", "--results-dir",
    str(source_results), "--run-root", str(source_root / "run"), "--cpu-only",
    "--concurrency", "1", "--repeat-count", "1", "--host-ram-stop-avail-mib",
    "1024", "--pressure-poll-interval-sec", "0.1", "--post-launch-sample-delay-sec", "0",
]
source_args = reference.parse_args()
reference.parse_args = lambda: source_args
reference.validate_args = lambda _args: None
reference.configure_root_logging = lambda: None
reference.set_logging_context_provider = lambda _provider: None
reference.bootstrap_runtime = lambda *_args: None
reference.enable_optional_gpu_accelerators = lambda: None
reference.seed_run_root_from_source = lambda *_args: {"files_copied": 0, "files_skipped": 0}
reference.build_suite_jobs = lambda *_args: [source_spec]
reference.sample_host_memory = host_sample
reference.sample_gpu_memory = lambda *_args: None
reference.current_pids = lambda _runtimes: []
reference.sample_non_python_host_memory = lambda **_kwargs: (0.0, [])
reference.sample_non_python_gpu_memory = lambda *_args, **_kwargs: (0.0, [], [])
reference.sample_runtime_child_memory = lambda *_args, **_kwargs: (0.0, 0.0, {})
reference.make_child_env = lambda *_args, **_kwargs: {"PATH": "/usr/bin"}
prior_popen = reference.subprocess.Popen
reference.subprocess.Popen = fake_popen
reference.time.sleep = lambda _seconds: None
reference.REPO_ROOT = repository
try:
    reference.main()
finally:
    reference.subprocess.Popen = prior_popen
source_report = json.loads((source_root / "run" / "final_report.json").read_text())
source_jobs = list(csv.DictReader((source_root / "run" / "task_summary.csv").open(newline="")))
source_events = list(csv.DictReader((source_root / "run" / "run_progress.csv").open(newline="")))
source_state = reference.load_job_state(
    reference.job_directory(source_root / "run", source_spec) / "job_state.json",
)

generic_root = root / "generic"
generic_results = generic_root / "results"
identity = RunIdentity(
    provider="toy", suite="suite", task_id="one", split_repeat=0, fold=0,
    sample=0, suite_repeat=1, model_route="ADP-W-All", seed=3,
    split_sha256="split", run_signature="generic-signature",
)
generic_spec = RunSpec(
    identity, DatasetTaskRef("toy", "suite", "one"), "probe",
    ModelRoute("ADP-W-All"), "split.npz", str(generic_results / "one"), {},
    ("python", "worker.py"),
)
plan = SuitePlan(
    "generic-signature", {"run_signature": "generic-signature", "jobs": [generic_spec.as_dict()]},
    (generic_spec,),
)
sample.update(phase="generic", count=0)
provider = CallbackResourceProvider(
    host_sampler=host_sample, gpu_sampler=lambda _index: None,
    tree_sampler=lambda pid: frozenset({pid}),
    child_sampler=lambda _pid, **_kwargs: ProcessMemorySample(),
    external_host_sampler=lambda _excluded: ProcessMemorySample(),
    external_gpu_sampler=lambda _index, _excluded: ProcessMemorySample(),
)
coordinator = SuiteCoordinator(
    plan, RunnerSettings(cpu_only=True, concurrency=1, host_ram_stop_avail_mib=1024.0,
        pressure_poll_seconds=0.1, post_launch_sample_delay_seconds=0.0,
        run_hardness_screen=False),
    run_root=generic_root / "run", results_dir=generic_results,
    repository_root=repository, resource_provider=provider,
)
coordinator.sleep = lambda _seconds: None
prior_generic_popen = processes.subprocess.Popen
processes.subprocess.Popen = fake_popen
try:
    generic_report = coordinator.run()
finally:
    processes.subprocess.Popen = prior_generic_popen
generic_jobs = list(csv.DictReader((generic_root / "run" / "task_summary.csv").open(newline="")))
generic_events = list(csv.DictReader((generic_root / "run" / "events.csv").open(newline="")))
generic_state = coordinator.state_store.load(identity.run_id)

assert source_report["completed_jobs"] == generic_report["completed"] == 1
assert source_jobs[0]["status"] == generic_jobs[0]["status"] == "completed"
assert source_state["pause_count"] == generic_state["pause_count"] == 0
assert not any(row.get("event") == "pause" for row in source_events)
assert not any(row.get("event") == "pause" for row in generic_events)
assert set(hard_pressure_samples) == {"source", "generic"}
print("source and generic loops reap success before hard-pressure victim selection")
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository), str(tmp_path)],
        cwd=repository, check=False, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout


def test_source_and_generic_standalone_tree_termination_match_escalation_paths(tmp_path):
    """Compare standalone cleanup for empty, surviving, and missing-root paths."""
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import sys
from pathlib import Path

repository = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
source = importlib.import_module("utils.pressure_aware_wrapper")
from simple_adp.runner import processes

def execute(source_mode, survivor_rounds, lookup_error=False):
    events = []
    class Node:
        def __init__(self, name):
            self.name = name
        def children(self, *, recursive):
            assert recursive is True
            events.append(("CHILDREN", self.name))
            return [child]
        def terminate(self):
            events.append("TERM:" + self.name)
        def kill(self):
            events.append("KILL:" + self.name)
    root = Node("root")
    child = Node("child")
    nodes = {"root": root, "child": child}
    class FakePopen:
        pid = 6789
        def terminate(self):
            events.append("HANDLE_TERM")
        def wait(self, *, timeout):
            events.append(("HANDLE_WAIT", timeout))
    handle = FakePopen()
    prior_popen = processes.subprocess.Popen
    prior_process = source.psutil.Process
    prior_wait = source.psutil.wait_procs
    waits = iter(survivor_rounds)
    def open_process(_pid):
        if lookup_error:
            raise RuntimeError("lookup failed")
        return root
    def wait_procs(items, timeout):
        events.append(("WAIT", [item.name for item in items], timeout))
        return [], [nodes[name] for name in next(waits)]
    processes.subprocess.Popen = FakePopen
    source.psutil.Process = open_process
    source.psutil.wait_procs = wait_procs
    try:
        if source_mode:
            source._ACTIVE_CHILD_PROCESS = handle
            source._terminate_active_child_tree(timeout_sec=1.75)
        else:
            processes.terminate_standalone_pressure_tree(handle, timeout_seconds=1.75)
    finally:
        processes.subprocess.Popen = prior_popen
        source.psutil.Process = prior_process
        source.psutil.wait_procs = prior_wait
        source._ACTIVE_CHILD_PROCESS = None
    return events

empty_expected = [
    ("CHILDREN", "root"), "TERM:child", "TERM:root",
    ("WAIT", ["root", "child"], 1.75),
    ("WAIT", [], 1.75), ("HANDLE_WAIT", 1.75),
]
survivor_expected = [
    ("CHILDREN", "root"), "TERM:child", "TERM:root",
    ("WAIT", ["root", "child"], 1.75),
    "KILL:child", "KILL:root",
    ("WAIT", ["child", "root"], 1.75), "KILL:root",
    ("HANDLE_WAIT", 1.75),
]
for survivor_rounds, lookup_error, expected_trace in (
    (((), ()), False, empty_expected),
    ((("child", "root"), ("root",)), False, survivor_expected),
    (((), ()), True, ["HANDLE_TERM"]),
):
    source_trace = execute(True, survivor_rounds, lookup_error)
    generic_trace = execute(False, survivor_rounds, lookup_error)
    assert source_trace == generic_trace == expected_trace, (
        lookup_error, source_trace, generic_trace,
    )
print("standalone source and generic tree cleanup traces match")
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository)],
        cwd=repository, check=False, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout


def test_source_and_generic_startup_retry_budget_and_terminal_resume_failure(tmp_path):
    """Relaunch ordinary failures but retain terminal exact-resume failures."""
    repository = Path(__file__).parents[1]
    script = r'''
import importlib
import json
import os
import sys
from pathlib import Path

repository = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
for key in ("CPU_ONLY", "TABULAR_CPU_ONLY", "TRAINING_CONTROL_CPU_ONLY",
            "TRAINING_CONTROL_BACKEND", "TABULAR_BACKEND",
            "OPF_ADP_DISABLE_GPU_ACCELERATORS", "SIMPLE_ADP_DISABLE_GPU_ACCELERATORS"):
    os.environ.pop(key, None)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, str(repository / "references" / "opf_adp_runner_suite"))
reference = importlib.import_module("utils.opf_massive_suite_runner")
from simple_adp.runner import processes
from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.datasets import DatasetTaskRef
from simple_adp.runner.models import ModelRoute
from simple_adp.runner.plan import RunIdentity, RunSpec, SuitePlan
from simple_adp.runner.resources import (
    CallbackResourceProvider, HostMemorySample, ProcessMemorySample,
)
from simple_adp.runner.scheduler import SuiteCoordinator
from simple_adp.runner.state import JobRuntime, RunStatus

phase = {"name": "source"}
launches = []
class Child:
    pid = 8123
    returncode = 0
    def poll(self):
        return 0

def fake_popen(command, **kwargs):
    launches.append((phase["name"], list(command), kwargs.get("start_new_session")))
    return Child()

def source_run():
    source_root = root / "source"
    results = source_root / "results"
    specs = [
        reference.JobSpec(
            job_id="retry-exhausted", phase="probe", family="ADP", case_name="toy",
            repeat_index=0, depth=1, width=1, seed=6,
            results_dir=str(results / "ordinary"), command=("python", "worker.py", "ordinary"),
        ),
        reference.JobSpec(
            job_id="terminal-resume", phase="probe", family="ADP", case_name="toy",
            repeat_index=0, depth=1, width=2, seed=7,
            results_dir=str(results / "terminal"), command=("python", "worker.py", "terminal"),
        ),
    ]
    for spec, reason, exit_code in (
        (specs[0], "nonzero_exit:7", 7),
        (specs[1], "inexact_checkpoint_resume", 9),
    ):
        job_dir = reference.job_directory(source_root / "run", spec)
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "job_state.json").write_text(json.dumps({
            "job_id": spec.job_id, "status": "failed", "attempt": 2,
            "retry_count": 1, "exit_code": exit_code, "last_exit_code": exit_code,
            "reason": reason, "device_mode": "cpu",
            "command": list(spec.command), "paused": False,
        }), encoding="utf-8")
    sys.argv = ["opf_massive_suite_runner", "--mode", "massive",
        "--results-dir", str(results), "--run-root", str(source_root / "run"),
        "--cpu-only", "--concurrency", "1", "--max-retries", "1",
        "--repeat-count", "1", "--pressure-poll-interval-sec", "0.1",
        "--post-launch-sample-delay-sec", "0"]
    args = reference.parse_args()
    reference.parse_args = lambda: args
    reference.validate_args = lambda _args: None
    reference.configure_root_logging = lambda: None
    reference.set_logging_context_provider = lambda _provider: None
    reference.bootstrap_runtime = lambda *_args: None
    reference.enable_optional_gpu_accelerators = lambda: None
    reference.seed_run_root_from_source = lambda *_args: {"files_copied": 0, "files_skipped": 0}
    reference.build_suite_jobs = lambda *_args: specs
    reference.sample_host_memory = lambda: reference.MemorySample(8192, 8192, 0, 0, 0, 0)
    reference.sample_gpu_memory = lambda *_args: None
    reference.current_pids = lambda _runtimes: []
    reference.sample_non_python_host_memory = lambda **_kwargs: (0.0, [])
    reference.sample_non_python_gpu_memory = lambda *_args, **_kwargs: (0.0, [], [])
    reference.sample_runtime_child_memory = lambda *_args, **_kwargs: (0.0, 0.0, {})
    reference.make_child_env = lambda *_args, **_kwargs: {"PATH": "/usr/bin"}
    reference.time.sleep = lambda _seconds: None
    reference.REPO_ROOT = repository
    prior_popen = reference.subprocess.Popen
    reference.subprocess.Popen = fake_popen
    try:
        reference.main()
    finally:
        reference.subprocess.Popen = prior_popen
    states = [reference.load_job_state(
        reference.job_directory(source_root / "run", spec) / "job_state.json",
    ) for spec in specs]
    return json.loads((source_root / "run" / "final_report.json").read_text()), states

phase["name"] = "source"
source_report, source_states = source_run()
source_launch_count = sum(1 for item in launches if item[0] == "source")

phase["name"] = "generic"
generic_root = root / "generic"
results = generic_root / "results"
specs = []
for task_id, model_id, seed, width, suffix in (
    ("retry-exhausted", "ADP-W-All", 6, 1, "ordinary"),
    ("terminal-resume", "ADP-W-All", 7, 2, "terminal"),
):
    identity = RunIdentity(
        provider="toy", suite="suite", task_id=task_id, split_repeat=0,
        fold=0, sample=0, suite_repeat=1, model_route=model_id, seed=seed,
        split_sha256="split", run_signature="generic-signature",
    )
    specs.append(RunSpec(
        identity, DatasetTaskRef("toy", "suite", task_id), "probe",
        ModelRoute(model_id, init_depth=1, init_width=width), "split.npz",
        str(results / suffix), {}, ("python", "worker.py", suffix),
    ))
plan = SuitePlan("generic-signature", {
    "run_signature": "generic-signature", "jobs": [spec.as_dict() for spec in specs],
}, tuple(specs))
provider = CallbackResourceProvider(
    host_sampler=lambda: HostMemorySample(8192, 8192, 0, 0, 0, 0),
    gpu_sampler=lambda _index: None, tree_sampler=lambda pid: frozenset({pid}),
    child_sampler=lambda _pid, **_kwargs: ProcessMemorySample(),
    external_host_sampler=lambda _excluded: ProcessMemorySample(),
    external_gpu_sampler=lambda _index, _excluded: ProcessMemorySample(),
)
coordinator = SuiteCoordinator(
    plan, RunnerSettings(cpu_only=True, concurrency=1, max_retries=1,
        pressure_poll_seconds=0.1, post_launch_sample_delay_seconds=0,
        run_hardness_screen=False), run_root=generic_root / "run",
    results_dir=results, repository_root=repository, resource_provider=provider,
)
for spec, reason, exit_code in (
    (specs[0], "nonzero_exit:7", 7),
    (specs[1], "inexact_checkpoint_resume", 9),
):
    coordinator.state_store.save(JobRuntime(
        spec, status=RunStatus.FAILED, attempt=2, retry_count=1,
        exit_code=exit_code, last_exit_code=exit_code, last_error=reason,
        device_mode="cpu",
    ))
coordinator.sleep = lambda _seconds: None
prior_popen = processes.subprocess.Popen
processes.subprocess.Popen = fake_popen
try:
    generic_report = coordinator.run()
finally:
    processes.subprocess.Popen = prior_popen
generic_states = [coordinator.state_store.load(spec.identity.run_id) for spec in specs]
generic_launch_count = sum(1 for item in launches if item[0] == "generic")

assert source_launch_count == generic_launch_count == 1
assert source_states[0]["status"] == generic_states[0]["status"] == "completed"
assert source_states[0]["attempt"] == generic_states[0]["attempt"] == 3
assert source_states[0]["retry_count"] == generic_states[0]["retry_count"] == 1
assert source_states[1]["status"] == generic_states[1]["status"] == "failed"
assert source_states[1]["reason"] == generic_states[1]["last_error"] == "inexact_checkpoint_resume"
assert source_report["completed_jobs"] == generic_report["completed"] == 1
assert source_report["failed_jobs"] == generic_report["terminal_failures"] == 1
assert source_report["retried_jobs"] == generic_report["retries"] == 0
assert not any("terminal" in item[1] for item in launches)
print("source and generic startup relaunch ordinary failure and retain exact-resume failure")
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(repository), str(tmp_path)],
        cwd=repository, check=False, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout
