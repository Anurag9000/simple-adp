from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import logging
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import shutil
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import psutil

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
VANILLA_ROOT = REPO_ROOT / "DNN" / "VANILLA"
if str(VANILLA_ROOT) not in sys.path:
    sys.path.insert(0, str(VANILLA_ROOT))

from utils.runtime_tuning import bootstrap_runtime, launcher_child_env
from utils.ml_backends import enable_optional_gpu_accelerators
from utils.logging_utils import configure_root_logging, set_logging_context_provider
from utils.opf_shared_defaults import derive_suite_seed, parse_epoch_limit
from Dyn_DNN4OPF.utils.run_defaults import RUNNER_DEFAULTS
from Dyn_DNN4OPF.utils.run_defaults import (
    DEFAULT_CASES,
    DEFAULT_DEPTH_ONLY_WIDTHS,
    DEFAULT_GLOBAL_SEED,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_WIDTH,
    DEFAULT_MAX_NEURONS,
    DEFAULT_VANILLA_DEPTHS,
    DEFAULT_VANILLA_WIDTHS,
    DEFAULT_WIDTH_ONLY_DEPTHS,
    DEFAULT_WIDTH_TO_DEPTH_WIDTHS,
)
UNBOUNDED_CONCURRENCY = 10**9
HEAD_VARIANTS = ("single_head", "two_heads", "four_heads")
HEAD_VARIANT_ALIASES = {
    "single": "single_head",
    "single_head": "single_head",
    "1": "single_head",
    "one": "single_head",
    "one_head": "single_head",
    "2": "two_heads",
    "2head": "two_heads",
    "2_head": "two_heads",
    "two": "two_heads",
    "two_head": "two_heads",
    "two_heads": "two_heads",
    "4": "four_heads",
    "4head": "four_heads",
    "4_head": "four_heads",
    "four": "four_heads",
    "four_head": "four_heads",
    "four_heads": "four_heads",
}


def _truthy(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on", "y"}:
        return True
    if lowered in {"0", "false", "no", "off", "n", ""}:
        return False
    return default

# The CLI must decide CPU/GPU admission before optional accelerator import
# hooks are installed. The controller's import must remain side-effect-light.


def _parent_cpu_masked() -> bool:
    """A child must not reverse a caller's CPU-only scheduler decision."""
    visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    return (
        os.environ.get("TRAINING_CONTROL_BACKEND", "").strip().lower() == "cpu"
        or (visibility is not None and visibility.strip() in {"", "-1"})
        or any(_truthy(os.environ.get(name)) for name in (
            "CPU_ONLY", "TRAINING_CONTROL_CPU_ONLY", "OPF_ADP_DISABLE_GPU_ACCELERATORS",
        ))
    )


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _atomic_write_text(path: Path, text: str) -> None:
    _ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))


def _json_signature(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_csv_row(path: Path, fieldnames: Sequence[str], row: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except Exception:
            pass


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _normalize_head_variants(value: Any) -> List[str]:
    raw = str(value or "").strip().lower()
    if not raw or raw == "all":
        return list(HEAD_VARIANTS)

    variants: List[str] = []
    seen: set[str] = set()
    aliases = {
        "1": "single_head",
        "1head": "single_head",
        "1_head": "single_head",
        "single": "single_head",
        "2": "two_heads",
        "2head": "two_heads",
        "2_head": "two_heads",
    }
    for part in raw.split(","):
        item = part.strip().lower()
        if not item:
            continue
        item = aliases.get(item, HEAD_VARIANT_ALIASES.get(item, item))
        if item not in HEAD_VARIANTS:
            raise SystemExit(f"Unsupported head variant: {part}")
        if item in seen:
            continue
        variants.append(item)
        seen.add(item)
    return variants or list(HEAD_VARIANTS)


def _phase_for_head_variant(family: str, head_variant: str) -> str:
    return family if head_variant == "single_head" else f"{family}__{head_variant}"


def _model_name_for_family(family: str, head_variant: str) -> str:
    if head_variant == "single_head":
        return f"{family}_single_head"
    if head_variant == "two_heads":
        return f"{family}_two_heads"
    if head_variant == "four_heads":
        return f"{family}_four_heads"

    raise SystemExit(f"Unsupported family/head_variant combination: family={family} head_variant={head_variant}")


def _parse_csv_ints(text: str) -> List[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def _parse_range_tuple(text: str) -> Tuple[int, int]:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected two comma-separated integers, got: {text}")
    return int(parts[0]), int(parts[1])


def _format_kv(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return str(value)
    return json.dumps(value, sort_keys=True)


def _proc_group_kwargs() -> Dict[str, Any]:
    if os.name == "nt":
        flag = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        return {"creationflags": flag} if flag else {}
    return {"start_new_session": True}


def _terminate_process_tree(proc: subprocess.Popen[Any], timeout_sec: float = 10.0) -> None:
    if proc.poll() is not None:
        return
    try:
        parent = psutil.Process(proc.pid)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
        return

    children = parent.children(recursive=True)
    processes = [parent, *children]
    for child in reversed(children):
        try:
            child.terminate()
        except Exception:
            pass
    try:
        parent.terminate()
    except Exception:
        pass

    _, alive = psutil.wait_procs(processes, timeout=max(0.1, float(timeout_sec)))
    if not alive:
        try:
            proc.wait(timeout=0.1)
        except Exception:
            pass
        return

    for child in alive:
        try:
            child.kill()
        except Exception:
            pass
    _, still_alive = psutil.wait_procs(alive, timeout=max(0.1, float(timeout_sec)))
    for child in still_alive:
        try:
            child.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=max(0.1, float(timeout_sec)))
    except Exception:
        pass


def _close_runtime_log_handle(runtime: "JobRuntime") -> None:
    handle = runtime.log_handle
    if handle is None:
        return
    try:
        handle.flush()
    except Exception:
        pass
    try:
        handle.close()
    except Exception:
        pass
    runtime.log_handle = None


@dataclass(frozen=True)
class MemorySample:
    total_mib: int
    available_mib: int
    used_pct: float
    swap_total_mib: int
    swap_used_mib: int
    swap_used_pct: float


@dataclass(frozen=True)
class GpuMemorySample:
    total_mib: int
    used_mib: int
    free_mib: int
    used_pct: float


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    phase: str
    family: str
    case_name: str
    repeat_index: int
    depth: int
    width: int
    seed: int
    results_dir: str
    command: Tuple[str, ...]
    device_capable: bool = True
    head_variant: str = "single_head"


@dataclass
class JobRuntime:
    spec: JobSpec
    state_path: Path
    log_path: Path
    log_handle: Optional[Any] = None
    proc: Optional[subprocess.Popen[Any]] = None
    device_mode: str = "gpu"
    attempt: int = 0
    retry_count: int = 0
    paused: bool = False
    pause_reason: str = ""
    pause_count: int = 0
    gpu_pause_count: int = 0
    host_pause_count: int = 0
    # Retained to read legacy in-memory callers; persisted flags are ignored.
    force_cpu_after_cuda_oom: bool = False
    last_exit_code: Optional[int] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    last_progress_ts: float = field(default_factory=time.time)

    def to_state(self) -> Dict[str, Any]:
        return {
            "job_id": self.spec.job_id,
            "phase": self.spec.phase,
            "family": self.spec.family,
            "head_variant": self.spec.head_variant,
            "case_name": self.spec.case_name,
            "repeat_index": self.spec.repeat_index,
            "depth": self.spec.depth,
            "width": self.spec.width,
            "seed": self.spec.seed,
            "results_dir": self.spec.results_dir,
            "command": list(self.spec.command),
            "device_capable": self.spec.device_capable,
            "device_mode": self.device_mode,
            "attempt": self.attempt,
            "retry_count": self.retry_count,
            "paused": self.paused,
            "pause_reason": self.pause_reason,
            "pause_count": self.pause_count,
            "gpu_pause_count": self.gpu_pause_count,
            "host_pause_count": self.host_pause_count,
            "force_cpu_after_cuda_oom": self.force_cpu_after_cuda_oom,
            "last_exit_code": self.last_exit_code,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "status": self.status,
            "updated_at": _now_stamp(),
        }

    @property
    def status(self) -> str:
        if self.ended_at and self.last_exit_code == 0:
            return "completed"
        if self.proc is not None and self.proc.poll() is None:
            return "running"
        if self.paused:
            return "paused"
        if self.last_exit_code is None:
            return "queued"
        return "failed" if self.last_exit_code != 0 else "completed"


@dataclass
class RecoveredProcessHandle:
    """Minimal process handle for an already-running child discovered after restart."""

    pid: int

    def poll(self) -> Optional[int]:
        return None if _is_live_pid(self.pid) else 1


def sample_host_memory() -> MemorySample:
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    total_mib = int(vm.total // (1024 * 1024))
    available_mib = int(vm.available // (1024 * 1024))
    used_pct = float(vm.percent)
    swap_total_mib = int(swap.total // (1024 * 1024))
    swap_used_mib = int(swap.used // (1024 * 1024))
    swap_used_pct = float(swap.percent)
    return MemorySample(
        total_mib=total_mib,
        available_mib=available_mib,
        used_pct=used_pct,
        swap_total_mib=swap_total_mib,
        swap_used_mib=swap_used_mib,
        swap_used_pct=swap_used_pct,
    )


def sample_gpu_memory(device_index: int = 0) -> Optional[GpuMemorySample]:
    query = [
        "nvidia-smi",
        f"--id={int(device_index)}",
        "--query-gpu=memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(query, stderr=subprocess.DEVNULL)
        row = out.decode("utf-8", errors="ignore").strip().splitlines()[0]
        total_text, used_text = [part.strip() for part in row.split(",", 1)]
        total_mib = int(float(total_text))
        used_mib = int(float(used_text))
        free_mib = max(0, total_mib - used_mib)
        used_pct = 0.0 if total_mib <= 0 else (float(used_mib) / float(total_mib)) * 100.0
        return GpuMemorySample(
            total_mib=total_mib,
            used_mib=used_mib,
            free_mib=free_mib,
            used_pct=used_pct,
        )
    except Exception:
        return None


def sample_gpu_process_memory(device_index: int = 0, pid_filter: Optional[Iterable[int]] = None) -> float:
    pid_set = {int(pid) for pid in pid_filter} if pid_filter is not None else None
    query = [
        "nvidia-smi",
        f"--id={int(device_index)}",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    total = 0.0
    try:
        out = subprocess.check_output(query, stderr=subprocess.DEVNULL)
        for line in out.decode("utf-8", errors="ignore").strip().splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 3:
                continue
            pid_text, _name, mem_text = parts[:3]
            try:
                pid = int(pid_text)
            except Exception:
                continue
            if pid_set is not None and pid not in pid_set:
                continue
            try:
                total += float(mem_text)
            except Exception:
                continue
    except Exception:
        pass
    return total


def _process_tree_pids(pid: int) -> List[int]:
    pids = [int(pid)]
    try:
        proc = psutil.Process(int(pid))
        for child in proc.children(recursive=True):
            try:
                pids.append(int(child.pid))
            except Exception:
                continue
    except Exception:
        pass
    return sorted(set(pids))


def _is_python_process_name(name: str) -> bool:
    lowered = str(name or "").strip().lower()
    return "python" in lowered or lowered.startswith("py")


def sample_non_python_host_memory(exclude_pids: Optional[Iterable[int]] = None) -> Tuple[float, List[int]]:
    exclude = {int(pid) for pid in exclude_pids} if exclude_pids is not None else set()
    total = 0.0
    pids: List[int] = []
    for proc in psutil.process_iter(attrs=["pid", "name", "cmdline", "memory_info"]):
        try:
            pid = int(proc.info.get("pid") or proc.pid)
            if pid in exclude:
                continue
            name = str(proc.info.get("name") or "")
            cmdline = " ".join(proc.info.get("cmdline") or [])
            if _is_python_process_name(name) or _is_python_process_name(cmdline):
                continue
            mem = proc.info.get("memory_info")
            rss = float(mem.rss) / (1024 * 1024) if mem is not None else float(proc.memory_info().rss) / (1024 * 1024)
            total += rss
            pids.append(pid)
        except Exception:
            continue
    return total, sorted(set(pids))


def sample_non_python_gpu_memory(device_index: int = 0, exclude_pids: Optional[Iterable[int]] = None) -> Tuple[float, List[int], List[str]]:
    exclude = {int(pid) for pid in exclude_pids} if exclude_pids is not None else set()
    query = [
        "nvidia-smi",
        f"--id={int(device_index)}",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    total = 0.0
    pids: List[int] = []
    names: List[str] = []
    try:
        out = subprocess.check_output(query, stderr=subprocess.DEVNULL)
        for line in out.decode("utf-8", errors="ignore").strip().splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 3:
                continue
            pid_text, name_text, mem_text = parts[:3]
            try:
                pid = int(pid_text)
            except Exception:
                continue
            if pid in exclude:
                continue
            if _is_python_process_name(name_text):
                continue
            try:
                total += float(mem_text)
                pids.append(pid)
                names.append(name_text)
            except Exception:
                continue
    except Exception:
        pass
    return total, sorted(set(pids)), sorted(set(names))


def sample_runtime_child_memory(runtime: JobRuntime, gpu_device_index: int = 0) -> Tuple[float, float, List[int]]:
    """Best-effort memory footprint for a live child job.

    Returns (host_rss_mib, gpu_vram_mib, pid_tree).
    """

    if runtime.proc is None or runtime.proc.poll() is not None:
        return 0.0, 0.0, []
    try:
        pid_tree = _process_tree_pids(int(runtime.proc.pid))
    except Exception:
        pid_tree = [int(runtime.proc.pid)]

    host_rss = 0.0
    for pid in pid_tree:
        try:
            host_rss += float(psutil.Process(pid).memory_info().rss) / (1024 * 1024)
        except Exception:
            continue

    gpu_vram = 0.0
    if runtime.device_mode == "gpu":
        gpu_vram = float(sample_gpu_process_memory(int(gpu_device_index), pid_filter=pid_tree))
    return host_rss, gpu_vram, pid_tree


def pressure_pause_reopened(
    *,
    pause_reason: str,
    device_mode: str,
    host_available_mib: float,
    gpu_free_mib: Optional[float],
    host_used_mib: float,
    gpu_used_mib: Optional[float],
    host_resume_avail_mib: float,
    gpu_resume_avail_mib: float,
    host_peak_used_mib: float,
    gpu_peak_used_mib: float,
    external_drop_threshold_mib: float,
    external_host_drop_mib: float = 0.0,
    external_gpu_drop_mib: float = 0.0,
) -> bool:
    """Return whether every pressure domain that paused a job is reopened."""
    reason = str(pause_reason or "").lower()
    host_pause = "host" in reason or "swap" in reason or device_mode == "cpu"
    gpu_pause = "gpu" in reason or "cuda" in reason
    if device_mode == "gpu" and not (host_pause or gpu_pause):
        gpu_pause = True

    threshold = float(external_drop_threshold_mib)
    host_ready = float(host_available_mib) >= float(host_resume_avail_mib) or (
        float(host_peak_used_mib) > 0.0
        and float(host_peak_used_mib) - float(host_used_mib) >= threshold
    ) or float(external_host_drop_mib) >= threshold
    gpu_ready = gpu_free_mib is not None and (
        float(gpu_free_mib) >= float(gpu_resume_avail_mib)
        or (
            gpu_used_mib is not None
            and float(gpu_peak_used_mib) > 0.0
            and float(gpu_peak_used_mib) - float(gpu_used_mib) >= threshold
        )
        or float(external_gpu_drop_mib) >= threshold
    )
    return (not host_pause or host_ready) and (not gpu_pause or gpu_ready)


def _last_job_attempt_log(path: Path, fallback_bytes: int = 200_000) -> str:
    """Return the current attempt's log, excluding errors from older retries."""
    marker = b"=== LAUNCH "
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            end = stream.tell()
            position = end
            carry = b""
            while position > 0:
                size = min(64 * 1024, position)
                position -= size
                stream.seek(position)
                block = stream.read(size)
                combined = block + carry
                index = combined.rfind(marker)
                if index >= 0:
                    start = position + index
                    stream.seek(start)
                    return stream.read().decode("utf-8", errors="replace")
                carry = combined[: len(marker) - 1]
            stream.seek(max(0, end - int(fallback_bytes)))
            return stream.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def is_gpu_oom_log(path: Path) -> bool:
    lowered = _last_job_attempt_log(path)[-4000:].lower()
    return (
        "cuda out of memory" in lowered
        or "cudnn_status_alloc_failed" in lowered
        or "cublas_status_alloc_failed" in lowered
        or "cublascreate(handle)" in lowered
        or "out of memory" in lowered and "cuda" in lowered
    )


def is_memory_oom_log(path: Path) -> bool:
    lowered = _last_job_attempt_log(path)[-4000:].lower()
    return (
        "out of memory" in lowered
        or "cannot allocate memory" in lowered
        or "memoryerror" in lowered
        or "killed" in lowered and "oom" in lowered
        or "cuda out of memory" in lowered
        or "cudnn_status_alloc_failed" in lowered
    )


def is_inexact_resume_log(path: Path) -> bool:
    """Recognize missing checkpoint state that a retry cannot recover."""
    lowered = _last_job_attempt_log(path).lower()
    return "exact-resume failure:" in lowered or "lacks exact-resume guarantees" in lowered


def compute_opf_params(depth: int, width: int) -> int:
    input_dim = 14 * 2
    output_dim = 14 * 4
    prev = input_dim
    total = 0
    for _ in range(int(depth)):
        total += prev * width + width
        total += 2 * width
        prev = width
    total += prev * output_dim + output_dim
    return int(total)


def binary_search_width(depth: int, target_params: int, max_width: int = 50000) -> int:
    low = 16
    high = int(max_width)
    best_w = low
    best_diff = float("inf")
    while low <= high:
        mid = (low + high) // 2
        p = compute_opf_params(depth, mid)
        diff = abs(p - target_params)
        if diff < best_diff:
            best_diff = diff
            best_w = mid
        if p < target_params:
            low = mid + 1
        else:
            high = mid - 1
    return best_w


def generate_vanilla_grid(
    depths: Sequence[int],
    min_band: int = 1,
    max_band: int = 6,
    samples: int = 3,
    max_width: int = 50000,
) -> List[Tuple[int, int]]:
    targets: List[int] = []
    for band in range(int(min_band), int(max_band)):
        low_val = 10**band
        high_val = 10 ** (band + 1)
        for i in range(int(samples)):
            ratio = i / max(1, int(samples) - 1)
            val = low_val + ratio * (high_val - low_val)
            targets.append(int(val))
    seen = set()
    configs: List[Tuple[int, int, int]] = []
    for depth in depths:
        for target in targets:
            width = binary_search_width(int(depth), int(target), max_width=max_width)
            if width > max_width:
                continue
            if (int(depth), int(width)) in seen:
                continue
            seen.add((int(depth), int(width)))
            configs.append((int(depth), int(width), compute_opf_params(int(depth), int(width))))
    return [(depth, width) for depth, width, _ in sorted(configs, key=lambda item: (item[0], item[1]))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OPF_ADP massive suite orchestrator with pressure-aware GPU-first scheduling.")
    parser.add_argument("--mode", choices=["case", "massive"], required=True)
    parser.add_argument("--results-dir", default="Results/massive_opf_suite")
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--source-run-root", default=None)
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES))
    parser.add_argument("--repeat-count", type=int, default=int(RUNNER_DEFAULTS["repeat_count"]))
    parser.add_argument(
        "--max-retries",
        type=int,
        default=int(RUNNER_DEFAULTS["max_retries"]),
        help="retries after a failed attempt; 0 retries without a fixed limit",
    )
    parser.add_argument("--surface", choices=["active", "backlog", "all"], default="all")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--scheduler", choices=["pressure_aware", "gpu_first", "fixed"], default="gpu_first")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(RUNNER_DEFAULTS["concurrency"]),
        help="Requested launch concurrency; 0 means unbounded and resource-gated only.",
    )
    parser.add_argument("--concurrency-file", default=None)
    parser.add_argument("--job-start-index", type=int, default=0)
    parser.add_argument("--job-limit", type=int, default=0)
    parser.add_argument("--max-active-jobs", type=int, default=0)
    parser.add_argument("--max-active-gpu-jobs", type=int, default=0)
    parser.add_argument("--host-ram-stop-avail-mib", type=float, default=float(RUNNER_DEFAULTS["host_ram_stop_avail_mib"]))
    parser.add_argument("--host-ram-resume-avail-mib", type=float, default=float(RUNNER_DEFAULTS["host_ram_resume_avail_mib"]))
    parser.add_argument("--gpu-stop-avail-mib", type=float, default=float(RUNNER_DEFAULTS["gpu_stop_avail_mib"]))
    parser.add_argument("--gpu-resume-avail-mib", type=float, default=float(RUNNER_DEFAULTS["gpu_resume_avail_mib"]))
    parser.add_argument("--swap-pressure-limit-pct", type=float, default=float(RUNNER_DEFAULTS["swap_pressure_limit_pct"]), help="Legacy compatibility option; swap occupancy no longer affects scheduling.")
    parser.add_argument("--swap-resume-pct", type=float, default=float(RUNNER_DEFAULTS["swap_resume_pct"]), help="Legacy compatibility option; swap occupancy no longer affects scheduling.")
    parser.add_argument("--gpu-device-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=int(RUNNER_DEFAULTS.get("seed", DEFAULT_GLOBAL_SEED)))
    parser.add_argument("--launch-host-reserve-mib", type=float, default=float(RUNNER_DEFAULTS["launch_host_reserve_mib"]))
    parser.add_argument("--launch-gpu-reserve-mib", type=float, default=float(RUNNER_DEFAULTS["launch_gpu_reserve_mib"]))
    parser.add_argument("--gpu-child-gate-close-buffer-mib", type=float, default=float(RUNNER_DEFAULTS["gpu_child_gate_close_buffer_mib"]))
    parser.add_argument("--host-child-gate-close-buffer-mib", type=float, default=float(RUNNER_DEFAULTS["host_child_gate_close_buffer_mib"]))
    parser.add_argument("--external-memory-drop-reopen-mib", type=float, default=float(RUNNER_DEFAULTS["external_memory_drop_reopen_mib"]))
    parser.add_argument("--pressure-poll-interval-sec", type=float, default=float(RUNNER_DEFAULTS["pressure_poll_interval_sec"]))
    parser.add_argument("--post-launch-sample-delay-sec", type=float, default=float(RUNNER_DEFAULTS["post_launch_sample_delay_sec"]))
    parser.add_argument("--batch-backoff-factor", type=float, default=float(RUNNER_DEFAULTS["batch_backoff_factor"]))
    parser.add_argument("--param-band", default=None)
    parser.add_argument("--legacy-architecture-grid", action="store_true")
    parser.add_argument("--cpu-only", action="store_true", default=False)
    parser.add_argument("--include-small-stl", action="store_true")
    parser.add_argument("--missing-present-task-repeats", type=int, default=int(RUNNER_DEFAULTS["missing_present_task_repeats"]))
    parser.add_argument("--prediction-repeats", type=int, default=int(RUNNER_DEFAULTS["prediction_repeats"]))
    parser.add_argument(
        "--head-variants",
        default=",".join(HEAD_VARIANTS),
        help="Comma-separated head variants to run. Use single_head,two_heads,four_heads or all.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-vanilla", dest="include_vanilla", action="store_true")
    parser.add_argument("--no-include-vanilla", dest="include_vanilla", action="store_false")
    parser.set_defaults(include_vanilla=None)
    parser.add_argument("--include-width-only", action="store_true", default=True)
    parser.add_argument("--no-include-width-only", dest="include_width_only", action="store_false")
    parser.add_argument("--include-width-to-depth", action="store_true", default=True)
    parser.add_argument("--no-include-width-to-depth", dest="include_width_to_depth", action="store_false")
    parser.add_argument("--include-depth-to-width", action="store_true", default=True)
    parser.add_argument("--no-include-depth-to-width", dest="include_depth_to_width", action="store_false")
    parser.add_argument("--include-depth-only", action="store_true", default=True)
    parser.add_argument("--no-include-depth-only", dest="include_depth_only", action="store_false")
    parser.add_argument("--vanilla-min-band", type=int, default=int(RUNNER_DEFAULTS["vanilla_min_band"]))
    parser.add_argument("--vanilla-max-band", type=int, default=int(RUNNER_DEFAULTS["vanilla_max_band"]))
    parser.add_argument("--vanilla-samples", type=int, default=int(RUNNER_DEFAULTS["vanilla_samples"]))
    parser.add_argument("--vanilla-depths", default=",".join(str(item) for item in DEFAULT_VANILLA_DEPTHS))
    parser.add_argument("--vanilla-widths", default=",".join(str(item) for item in DEFAULT_VANILLA_WIDTHS))
    parser.add_argument("--width-only-depths", default=",".join(str(item) for item in DEFAULT_WIDTH_ONLY_DEPTHS))
    parser.add_argument("--width-to-depth-widths", default=",".join(str(item) for item in DEFAULT_WIDTH_TO_DEPTH_WIDTHS))
    parser.add_argument("--depth-only-widths", default=",".join(str(item) for item in DEFAULT_DEPTH_ONLY_WIDTHS))
    parser.add_argument("--max-width", type=int, default=int(RUNNER_DEFAULTS.get("max_width", DEFAULT_MAX_WIDTH)))
    parser.add_argument("--max-depth", type=int, default=int(RUNNER_DEFAULTS.get("max_depth", DEFAULT_MAX_DEPTH)))
    parser.add_argument("--max-neurons", type=int, default=int(RUNNER_DEFAULTS.get("max_neurons", DEFAULT_MAX_NEURONS)))
    parser.add_argument("--max-epochs-vanilla", type=parse_epoch_limit, default=int(RUNNER_DEFAULTS["max_epochs_vanilla"]))
    parser.add_argument("--max-epochs-adp", type=parse_epoch_limit, default=int(RUNNER_DEFAULTS["max_epochs_adp"]))
    parser.add_argument("--width-only-width", type=int, default=1)
    parser.add_argument("--width-to-depth-width", type=int, default=1)
    parser.add_argument("--width-to-depth-depth", type=int, default=1)
    parser.add_argument("--modern-script", default="DNN/VANILLA/Dyn_DNN4OPF/models/mlp_w2d_modern.py")
    parser.add_argument("--legacy-script", default="DNN/VANILLA/Dyn_DNN4OPF/models/mlp_w2d_legacy.py")
    parser.add_argument("--vanilla-model", default="fixed_mlp_one_shot_grid")
    parser.add_argument("--lambda-eq", type=float, default=0.0)
    parser.add_argument("--lambda-ineq", type=float, default=0.0)
    parser.add_argument("--lambda-loss", type=float, default=None)
    parser.add_argument("--max-epochs", type=parse_epoch_limit, default=None)
    parser.add_argument("--case-name", default=None)
    parser.add_argument("--adp-mode", default=None)
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def _normalize_runner_args(runner_args: Sequence[str]) -> List[str]:
    args = list(runner_args)
    if args and args[0] == "--":
        args = args[1:]
    return args


def validate_args(args: argparse.Namespace) -> None:
    if int(args.concurrency) < 0:
        raise SystemExit("--concurrency must be >= 0")
    if int(args.gpu_device_index) < 0:
        raise SystemExit("--gpu-device-index must be >= 0")
    if float(args.pressure_poll_interval_sec) <= 0.0 or float(args.post_launch_sample_delay_sec) < 0.0:
        raise SystemExit("pressure polling must be > 0 and post-launch delay must be >= 0")
    if float(args.host_ram_resume_avail_mib) < float(args.host_ram_stop_avail_mib):
        raise SystemExit("--host-ram-resume-avail-mib must be >= --host-ram-stop-avail-mib")
    if float(args.gpu_resume_avail_mib) < float(args.gpu_stop_avail_mib):
        raise SystemExit("--gpu-resume-avail-mib must be >= --gpu-stop-avail-mib")
    if not (0.0 < float(args.batch_backoff_factor) < 1.0):
        raise SystemExit("--batch-backoff-factor must be between 0 and 1")
    if int(args.repeat_count) < 0:
        raise SystemExit("--repeat-count must be >= 0")
    if int(args.max_retries) < 0:
        raise SystemExit("--max-retries must be >= 0")
    if int(args.prediction_repeats) < 1 or int(args.missing_present_task_repeats) < 1:
        raise SystemExit("--prediction-repeats and --missing-present-task-repeats must be >= 1")
    if int(args.max_active_jobs) < 0 or int(args.max_active_gpu_jobs) < 0:
        raise SystemExit("--max-active-jobs and --max-active-gpu-jobs must be >= 0")
    if float(args.launch_host_reserve_mib) < 0.0 or float(args.launch_gpu_reserve_mib) < 0.0:
        raise SystemExit("--launch-host-reserve-mib and --launch-gpu-reserve-mib must be >= 0")
    if float(args.gpu_child_gate_close_buffer_mib) < 0.0 or float(args.host_child_gate_close_buffer_mib) < 0.0:
        raise SystemExit("--gpu-child-gate-close-buffer-mib and --host-child-gate-close-buffer-mib must be >= 0")
    if float(args.external_memory_drop_reopen_mib) <= 0.0:
        raise SystemExit("--external-memory-drop-reopen-mib must be > 0")
    if int(args.width_to_depth_width) < 1 or int(args.width_to_depth_depth) < 1:
        raise SystemExit("--width-to-depth-width and --width-to-depth-depth must be >= 1")
    if int(args.max_width) < 1 or int(args.max_depth) < 1:
        raise SystemExit("--max-width and --max-depth must be >= 1")
    if int(args.max_neurons) < 1:
        raise SystemExit("--max-neurons must be >= 1")
    if int(args.max_epochs_vanilla) < 0 or int(args.max_epochs_adp) < 0:
        raise SystemExit("epoch limits must be non-negative; 0 means unlimited")
    depth_only_widths = [int(item) for item in str(args.depth_only_widths).split(",") if item.strip()]
    width_to_depth_widths = [int(item) for item in str(args.width_to_depth_widths).split(",") if item.strip()]
    vanilla_depths = [int(item) for item in str(args.vanilla_depths).split(",") if item.strip()]
    vanilla_widths = [int(item) for item in str(args.vanilla_widths).split(",") if item.strip()]
    if not depth_only_widths:
        raise SystemExit("--depth-only-widths must contain at least one width")
    if not width_to_depth_widths:
        raise SystemExit("--width-to-depth-widths must contain at least one width")
    if not vanilla_depths:
        raise SystemExit("--vanilla-depths must contain at least one depth")
    if not vanilla_widths:
        raise SystemExit("--vanilla-widths must contain at least one width")
    if any(width < 1 for width in depth_only_widths):
        raise SystemExit("--depth-only-widths values must be >= 1")
    if any(width < 1 for width in width_to_depth_widths):
        raise SystemExit("--width-to-depth-widths values must be >= 1")
    if any(width > int(args.max_width) for width in width_to_depth_widths):
        raise SystemExit("--width-to-depth-widths values must be <= --max-width")
    if any(depth < 1 or depth > int(args.max_depth) for depth in vanilla_depths):
        raise SystemExit("--vanilla-depths values must be between 1 and --max-depth")
    if any(width < 1 or width > int(args.max_width) for width in vanilla_widths):
        raise SystemExit("--vanilla-widths values must be between 1 and --max-width")
    width_only_depths = [int(item) for item in str(args.width_only_depths).split(",") if item.strip()]
    if not width_only_depths:
        raise SystemExit("--width-only-depths must contain at least one depth")
    if any(depth < 1 or depth > int(args.max_depth) for depth in width_only_depths):
        raise SystemExit("--width-only-depths values must be between 1 and --max-depth")
    if any(width > int(args.max_width) for width in depth_only_widths):
        raise SystemExit("--depth-only-widths values must be <= --max-width")
    if int(args.width_only_width) < 1 or int(args.width_only_width) > int(args.max_width):
        raise SystemExit("--width-only-width must be between 1 and --max-width")
    if int(args.width_to_depth_width) > int(args.max_width) or int(args.width_to_depth_depth) > int(args.max_depth):
        raise SystemExit("--width-to-depth-width/--width-to-depth-depth must not exceed the global caps")


def resolve_concurrency(args: argparse.Namespace) -> int:
    if args.concurrency_file:
        try:
            text = Path(args.concurrency_file).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise SystemExit(f"--concurrency-file could not be read: {exc}") from exc
        if text:
            try:
                value = int(text)
            except ValueError as exc:
                raise SystemExit("--concurrency-file must contain one integer >= 0") from exc
            if value < 0:
                raise SystemExit("--concurrency-file must contain one integer >= 0")
            return value if value > 0 else UNBOUNDED_CONCURRENCY
    value = int(args.concurrency)
    return value if value > 0 else UNBOUNDED_CONCURRENCY


def resolve_launch_capacity(
    args: argparse.Namespace,
    host_available_effective: float,
    gpu_free_effective: Optional[float],
) -> int:
    requested = resolve_concurrency(args)
    host_reserve = float(args.launch_host_reserve_mib)
    gpu_reserve = float(args.launch_gpu_reserve_mib)

    # Preserve the caller's contract: a zero reserve means "do not impose an
    # extra synthetic reserve here". Admission is still bounded by the live
    # pressure gates and explicit max-active limits elsewhere.
    if host_reserve <= 0.0:
        host_slots = UNBOUNDED_CONCURRENCY
    else:
        host_slots = max(1, int(max(0.0, host_available_effective) // host_reserve))

    if gpu_reserve <= 0.0 or gpu_free_effective is None:
        gpu_slots = UNBOUNDED_CONCURRENCY if gpu_free_effective is not None else host_slots
    else:
        gpu_slots = max(1, int(max(0.0, gpu_free_effective) // gpu_reserve))

    launch_capacity = max(1, min(requested, host_slots, gpu_slots))
    if int(args.max_active_jobs) > 0:
        launch_capacity = min(launch_capacity, int(args.max_active_jobs))
    return max(1, launch_capacity)


def build_job_id(phase: str, case_name: str, repeat_index: int, label: str) -> str:
    return f"{phase}/{case_name}/repeat_{repeat_index:02d}/{label}"


def build_child_command(args: argparse.Namespace, spec: Dict[str, Any], results_dir: Path) -> Tuple[str, ...]:
    python_bin = sys.executable
    case_name = spec["case_name"]
    family = str(spec["family"])
    head_variant = str(spec.get("head_variant", "single_head")).strip().lower()
    if spec["family"] == "vanilla":
        cmd = [
            python_bin,
            "-u",
            "universal_main.py",
            "--model",
            args.vanilla_model,
            "--case-name",
            case_name,
            "--results-dir",
            str(results_dir),
            "--seed",
            str(int(spec["seed"])),
            "--lambda-eq",
            str(args.lambda_eq),
            "--lambda-ineq",
            str(args.lambda_ineq),
            "--max-epochs",
            str(args.max_epochs_vanilla),
            f"max_width={int(args.max_width)}",
            f"max_depth={int(args.max_depth)}",
            f"max_neurons={int(args.max_neurons)}",
            "patience=10",
            f"depth={int(spec['depth'])}",
            f"width={int(spec['width'])}",
        ]
        if args.lambda_loss is not None:
            cmd.extend(["--lambda-loss", str(args.lambda_loss)])
        return tuple(cmd)

    model_name = _model_name_for_family(family, head_variant)

    init_width = int(spec.get("width", 1))
    init_depth = int(spec.get("depth", 1))
    if spec["family"] in {"stagewise_depth_growth_start_new_stage_at_width_1", "stagewise_depth_growth_start_new_stage_at_current_width"}:
        init_width = int(spec["width"])
        init_depth = 1
    elif spec["family"] in {
        "loss_guided_neural_densification_width_growth_stop_on_first_non_improvement",
        "patience_based_width_growth_all_layers_together",
        "patience_based_width_growth_round_robin_layer_wise",
    }:
        init_width = int(args.width_only_width)
    elif spec["family"] in {
        "width_first_round_robin_widen_then_add_depth_seed_new_depth_at_width_1",
        "width_first_all_layers_widen_then_add_depth_seed_new_depth_at_current_width",
        "width_first_round_robin_widen_then_add_depth_seed_new_depth_at_current_width",
        "width_first_all_layers_widen_then_add_depth_seed_new_depth_at_width_1",
        "depth_first_round_robin_widen_then_add_width_seed_new_width_at_width_1",
        "depth_first_all_layers_widen_then_add_width_seed_new_width_at_current_width",
        "depth_first_round_robin_widen_then_add_width_seed_new_width_at_current_width",
        "depth_first_all_layers_widen_then_add_width_seed_new_width_at_width_1",
    }:
        init_width = int(args.width_to_depth_width)
        init_depth = int(args.width_to_depth_depth)

    cmd = [
        python_bin,
        "-u",
        "universal_main.py",
        "--model",
        model_name,
        "--case-name",
        case_name,
        "--results-dir",
        str(results_dir),
        "--max-epochs",
        str(args.max_epochs_adp if args.max_epochs is None else args.max_epochs),
        "--seed",
        str(int(spec["seed"])),
        f"max_width={int(args.max_width)}",
        f"max_depth={int(args.max_depth)}",
        f"init_width={init_width}",
        f"init_depth={init_depth}",
        f"patience=10",
        f"trials_depth=10",
        f"trials_width=10",
        f"max_neurons={int(args.max_neurons)}",
        f"width_growth_strategy={'round_robin' if spec['family'] in {'patience_based_width_growth_round_robin_layer_wise', 'width_first_round_robin_widen_then_add_depth_seed_new_depth_at_width_1', 'width_first_round_robin_widen_then_add_depth_seed_new_depth_at_current_width', 'depth_first_round_robin_widen_then_add_width_seed_new_width_at_width_1', 'depth_first_round_robin_widen_then_add_width_seed_new_width_at_current_width'} else 'all_layers'}",
    ]
    if spec["family"] == "width_first_all_layers_widen_then_add_depth_seed_new_depth_at_current_width":
        cmd.append("seed_new_depth_to_current_width=1")
    elif spec["family"] == "width_first_round_robin_widen_then_add_depth_seed_new_depth_at_current_width":
        cmd.append("seed_new_depth_to_current_width=1")
    elif spec["family"] == "width_first_all_layers_widen_then_add_depth_seed_new_depth_at_width_1":
        cmd.append("seed_new_depth_to_current_width=0")
    if spec["family"] == "depth_first_round_robin_widen_then_add_width_seed_new_width_at_width_1":
        cmd.append("loop_order=depth_to_width")
    elif spec["family"] == "depth_first_all_layers_widen_then_add_width_seed_new_width_at_current_width":
        cmd.append("loop_order=depth_to_width")
        cmd.append("seed_new_depth_to_current_width=1")
    elif spec["family"] == "depth_first_round_robin_widen_then_add_width_seed_new_width_at_current_width":
        cmd.append("loop_order=depth_to_width")
        cmd.append("seed_new_depth_to_current_width=1")
    elif spec["family"] == "depth_first_all_layers_widen_then_add_width_seed_new_width_at_width_1":
        cmd.append("loop_order=depth_to_width")
        cmd.append("seed_new_depth_to_current_width=0")
    elif spec["family"] == "stagewise_depth_growth_start_new_stage_at_width_1":
        cmd.append("seed_new_depth_to_current_width=0")
    elif spec["family"] == "stagewise_depth_growth_start_new_stage_at_current_width":
        cmd.append("seed_new_depth_to_current_width=1")
    cmd.extend(_normalize_runner_args(getattr(args, "runner_args", [])))
    return tuple(cmd)


def build_suite_jobs(args: argparse.Namespace, run_root: Path, results_root: Path) -> List[JobSpec]:
    cases = [case.strip() for case in str(args.cases).split(",") if case.strip()]
    width_only_depths = [int(item) for item in str(args.width_only_depths).split(",") if item.strip()]
    depth_only_widths = [int(item) for item in str(args.depth_only_widths).split(",") if item.strip()]
    vanilla_depths = [int(item) for item in str(args.vanilla_depths).split(",") if item.strip()]
    vanilla_widths = [int(item) for item in str(args.vanilla_widths).split(",") if item.strip()]
    head_variants = _normalize_head_variants(getattr(args, "head_variants", None))
    both_grow_width = int(args.width_to_depth_width)
    both_grow_depth = int(args.width_to_depth_depth)
    include_vanilla = bool(args.include_vanilla)
    selected_tasks = {task.strip() for task in str(args.tasks).split(",") if task.strip()}
    if not selected_tasks or "all" in selected_tasks:
        selected_tasks = {"width_only", "width_to_depth", "depth_to_width", "depth_only"}
    surface = str(args.surface).strip().lower()
    if surface == "active":
        selected_tasks &= {"width_only", "width_to_depth", "depth_to_width", "depth_only"}
    elif surface == "backlog":
        selected_tasks.clear()
    elif surface != "all":
        raise SystemExit(f"Unsupported surface: {args.surface}")
    include_width_only = bool(args.include_width_only and ("width_only" in selected_tasks))
    include_width_to_depth = bool(args.include_width_to_depth and ("width_to_depth" in selected_tasks))
    include_depth_to_width = bool(args.include_depth_to_width and ("depth_to_width" in selected_tasks))
    include_depth_only = bool(args.include_depth_only and ("depth_only" in selected_tasks))

    jobs: List[JobSpec] = []
    for case_name in cases:
        adp_repeat_count = int(args.missing_present_task_repeats) if (include_width_only or include_width_to_depth or include_depth_to_width or include_depth_only) else 0
        repeat_count = max(int(args.repeat_count), adp_repeat_count, 1)
        for repeat_index in range(1, repeat_count + 1):
            repeat_dir = results_root / case_name / f"repeat_{repeat_index:02d}"

            def emit_job(family: str, head_variant: str, depth: int, width: int, *, width_digits: int) -> None:
                phase = _phase_for_head_variant(family, head_variant)
                depth_text = f"{int(depth):02d}"
                width_text = f"{int(width):0{int(width_digits)}d}"
                label = f"{phase}_d{depth_text}_w{width_text}"
                job_seed = derive_suite_seed(
                    int(args.seed),
                    mode=str(args.mode),
                    case_name=case_name,
                    repeat_index=repeat_index,
                    phase=phase,
                    family=family,
                    head_variant=head_variant,
                    depth=int(depth),
                    width=int(width),
                )
                job_id = build_job_id(phase, case_name, repeat_index, label)
                result_dir = repeat_dir / phase / f"d{depth_text}_w{width_text}"
                command = build_child_command(
                    args,
                    {
                        "family": family,
                        "case_name": case_name,
                        "depth": int(depth),
                        "width": int(width),
                        "head_variant": head_variant,
                        "seed": int(job_seed),
                    },
                    result_dir,
                )
                jobs.append(
                    JobSpec(
                        job_id=job_id,
                        phase=phase,
                        family=family,
                        case_name=case_name,
                        repeat_index=repeat_index,
                        depth=int(depth),
                        width=int(width),
                        seed=int(job_seed),
                        results_dir=str(result_dir),
                        command=command,
                        head_variant=head_variant,
                    )
                )

            if include_width_only and repeat_index <= max(1, adp_repeat_count or int(args.repeat_count)):
                for head_variant in head_variants:
                    for depth in width_only_depths:
                        emit_job("loss_guided_neural_densification_width_growth_stop_on_first_non_improvement", head_variant, int(depth), int(args.width_only_width), width_digits=2)
                    for depth in width_only_depths:
                        emit_job("patience_based_width_growth_round_robin_layer_wise", head_variant, int(depth), int(args.width_only_width), width_digits=2)
                    for depth in width_only_depths:
                        emit_job("patience_based_width_growth_all_layers_together", head_variant, int(depth), int(args.width_only_width), width_digits=2)

            if include_width_to_depth and repeat_index <= max(1, adp_repeat_count or int(args.repeat_count)):
                for head_variant in head_variants:
                    for family in (
                        "width_first_round_robin_widen_then_add_depth_seed_new_depth_at_width_1",
                        "width_first_all_layers_widen_then_add_depth_seed_new_depth_at_current_width",
                        "width_first_round_robin_widen_then_add_depth_seed_new_depth_at_current_width",
                        "width_first_all_layers_widen_then_add_depth_seed_new_depth_at_width_1",
                    ):
                        emit_job(family, head_variant, both_grow_depth, both_grow_width, width_digits=3)

            if include_depth_to_width and repeat_index <= max(1, adp_repeat_count or int(args.repeat_count)):
                for head_variant in head_variants:
                    for family in (
                        "depth_first_round_robin_widen_then_add_width_seed_new_width_at_width_1",
                        "depth_first_all_layers_widen_then_add_width_seed_new_width_at_current_width",
                        "depth_first_round_robin_widen_then_add_width_seed_new_width_at_current_width",
                        "depth_first_all_layers_widen_then_add_width_seed_new_width_at_width_1",
                    ):
                        emit_job(family, head_variant, both_grow_depth, both_grow_width, width_digits=3)

            if include_depth_only and repeat_index <= max(1, adp_repeat_count or int(args.repeat_count)):
                for head_variant in head_variants:
                    for width in depth_only_widths:
                        emit_job("stagewise_depth_growth_start_new_stage_at_width_1", head_variant, 1, int(width), width_digits=3)
                    for width in depth_only_widths:
                        emit_job("stagewise_depth_growth_start_new_stage_at_current_width", head_variant, 1, int(width), width_digits=3)

            if include_vanilla and surface != "backlog":
                for depth in vanilla_depths:
                    for width in vanilla_widths:
                        emit_job("vanilla", "single_head", int(depth), int(width), width_digits=max(3, len(str(int(width)))))

    start = max(0, int(args.job_start_index))
    jobs = jobs[start:]
    if int(args.job_limit) > 0:
        jobs = jobs[: int(args.job_limit)]
    return jobs


def load_job_state(path: Path) -> Dict[str, Any]:
    payload = _load_json(path, default={})
    return payload if isinstance(payload, dict) else {}


def save_job_state(path: Path, payload: Dict[str, Any]) -> None:
    _atomic_write_json(path, payload)


def _is_live_pid(pid: Any) -> bool:
    try:
        pid_int = int(pid)
    except Exception:
        return False
    if pid_int <= 0:
        return False
    try:
        if not psutil.pid_exists(pid_int):
            return False
        proc = psutil.Process(pid_int)
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except Exception:
        return False


def _normalize_cmdline(cmdline: Any) -> tuple[str, ...]:
    if not cmdline:
        return ()
    try:
        return tuple(str(part) for part in cmdline if str(part))
    except Exception:
        return ()


def _find_live_pid_for_command(command: Any) -> int | None:
    expected = _normalize_cmdline(command)
    if not expected:
        return None

    candidates: list[tuple[float, int]] = []
    try:
        for proc in psutil.process_iter(["pid", "cmdline", "create_time"]):
            try:
                cmdline = _normalize_cmdline(proc.info.get("cmdline"))
                if not cmdline or cmdline != expected:
                    continue
                pid = int(proc.info["pid"])
                create_time = float(proc.info.get("create_time") or 0.0)
                if pid > 0 and proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
                    candidates.append((create_time, pid))
            except Exception:
                continue
    except Exception:
        return None

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][1]


def _reconcile_stale_job_state(state_path: Path, state: Dict[str, Any]) -> Dict[str, Any]:
    status = str(state.get("status") or "")
    live_pid = state.get("pid") if _is_live_pid(state.get("pid")) else None
    if live_pid is None:
        live_pid = _find_live_pid_for_command(state.get("command"))
    if live_pid is not None and status in {"running", "launching", "retrying"}:
        if status == "running" and int(state.get("pid") or 0) == int(live_pid):
            return state

        recovered_at = _now_stamp()
        recovered = dict(state)
        recovered.update(
            {
                "status": "running",
                "paused": False,
                "pause_reason": "",
                "pid": live_pid,
                "last_exit_code": None,
                "ended_at": None,
                "recovered_from_status": status,
                "recovered_at": recovered_at,
                "updated_at": recovered_at,
            }
        )
        save_job_state(state_path, recovered)
        return recovered

    if status == "paused":
        recovered_at = _now_stamp()
        recovered = dict(state)
        recovered.update(
            {
                "paused": True,
                "pause_reason": str(state.get("pause_reason") or ""),
                "pid": None,
                "last_exit_code": None,
                "ended_at": None,
                "recovered_from_status": status,
                "recovered_at": recovered_at,
                "updated_at": recovered_at,
            }
        )
        save_job_state(state_path, recovered)
        return recovered

    if status not in {"running", "launching", "retrying"}:
        return state

    recovered_at = _now_stamp()
    recovered = dict(state)
    recovered.update(
        {
            "status": "queued",
            "paused": False,
            "pause_reason": "",
            "pid": None,
            "last_exit_code": None,
            "ended_at": None,
            "recovered_from_status": status,
            "recovered_at": recovered_at,
            "updated_at": recovered_at,
        }
    )
    save_job_state(state_path, recovered)
    return recovered


def _requeue_recovered_runtime_after_exit(
    runtime: JobRuntime,
    active: List[JobRuntime],
    queue: Deque[JobRuntime],
) -> bool:
    """Requeue a child whose real exit status was unknowable after restart."""
    if not isinstance(runtime.proc, RecoveredProcessHandle):
        return False
    if runtime in active:
        active.remove(runtime)
    runtime.proc = None
    runtime.paused = False
    runtime.pause_reason = ""
    runtime.last_exit_code = None
    runtime.ended_at = None
    save_job_state(
        runtime.state_path,
        {
            **runtime.to_state(),
            "status": "queued",
            "pid": None,
            "reason": "recovered_child_exit_status_unknown",
            "updated_at": _now_stamp(),
        },
    )
    if runtime not in queue:
        queue.append(runtime)
    return True


def write_manifest(run_root: Path, args: argparse.Namespace, jobs: Sequence[JobSpec]) -> None:
    head_variants = _normalize_head_variants(getattr(args, "head_variants", None))
    vanilla_depths = [int(item) for item in str(args.vanilla_depths).split(",") if item.strip()]
    vanilla_widths = [int(item) for item in str(args.vanilla_widths).split(",") if item.strip()]
    effective_repeat_count = max(
        int(args.repeat_count),
        int(args.prediction_repeats),
        int(args.missing_present_task_repeats),
        1,
    )
    manifest = {
        "schema": 1,
        "run_root": str(run_root),
        "results_dir": str(Path(args.results_dir)),
        "source_run_root": args.source_run_root,
        "scheduler": args.scheduler,
        "surface": args.surface,
        "tasks": args.tasks,
        "concurrency": resolve_concurrency(args),
        "max_active_jobs": int(args.max_active_jobs),
        "max_active_gpu_jobs": int(args.max_active_gpu_jobs),
        "cpu_only": bool(getattr(args, "cpu_only", False)),
        "cases": [case.strip() for case in str(args.cases).split(",") if case.strip()],
        "repeat_count": int(args.repeat_count),
        "max_retries": int(args.max_retries),
        "effective_repeat_count": int(effective_repeat_count),
        "prediction_repeats": int(args.prediction_repeats),
        "missing_present_task_repeats": int(args.missing_present_task_repeats),
        "head_variants": head_variants,
        "param_band": args.param_band,
        "legacy_architecture_grid": bool(args.legacy_architecture_grid),
        "include_small_stl": bool(args.include_small_stl),
        "include_vanilla": bool(args.include_vanilla),
        "include_width_only": bool(args.include_width_only),
        "include_width_to_depth": bool(args.include_width_to_depth),
        "include_depth_to_width": bool(args.include_depth_to_width),
        "include_depth_only": bool(args.include_depth_only),
        "width_only_depths": [int(item) for item in str(args.width_only_depths).split(",") if item.strip()],
        "width_only_width": int(args.width_only_width),
        "width_to_depth_width": int(args.width_to_depth_width),
        "width_to_depth_depth": int(args.width_to_depth_depth),
        "width_to_depth_widths": [int(item) for item in str(args.width_to_depth_widths).split(",") if item.strip()],
        "depth_only_widths": [int(item) for item in str(args.depth_only_widths).split(",") if item.strip()],
        "vanilla_depths": vanilla_depths,
        "vanilla_widths": vanilla_widths,
        "vanilla_model": args.vanilla_model,
        "max_width": int(args.max_width),
        "max_depth": int(args.max_depth),
        "max_neurons": int(args.max_neurons),
        "gpu_child_gate_close_buffer_mib": float(args.gpu_child_gate_close_buffer_mib),
        "host_child_gate_close_buffer_mib": float(args.host_child_gate_close_buffer_mib),
        "external_memory_drop_reopen_mib": float(args.external_memory_drop_reopen_mib),
        "host_ram_stop_avail_mib": float(args.host_ram_stop_avail_mib),
        "host_ram_resume_avail_mib": float(args.host_ram_resume_avail_mib),
        "gpu_stop_avail_mib": float(args.gpu_stop_avail_mib),
        "gpu_resume_avail_mib": float(args.gpu_resume_avail_mib),
        "swap_pressure_limit_pct": float(args.swap_pressure_limit_pct),
        "swap_resume_pct": float(args.swap_resume_pct),
        "launch_host_reserve_mib": float(args.launch_host_reserve_mib),
        "launch_gpu_reserve_mib": float(args.launch_gpu_reserve_mib),
        "pressure_poll_interval_sec": float(args.pressure_poll_interval_sec),
        "post_launch_sample_delay_sec": float(args.post_launch_sample_delay_sec),
        "batch_backoff_factor": float(args.batch_backoff_factor),
        "gpu_device_index": int(args.gpu_device_index),
        "seed": int(args.seed),
        "max_epochs_vanilla": int(args.max_epochs_vanilla),
        "max_epochs_adp": int(args.max_epochs_adp),
        "job_count": len(jobs),
        "jobs": [dataclasses.asdict(job) for job in jobs],
    }
    manifest["signature"] = _json_signature(manifest)
    manifest_path = run_root / "manifest.json"
    job_manifest_path = run_root / "job_manifest.json"
    if manifest_path.exists():
        existing = load_job_state(manifest_path)
        if isinstance(existing, dict) and (
            existing.get("signature") == manifest["signature"]
            or _manifest_equivalent_under_interpreter_aliases(existing, manifest)
        ):
            if not job_manifest_path.exists():
                _atomic_write_json(job_manifest_path, existing)
            return
        _atomic_write_json(run_root / "manifest_mismatch.json", {"existing": existing, "new": manifest})
        raise SystemExit(f"manifest mismatch for {run_root}; refusing to overwrite existing run plan")
    _atomic_write_json(manifest_path, manifest)
    _atomic_write_json(job_manifest_path, manifest)


def _manifest_equivalent_under_interpreter_aliases(existing: Dict[str, Any], new: Dict[str, Any]) -> bool:
    """Accept a restart when only argv[0]'s symlink spelling changed.

    The runner's generated jobs use ``sys.executable``. On Linux, the venv's
    ``python`` and ``python3`` names can resolve to the same binary while
    producing different manifest signatures. Normalize only absolute argv[0]
    paths for comparison; all other plan fields must remain byte-for-byte
    equivalent after JSON normalization.
    """
    try:
        old_view = json.loads(json.dumps(existing))
        new_view = json.loads(json.dumps(new))
        old_view.pop("signature", None)
        new_view.pop("signature", None)
        for view in (old_view, new_view):
            for job in view.get("jobs", []):
                command = job.get("command") if isinstance(job, dict) else None
                if isinstance(command, list) and command and os.path.isabs(str(command[0])):
                    command[0] = os.path.realpath(str(command[0]))
        return old_view == new_view
    except Exception:
        return False


def update_batch_backoff_state(run_root: Path, payload: Dict[str, Any]) -> None:
    _atomic_write_json(run_root / "batch_backoff_state.json", payload)


def load_batch_backoff_state(run_root: Path) -> Dict[str, Any]:
    payload = _load_json(run_root / "batch_backoff_state.json", default={})
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("batch_scale", 1.0)
    payload.setdefault("pressure_backoff_pending", False)
    payload.setdefault("last_backoff_reason", None)
    payload.setdefault("last_backoff_at", None)
    payload.setdefault("gpu_launches_enabled", True)
    payload.setdefault("cpu_launches_enabled", True)
    payload.setdefault("gpu_gate_closed", False)
    payload.setdefault("host_gate_closed", False)
    payload.setdefault("gpu_gate_lock_job_id", None)
    payload.setdefault("host_gate_lock_job_id", None)
    payload.setdefault("gpu_gate_close_buffer_mib", 100.0)
    payload.setdefault("host_gate_close_buffer_mib", 500.0)
    payload.setdefault("gpu_hold_until", 0.0)
    payload.setdefault("cpu_hold_until", 0.0)
    payload.setdefault("gpu_peak_vram_mib", 0.0)
    payload.setdefault("cpu_peak_host_mib", 0.0)
    payload.setdefault("last_child_gpu_vram_mib", 0.0)
    payload.setdefault("last_child_host_rss_mib", 0.0)
    payload.setdefault("last_child_gpu_remaining_mib", 0.0)
    payload.setdefault("last_child_host_remaining_mib", 0.0)
    payload.setdefault("last_child_job_id", None)
    payload.setdefault("external_host_non_python_peak_mib", 0.0)
    payload.setdefault("external_gpu_non_python_peak_mib", 0.0)
    payload.setdefault("last_external_host_non_python_mib", 0.0)
    payload.setdefault("last_external_gpu_non_python_mib", 0.0)
    payload.setdefault("external_host_non_python_pids", [])
    payload.setdefault("external_gpu_non_python_pids", [])
    return payload


def normalize_dual_gate_state(batch_state: Dict[str, Any]) -> Dict[str, Any]:
    batch_state.setdefault("gpu_launches_enabled", True)
    batch_state.setdefault("cpu_launches_enabled", True)
    batch_state.setdefault("gpu_gate_closed", False)
    batch_state.setdefault("host_gate_closed", False)
    batch_state.setdefault("gpu_gate_lock_job_id", None)
    batch_state.setdefault("host_gate_lock_job_id", None)
    batch_state.setdefault("gpu_gate_close_buffer_mib", 100.0)
    batch_state.setdefault("host_gate_close_buffer_mib", 500.0)
    batch_state.setdefault("gpu_hold_until", 0.0)
    batch_state.setdefault("cpu_hold_until", 0.0)
    batch_state.setdefault("gpu_peak_vram_mib", 0.0)
    batch_state.setdefault("cpu_peak_host_mib", 0.0)
    batch_state.setdefault("last_child_gpu_vram_mib", 0.0)
    batch_state.setdefault("last_child_host_rss_mib", 0.0)
    batch_state.setdefault("last_child_gpu_remaining_mib", 0.0)
    batch_state.setdefault("last_child_host_remaining_mib", 0.0)
    batch_state.setdefault("last_child_job_id", None)
    batch_state.setdefault("external_host_non_python_peak_mib", 0.0)
    batch_state.setdefault("external_gpu_non_python_peak_mib", 0.0)
    batch_state.setdefault("last_external_host_non_python_mib", 0.0)
    batch_state.setdefault("last_external_gpu_non_python_mib", 0.0)
    batch_state.setdefault("external_host_non_python_pids", [])
    batch_state.setdefault("external_gpu_non_python_pids", [])
    return batch_state


def job_directory(run_root: Path, spec: JobSpec) -> Path:
    return run_root / "jobs" / spec.job_id


def task_directory_for_spec(run_root: Path, spec: JobSpec) -> Path:
    return run_root / spec.case_name / f"repeat_{int(spec.repeat_index):02d}"


def method_directory_for_spec(run_root: Path, spec: JobSpec) -> Path:
    return task_directory_for_spec(run_root, spec) / spec.phase


def candidate_state_path_for_spec(run_root: Path, spec: JobSpec) -> Path:
    return job_directory(run_root, spec) / "candidate_state.json"


def method_state_path_for_spec(run_root: Path, spec: JobSpec) -> Path:
    return method_directory_for_spec(run_root, spec) / "method_state.json"


def task_state_path_for_spec(run_root: Path, spec: JobSpec) -> Path:
    return task_directory_for_spec(run_root, spec) / "task_state.json"


def replay_manifest_path_for_spec(run_root: Path, spec: JobSpec) -> Path:
    return task_directory_for_spec(run_root, spec) / "replay_manifest.json"


def candidate_summary_path_for_spec(run_root: Path, spec: JobSpec) -> Path:
    return job_directory(run_root, spec) / "candidate_summary.json"


def _load_job_state_file(job_dir: Path) -> Dict[str, Any]:
    payload = _load_json(job_dir / "job_state.json", default={})
    return payload if isinstance(payload, dict) else {}


def _iter_job_states(run_root: Path) -> Sequence[Dict[str, Any]]:
    jobs_root = run_root / "jobs"
    if not jobs_root.exists():
        return []
    states: List[Dict[str, Any]] = []
    for job_dir in sorted(jobs_root.iterdir()):
        if not job_dir.is_dir():
            continue
        state = _load_job_state_file(job_dir)
        if state:
            states.append(state)
    return states


def _summarize_group_states(run_root: Path, spec: JobSpec) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    task_dir = task_directory_for_spec(run_root, spec)
    method_dir = method_directory_for_spec(run_root, spec)
    candidate_dir = job_directory(run_root, spec)

    all_states = list(_iter_job_states(run_root))
    method_candidates = [
        state
        for state in all_states
        if str(state.get("case_name")) == spec.case_name
        and int(state.get("repeat_index", -1)) == int(spec.repeat_index)
        and str(state.get("phase")) == str(spec.phase)
    ]
    task_candidates = [
        state
        for state in all_states
        if str(state.get("case_name")) == spec.case_name
        and int(state.get("repeat_index", -1)) == int(spec.repeat_index)
    ]

    def _counts(states: Sequence[Dict[str, Any]]) -> Dict[str, int]:
        counts = {"completed": 0, "running": 0, "paused": 0, "failed": 0, "queued": 0}
        for state in states:
            status = str(state.get("status") or "queued")
            if status in counts:
                counts[status] += 1
            elif status == "launching":
                counts["running"] += 1
            else:
                counts["queued"] += 1
        return counts

    def _ended_at_key(state: Dict[str, Any]) -> Tuple[int, str]:
        ended_at = str(state.get("ended_at") or "")
        is_digit = ended_at.replace(".", "", 1).isdigit()
        return (1 if is_digit else 0, ended_at)

    def _latest_completed(states: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        completed = [state for state in states if str(state.get("status")) == "completed" and _safe_int(state.get("last_exit_code"), 1) == 0]
        if not completed:
            return None
        completed.sort(key=lambda item: (_ended_at_key(item), str(item.get("job_id") or "")))
        return completed[-1]

    method_state = {
        "schema": 1,
        "scope": "method",
        "task": spec.case_name,
        "repeat_index": int(spec.repeat_index),
        "family": spec.phase,
        "candidate_count": len(method_candidates),
        "counts": _counts(method_candidates),
        "latest_completed": _latest_completed(method_candidates),
        "candidate_root": str(method_dir),
        "candidate_states": [state.get("state_path") for state in method_candidates if state.get("state_path")],
        "best_candidate_dir": next((state.get("results_dir") for state in method_candidates if str(state.get("status")) == "completed" and _safe_int(state.get("last_exit_code"), 1) == 0), None),
        "updated_at": _now_stamp(),
    }

    unique_method_roots = sorted(
        {
            str(
                run_root
                / str(state.get("case_name") or spec.case_name)
                / f"repeat_{int(state.get('repeat_index', spec.repeat_index)):02d}"
                / str(state.get("phase") or "")
            )
            for state in task_candidates
            if state.get("phase")
        }
    )

    task_state = {
        "schema": 1,
        "scope": "task",
        "task": spec.case_name,
        "repeat_index": int(spec.repeat_index),
        "family_count": len({str(state.get("phase")) for state in task_candidates if state.get("phase")}),
        "candidate_count": len(task_candidates),
        "counts": _counts(task_candidates),
        "latest_completed": _latest_completed(task_candidates),
        "task_root": str(task_dir),
        "method_roots": unique_method_roots,
        "best_candidate_dir": next((state.get("results_dir") for state in task_candidates if str(state.get("status")) == "completed" and _safe_int(state.get("last_exit_code"), 1) == 0), None),
        "updated_at": _now_stamp(),
    }

    replay_manifest = {
        "schema": 1,
        "task": spec.case_name,
        "repeat_index": int(spec.repeat_index),
        "seed": int(spec.seed),
        "task_root": str(task_dir),
        "method_root": str(method_dir),
        "candidate_dir": str(candidate_dir),
        "candidates": [
            {
                "job_id": state.get("job_id"),
                "phase": state.get("phase"),
                "family": state.get("family"),
                "head_variant": state.get("head_variant"),
                "seed": state.get("seed"),
                "candidate_dir": state.get("results_dir"),
                "state_path": state.get("state_path"),
                "status": state.get("status"),
                "attempt": state.get("attempt"),
                "device_mode": state.get("device_mode"),
                "exit_code": state.get("last_exit_code"),
            }
            for state in task_candidates
        ],
        "updated_at": _now_stamp(),
    }
    return method_state, task_state, replay_manifest


def write_compat_state_views(run_root: Path, spec: JobSpec) -> None:
    candidate_dir = job_directory(run_root, spec)
    state = load_job_state(candidate_dir / "job_state.json")
    if not isinstance(state, dict):
        state = {}
    candidate_state = {
        "schema": 1,
        "scope": "candidate",
        "task": spec.case_name,
        "repeat_index": int(spec.repeat_index),
        "family": spec.family,
        "head_variant": spec.head_variant,
        "phase": spec.phase,
        "job_id": spec.job_id,
        "seed": int(spec.seed),
        "candidate_dir": str(candidate_dir),
        "results_dir": spec.results_dir,
        "architecture": {"depth": int(spec.depth), "width": int(spec.width)},
        "status": state.get("status", "queued"),
        "completed": str(state.get("status")) == "completed" and _safe_int(state.get("last_exit_code"), 1) == 0,
        "attempt": state.get("attempt", 0),
        "device_mode": state.get("device_mode"),
        "pause_count": state.get("pause_count", 0),
        "gpu_pause_count": state.get("gpu_pause_count", 0),
        "host_pause_count": state.get("host_pause_count", 0),
        "force_cpu_after_cuda_oom": False,
        "command": state.get("command") or list(spec.command),
        "last_exit_code": state.get("last_exit_code"),
        "started_at": state.get("started_at"),
        "ended_at": state.get("ended_at"),
        "best_val": state.get("best_val"),
        "best_epoch": state.get("best_epoch"),
        "final_epoch": state.get("final_epoch"),
        "best_checkpoint": state.get("best_checkpoint"),
        "last_checkpoint": state.get("last_checkpoint"),
        "history_rows": state.get("history_rows"),
        "updated_at": _now_stamp(),
    }
    _atomic_write_json(candidate_state_path_for_spec(run_root, spec), candidate_state)

    method_state, task_state, replay_manifest = _summarize_group_states(run_root, spec)
    _atomic_write_json(method_state_path_for_spec(run_root, spec), method_state)
    _atomic_write_json(task_state_path_for_spec(run_root, spec), task_state)
    _atomic_write_json(replay_manifest_path_for_spec(run_root, spec), replay_manifest)


def load_existing_runtime(run_root: Path, spec: JobSpec) -> JobRuntime:
    state_path = job_directory(run_root, spec) / "job_state.json"
    state = _reconcile_stale_job_state(state_path, load_job_state(state_path))
    runtime = JobRuntime(
        spec=spec,
        state_path=state_path,
        log_path=job_directory(run_root, spec) / "job.log",
    )
    live_pid = state.get("pid") if _is_live_pid(state.get("pid")) else None
    if live_pid is None:
        live_pid = _find_live_pid_for_command(state.get("command"))
    if live_pid is not None:
        runtime.proc = RecoveredProcessHandle(int(live_pid))
    runtime.attempt = _safe_int(state.get("attempt"), 0)
    runtime.retry_count = _safe_int(state.get("retry_count"), 0)
    runtime.device_mode = str(state.get("device_mode") or "gpu")
    runtime.paused = bool(state.get("paused", False))
    runtime.pause_reason = str(state.get("pause_reason") or "")
    runtime.pause_count = _safe_int(state.get("pause_count"), 0)
    runtime.gpu_pause_count = _safe_int(state.get("gpu_pause_count"), 0)
    runtime.host_pause_count = _safe_int(state.get("host_pause_count"), 0)
    runtime.force_cpu_after_cuda_oom = False
    runtime.last_exit_code = state.get("last_exit_code")
    runtime.started_at = state.get("started_at")
    runtime.ended_at = state.get("ended_at")
    return runtime


def should_retry(runtime: JobRuntime, max_retries: int) -> bool:
    # Zero is the uncapped suite setting; a positive value limits retries.
    return int(max_retries) <= 0 or int(max_retries) > int(runtime.retry_count)


def is_terminal_inexact_resume_failure(state: Dict[str, Any]) -> bool:
    """Keep unrecoverable exact-resume failures terminal across runner restarts."""
    return str(state.get("status", "")).lower() in {"failed", "fail"} and state.get("reason") == "inexact_checkpoint_resume"


def current_pids(active_jobs: Sequence[JobRuntime]) -> List[int]:
    pids: List[int] = []
    for runtime in active_jobs:
        if runtime.proc is not None and runtime.proc.poll() is None:
            try:
                pids.extend(_process_tree_pids(int(runtime.proc.pid)))
            except Exception:
                pass
    return sorted(set(pids))


def select_launch_device(
    runtime: JobRuntime,
    args: argparse.Namespace,
    host_sample: MemorySample,
    gpu_sample: Optional[GpuMemorySample],
    active_gpu_count: int,
    active_cpu_count: int,
    *,
    host_admission_enabled: bool = True,
    gpu_admission_enabled: bool = True,
) -> Optional[str]:
    if bool(getattr(args, "cpu_only", False)) or _parent_cpu_masked():
        return "cpu" if host_admission_enabled and host_sample.available_mib >= float(args.host_ram_resume_avail_mib) else None
    gpu_allowed = (
        gpu_admission_enabled
        and gpu_sample is not None
        and gpu_sample.free_mib >= float(args.gpu_resume_avail_mib)
        and (int(args.max_active_gpu_jobs) <= 0 or active_gpu_count < int(args.max_active_gpu_jobs))
    )
    host_allowed = (
        host_admission_enabled
        and host_sample.available_mib >= float(args.host_ram_resume_avail_mib)
        and (int(args.max_active_jobs) <= 0 or (active_cpu_count + active_gpu_count) < int(args.max_active_jobs))
    )


    if args.scheduler == "fixed":
        return "gpu" if gpu_allowed else ("cpu" if host_allowed else None)

    if args.scheduler == "gpu_first":
        if gpu_allowed:
            return "gpu"
        if host_allowed:
            return "cpu"
        return None

    if gpu_allowed and host_allowed:
        return "gpu"
    if host_allowed:
        return "cpu"
    return None


def make_child_env(
    args: argparse.Namespace,
    runtime: JobRuntime,
    concurrency: int,
    device_mode: str,
    run_root: Path,
) -> Dict[str, str]:
    env = launcher_child_env(
        os.environ,
        concurrency_hint=concurrency,
        job_key=runtime.spec.job_id,
        shared_cpu=False,
    )
    env["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:128",
    )
    env["OPF_ADP_RESULTS_DIR"] = runtime.spec.results_dir
    env["OPF_ADP_RUN_ROOT"] = str(run_root)
    env["OPF_ADP_JOB_ID"] = runtime.spec.job_id
    env["OPF_ADP_CASE_NAME"] = runtime.spec.case_name
    env["OPF_ADP_REPEAT_INDEX"] = str(runtime.spec.repeat_index)
    env["OPF_ADP_PHASE"] = runtime.spec.phase
    if device_mode not in {"cpu", "gpu"}:
        raise ValueError(f"invalid child backend: {device_mode!r}")
    if device_mode == "gpu":
        if _parent_cpu_masked():
            raise RuntimeError("GPU child cannot override central CPU admission")
        inherited = env.get("CUDA_VISIBLE_DEVICES")
        if inherited not in (None, "", "-1"):
            visible = [item.strip() for item in inherited.split(",") if item.strip()]
            if len(visible) == 1:
                # Respect an outer scheduler's single physical-device mask.
                assigned = visible[0]
            elif str(args.gpu_device_index) in visible:
                assigned = str(args.gpu_device_index)
            else:
                raise RuntimeError("GPU index is not in the inherited CUDA device mask")
        else:
            assigned = str(args.gpu_device_index)
        env["CUDA_VISIBLE_DEVICES"] = assigned
        env["OPF_ADP_DISABLE_GPU_ACCELERATORS"] = "0"
        env.pop("TRAINING_CONTROL_CPU_ONLY", None)
        env["TRAINING_CONTROL_BACKEND"] = "gpu"
        env["TRAINING_CONTROL_LOCAL_GPU_INDEX"] = "0"
    else:
        # A CPU child must also mask RAPIDS, JAX and independent
        # accelerator libraries; PyTorch device='cpu' alone is insufficient.
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["HIP_VISIBLE_DEVICES"] = ""
        env["ROCR_VISIBLE_DEVICES"] = ""
        env["JAX_PLATFORMS"] = "cpu"
        env["JAX_PLATFORM_NAME"] = "cpu"
        env["OPF_ADP_DISABLE_GPU_ACCELERATORS"] = "1"
        env["TRAINING_CONTROL_CPU_ONLY"] = "1"
        env["TRAINING_CONTROL_BACKEND"] = "cpu"
    # Partition CPU budgets across concurrent children so GPU-first suites do
    # not oversubscribe the host with every job claiming the full core set.
    env["TABULAR_CHILD_SHARED_CPU"] = "0"
    env["TABULAR_CPU_WORKERS"] = "0"
    env["TABULAR_CPU_JOB_CONCURRENCY"] = "1" if concurrency >= UNBOUNDED_CONCURRENCY else str(max(1, concurrency))
    # PyTorch warns when deterministic algorithms are enabled without a cuBLAS
    # workspace hint. Set it here so resumed runs inherit the stricter setting.
    env["CUBLAS_WORKSPACE_CONFIG"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return env


def launch_job(
    args: argparse.Namespace,
    runtime: JobRuntime,
    concurrency: int,
    device_mode: str,
    run_root: Path,
    launch_context: Optional[Dict[str, Any]] = None,
) -> None:
    job_dir = job_directory(run_root, runtime.spec)
    _ensure_dir(job_dir)
    _ensure_dir(Path(runtime.spec.results_dir))
    runtime.attempt += 1
    runtime.device_mode = device_mode
    runtime.started_at = _now_stamp()
    runtime.ended_at = None
    _atomic_write_json(job_dir / "command.json", list(runtime.spec.command))
    _atomic_write_json(
        job_dir / "job_metadata.json",
        {
            **dataclasses.asdict(runtime.spec),
            "attempt": runtime.attempt,
            "device_mode": device_mode,
            "launch_context": launch_context or {},
        },
    )
    if launch_context:
        _atomic_write_json(job_dir / "launch_context.json", launch_context)
    save_job_state(
        runtime.state_path,
        {
            **runtime.to_state(),
            "status": "launching",
            "device_mode": device_mode,
            "command": list(runtime.spec.command),
            "results_dir": runtime.spec.results_dir,
            "state_path": str(runtime.state_path),
            "launch_context": launch_context or {},
            "updated_at": _now_stamp(),
        },
    )
    write_compat_state_views(run_root, runtime.spec)
    runtime.last_progress_ts = time.time()

    env = make_child_env(args, runtime, concurrency, device_mode, run_root)
    runtime.log_handle = runtime.log_path.open("a", encoding="utf-8")
    runtime.log_handle.write(f"\n=== LAUNCH {runtime.spec.job_id} attempt={runtime.attempt} device={device_mode} ===\n")
    runtime.log_handle.write("COMMAND: " + " ".join(runtime.spec.command) + "\n")
    if launch_context:
        runtime.log_handle.write("LAUNCH_CONTEXT: " + json.dumps(launch_context, sort_keys=True) + "\n")
    runtime.log_handle.flush()

    try:
        proc = subprocess.Popen(
            list(runtime.spec.command),
            cwd=str(REPO_ROOT),
            stdout=runtime.log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            **_proc_group_kwargs(),
        )
    except Exception:
        try:
            runtime.log_handle.close()
        except Exception:
            pass
        runtime.log_handle = None
        raise
    runtime.proc = proc
    save_job_state(
        runtime.state_path,
            {
                **runtime.to_state(),
                "status": "running",
                "pid": proc.pid,
                "device_mode": device_mode,
                "attempt": runtime.attempt,
                "started_at": runtime.started_at,
                "state_path": str(runtime.state_path),
                "launch_context": launch_context or {},
                "updated_at": _now_stamp(),
            },
    )


def _job_pressure_score(runtime: JobRuntime) -> Tuple[int, int, str]:
    return (int(runtime.spec.depth), int(runtime.spec.width), str(runtime.spec.job_id))


def _latest_active_runtime(active: Sequence[JobRuntime], device_mode: Optional[str] = None) -> Optional[JobRuntime]:
    candidates = [
        runtime
        for runtime in active
        if runtime.proc is not None
        and runtime.proc.poll() is None
        and (device_mode is None or runtime.device_mode == device_mode)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda runtime: (str(runtime.started_at or ""), int(runtime.attempt), str(runtime.spec.job_id)))
    return candidates[-1]


def _tree_processes_for_runtime(runtime: JobRuntime) -> List[psutil.Process[Any]]:
    if runtime.proc is None or runtime.proc.poll() is not None:
        return []
    try:
        proc = psutil.Process(int(runtime.proc.pid))
    except Exception:
        return []

    processes: List[psutil.Process[Any]] = [proc]
    try:
        for child in proc.children(recursive=True):
            try:
                processes.append(child)
            except Exception:
                continue
    except Exception:
        pass
    seen: set[int] = set()
    unique: List[psutil.Process[Any]] = []
    for item in processes:
        try:
            pid = int(item.pid)
        except Exception:
            continue
        if pid in seen:
            continue
        seen.add(pid)
        unique.append(item)
    return unique


def _set_runtime_tree_paused(runtime: JobRuntime, paused: bool) -> None:
    processes = _tree_processes_for_runtime(runtime)
    if not processes:
        return
    for proc in processes:
        try:
            if paused:
                proc.suspend()
            else:
                proc.resume()
        except Exception:
            continue


def _suspend_runtime_for_pressure(
    runtime: JobRuntime,
    *,
    reason: str,
    suspended: List[JobRuntime],
    active: Optional[List[JobRuntime]] = None,
    run_root: Path,
    priority_snapshot: Optional[Dict[str, Any]] = None,
    timeout_sec: float = 10.0,
) -> bool:
    if runtime.proc is None or runtime.proc.poll() is not None or runtime.paused:
        return False
    try:
        _terminate_process_tree(runtime.proc, timeout_sec=timeout_sec)
    except Exception:
        return False

    runtime.paused = True
    runtime.pause_reason = reason
    runtime.pause_count += 1
    if "gpu" in reason or "cuda" in reason:
        runtime.gpu_pause_count += 1
    if "host" in reason or "swap" in reason:
        runtime.host_pause_count += 1
    runtime.last_progress_ts = time.time()
    runtime.last_exit_code = None
    runtime.ended_at = None
    _close_runtime_log_handle(runtime)
    runtime.proc = None
    if active is not None and runtime in active:
        active.remove(runtime)
    if runtime not in suspended:
        suspended.append(runtime)
    save_job_state(
        runtime.state_path,
        {
            **runtime.to_state(),
            "status": "paused",
            "paused": True,
            "pause_reason": reason,
            "pid": None,
            "updated_at": _now_stamp(),
            "resource_snapshot": priority_snapshot or {},
        },
    )
    write_compat_state_views(run_root, runtime.spec)
    return True


def _resume_runtime_from_pause(
    runtime: JobRuntime,
    args: argparse.Namespace,
    concurrency: int,
    run_root: Path,
    *,
    device_mode: str,
    active: Optional[List[JobRuntime]] = None,
    launch_context: Optional[Dict[str, Any]] = None,
) -> bool:
    if runtime.proc is not None and runtime.proc.poll() is None:
        try:
            _terminate_process_tree(runtime.proc)
        except Exception:
            pass
    previous_reason = runtime.pause_reason or "pressure"
    runtime.paused = False
    runtime.pause_reason = ""
    runtime.last_exit_code = None
    runtime.last_progress_ts = time.time()
    try:
        launch_job(
            args,
            runtime,
            concurrency,
            device_mode,
            run_root,
            launch_context=launch_context or {"resumed_from_pause": True},
        )
        if active is not None and runtime not in active:
            active.append(runtime)
    except Exception:
        runtime.proc = None
        _close_runtime_log_handle(runtime)
        runtime.paused = True
        runtime.pause_reason = previous_reason
        runtime.ended_at = None
        save_job_state(
            runtime.state_path,
            {
                **runtime.to_state(),
                "status": "paused",
                "paused": True,
                "pause_reason": previous_reason,
                "pid": None,
                "last_exit_code": None,
                "reason": "resume_launch_error",
                "updated_at": _now_stamp(),
            },
        )
        write_compat_state_views(run_root, runtime.spec)
        raise
    return True


def seed_run_root_from_source(run_root: Path, source_run_root: Optional[str]) -> Dict[str, int]:
    stats = {"files_copied": 0, "files_skipped": 0}
    if not source_run_root:
        return stats
    source = Path(source_run_root)
    if not source.exists():
        return stats

    top_level_files = [
        "manifest.json",
        "job_manifest.json",
        "run_metadata.json",
        "batch_backoff_state.json",
        "run_progress.csv",
        "task_summary.csv",
        "final_report.json",
        "final_report.md",
        "suite_summary.json",
        "suite_summary.md",
    ]
    for name in top_level_files:
        src = source / name
        dst = run_root / name
        if not src.exists():
            continue
        if dst.exists():
            stats["files_skipped"] += 1
            continue
        _ensure_dir(dst.parent)
        shutil.copy2(src, dst)
        stats["files_copied"] += 1

    for src in source.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(source)
        if rel.parts and rel.parts[0] in {"manifest.json", "job_manifest.json", "run_metadata.json", "batch_backoff_state.json", "run_progress.csv", "task_summary.csv", "final_report.json", "final_report.md", "suite_summary.json", "suite_summary.md"}:
            continue
        dst = run_root / rel
        if dst.exists():
            stats["files_skipped"] += 1
            continue
        _ensure_dir(dst.parent)
        shutil.copy2(src, dst)
        stats["files_copied"] += 1

    return stats


def finalize_job(
    args: argparse.Namespace,
    runtime: JobRuntime,
    exit_code: int,
    stdout_log_path: Path,
    progress_path: Path,
    run_root: Path,
    resource_snapshot: Optional[Dict[str, Any]],
    event: str,
    reason: str = "",
) -> None:
    runtime.last_exit_code = int(exit_code)
    runtime.ended_at = runtime.ended_at or _now_stamp()
    if runtime.proc is not None:
        runtime.proc = None
    if runtime.log_handle is not None:
        try:
            runtime.log_handle.write(f"=== FINISH {runtime.spec.job_id} exit={exit_code} reason={reason} ===\n")
            runtime.log_handle.flush()
        except Exception:
            pass
        try:
            runtime.log_handle.close()
        except Exception:
            pass
        runtime.log_handle = None
    state = runtime.to_state()
    state.update(
        {
            "status": "completed" if exit_code == 0 else "failed",
            "last_exit_code": int(exit_code),
            "ended_at": runtime.ended_at,
            "reason": reason,
            "best_val": state.get("best_val"),
            "best_epoch": state.get("best_epoch"),
            "final_epoch": state.get("final_epoch"),
            "best_checkpoint": state.get("best_checkpoint"),
            "last_checkpoint": state.get("last_checkpoint"),
            "history_rows": state.get("history_rows"),
            "state_path": str(runtime.state_path),
            "updated_at": _now_stamp(),
        }
    )
    save_job_state(runtime.state_path, state)
    write_compat_state_views(run_root, runtime.spec)
    _atomic_write_json(
        candidate_summary_path_for_spec(run_root, runtime.spec),
        {
            "schema": 1,
            "job_id": runtime.spec.job_id,
            "task": runtime.spec.case_name,
            "phase": runtime.spec.phase,
            "family": runtime.spec.family,
            "head_variant": runtime.spec.head_variant,
            "results_dir": runtime.spec.results_dir,
            "status": "completed" if exit_code == 0 else "failed",
            "exit_code": int(exit_code),
            "reason": reason,
            "attempt": runtime.attempt,
            "device_mode": runtime.device_mode,
            "updated_at": _now_stamp(),
        },
    )
    _write_csv_row(
        progress_path,
        [
            "timestamp",
            "event",
            "job_id",
            "phase",
            "family",
            "head_variant",
            "case_name",
            "repeat_index",
            "depth",
            "width",
            "attempt",
            "device_mode",
            "exit_code",
            "reason",
            "results_dir",
            "resource_snapshot",
        ],
        {
            "timestamp": _now_stamp(),
            "event": event,
            "job_id": runtime.spec.job_id,
            "phase": runtime.spec.phase,
            "family": runtime.spec.family,
            "head_variant": runtime.spec.head_variant,
            "case_name": runtime.spec.case_name,
            "repeat_index": runtime.spec.repeat_index,
            "depth": runtime.spec.depth,
            "width": runtime.spec.width,
            "attempt": runtime.attempt,
            "device_mode": runtime.device_mode,
            "exit_code": int(exit_code),
            "reason": reason,
            "results_dir": runtime.spec.results_dir,
            "resource_snapshot": json.dumps(resource_snapshot or {}, sort_keys=True),
        },
    )
    del stdout_log_path


def render_summary(run_root: Path, results_root: Path, jobs: Sequence[JobSpec], completed: int, failed: int, retried: int) -> Dict[str, Any]:
    return {
        "run_root": str(run_root),
        "results_dir": str(results_root),
        "total_jobs": len(jobs),
        "completed_jobs": completed,
        "failed_jobs": failed,
        "retried_jobs": retried,
        "timestamp": _now_stamp(),
    }


def _count_text_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)
    except Exception:
        return 0


_BATCH_DONE_RE = re.compile(r"\bbatch=\d+/\d+\s+done\b")
_EPOCH_DONE_RE = re.compile(r"\b(?:ES\s+)?epoch=\d+\b.*\b(?:val|validation|test|eval)=")


def _count_job_log_progress(job_log_path: Path) -> Tuple[int, int]:
    """Count completed batches and epochs from a live child job log.

    Older runs do not have the new ``progress_state.json`` sidecar yet, so the
    suite status loop falls back to the child stdout log. We count only the
    terminal per-batch / per-epoch lines so the numbers reflect completed work,
    not launches.
    """

    batch_count = 0
    epoch_count = 0
    try:
        with job_log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if _BATCH_DONE_RE.search(line):
                    batch_count += 1
                if _EPOCH_DONE_RE.search(line):
                    epoch_count += 1
    except Exception:
        return 0, 0
    return batch_count, epoch_count


def _count_csv_records(path: Path, wanted: set[str]) -> Dict[str, int]:
    counts = {kind: 0 for kind in wanted}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                # Terminal/final rows use CSV record_type=epoch for schema
                # compatibility but kind=final; count only actual epoch
                # records so this offline summary matches progress_state.json.
                kind = str(row.get("kind") or row.get("record_type") or "").strip().lower()
                if kind in counts:
                    counts[kind] += 1
    except Exception:
        return {kind: 0 for kind in wanted}
    return counts


def summarize_suite_telemetry(jobs: Sequence[JobSpec]) -> Dict[str, int]:
    batch_records = 0
    epoch_records = 0
    telemetry_dirs = 0
    telemetry_jobs = 0
    for spec in jobs:
        telemetry_root = Path(spec.results_dir) / "telemetry"
        if not telemetry_root.exists():
            continue
        telemetry_jobs += 1
        seen_any = False
        csv_files = [path for path in telemetry_root.rglob("telemetry.csv") if path.is_file()]
        if csv_files:
            for csv_file in csv_files:
                seen_any = True
                counts = _count_csv_records(csv_file, {"batch", "epoch"})
                batch_records += counts.get("batch", 0)
                epoch_records += counts.get("epoch", 0)
        else:
            for batch_file in telemetry_root.rglob("batch_metrics.jsonl"):
                if not batch_file.is_file():
                    continue
                seen_any = True
                batch_records += _count_text_lines(batch_file)
            for epoch_file in telemetry_root.rglob("epoch_metrics.jsonl"):
                if not epoch_file.is_file():
                    continue
                seen_any = True
                epoch_records += _count_text_lines(epoch_file)
        if seen_any:
            telemetry_dirs += 1
    return {
        "batches_done": int(batch_records),
        "epochs_done": int(epoch_records),
        "telemetry_jobs": int(telemetry_jobs),
        "telemetry_dirs": int(telemetry_dirs),
    }


def summarize_suite_runtime_progress(run_root: Path, jobs: Sequence[JobSpec]) -> Dict[str, int]:
    """Fast progress summary for live status logs.

    This intentionally avoids scanning large telemetry CSVs on every status
    tick. Per-job state files already carry enough lightweight runtime data to
    report overall launch/progress health without stalling the scheduler.
    """

    batches_done = 0
    epochs_done = 0
    live_batches_done = 0
    live_epochs_done = 0
    telemetry_jobs = 0
    telemetry_dirs = 0
    for spec in jobs:
        state = load_job_state(job_directory(run_root, spec) / "job_state.json")
        if not isinstance(state, dict):
            continue
        is_live = state.get("status") in {"running", "paused", "suspended"}
        telemetry_root = Path(spec.results_dir) / "telemetry"
        progress_files = [path for path in telemetry_root.rglob("progress_state.json") if path.is_file()] if telemetry_root.exists() else []
        if progress_files:
            telemetry_jobs += 1
            telemetry_dirs += 1
            latest_progress_file = sorted(progress_files, key=lambda path: (path.stat().st_mtime_ns, str(path)))[-1]
            progress_state = _load_json(latest_progress_file, default={})
            if isinstance(progress_state, dict):
                try:
                    job_batches = int(progress_state.get("history_rows") or 0)
                    batches_done += job_batches
                    if is_live:
                        live_batches_done += job_batches
                except Exception:
                    pass
                try:
                    job_epochs = int(progress_state.get("final_epoch") or 0)
                    epochs_done += job_epochs
                    if is_live:
                        live_epochs_done += job_epochs
                except Exception:
                    pass
        else:
            job_log_path = job_directory(run_root, spec) / "job.log"
            if job_log_path.exists():
                telemetry_jobs += 1
                telemetry_dirs += 1
                batch_count, epoch_count = _count_job_log_progress(job_log_path)
                batches_done += batch_count
                epochs_done += epoch_count
                if is_live:
                    live_batches_done += batch_count
                    live_epochs_done += epoch_count
            else:
                has_state_progress = state.get("history_rows") is not None or state.get("final_epoch") is not None
                if has_state_progress:
                    telemetry_jobs += 1
                if state.get("history_rows") is not None:
                    try:
                        job_batches = int(state.get("history_rows") or 0)
                        batches_done += job_batches
                        if is_live:
                            live_batches_done += job_batches
                    except Exception:
                        pass
                if state.get("final_epoch") is not None:
                    try:
                        job_epochs = int(state.get("final_epoch") or 0)
                        epochs_done += job_epochs
                        if is_live:
                            live_epochs_done += job_epochs
                    except Exception:
                        pass
    return {
        "batches_done": int(batches_done),
        "epochs_done": int(epochs_done),
        "live_batches_done": int(live_batches_done),
        "live_epochs_done": int(live_epochs_done),
        "telemetry_jobs": int(telemetry_jobs),
        "telemetry_dirs": int(telemetry_dirs),
    }


def main() -> None:
    args = parse_args()
    if getattr(args, "include_vanilla", None) is None:
        args.include_vanilla = _truthy(os.environ.get("INCLUDE_VANILLA"), default=True)
    configure_root_logging()
    logging_context_state: Dict[str, Any] = {
        "opf_gpu_running": 0,
        "opf_cpu_running": 0,
        "opf_gpu_vram_used_mib": 0.0,
        "opf_gpu_vram_total_mib": 0.0,
        "opf_host_ram_used_mib": 0.0,
        "opf_host_ram_total_mib": 0.0,
    }
    set_logging_context_provider(lambda: logging_context_state)
    validate_args(args)
    bootstrap_runtime("opf_massive_suite_runner")
    if not bool(getattr(args, "cpu_only", False)) and not _parent_cpu_masked():
        enable_optional_gpu_accelerators()
    logger = logging.getLogger(__name__)

    run_root = Path(args.run_root) if args.run_root else Path(args.results_dir) / "run_root"
    results_root = Path(args.results_dir)
    _ensure_dir(run_root)
    _ensure_dir(results_root)
    _ensure_dir(run_root / "jobs")

    seeded = seed_run_root_from_source(run_root, args.source_run_root)

    env_defaults = {
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", str(args.gpu_device_index)),
        "TABULAR_CHILD_SHARED_CPU": os.environ.get("TABULAR_CHILD_SHARED_CPU", "0"),
        "TABULAR_CPU_WORKERS": os.environ.get("TABULAR_CPU_WORKERS", "0"),
    }
    for key, value in env_defaults.items():
        os.environ[key] = value

    jobs = build_suite_jobs(args, run_root, results_root)
    write_manifest(run_root, args, jobs)

    run_metadata = {
        "run_root": str(run_root),
        "results_dir": str(results_root),
        "source_run_root": args.source_run_root,
        "seed": int(args.seed),
        "seed_strategy": "sha256(base_seed|mode|case|repeat|phase|family|head_variant|depth|width)",
        "created_at": _now_stamp(),
        "scheduler": args.scheduler,
        "surface": args.surface,
        "tasks": args.tasks,
        "cases": [case.strip() for case in str(args.cases).split(",") if case.strip()],
        "repeat_count": int(args.repeat_count),
        "max_retries": int(args.max_retries),
        "effective_repeat_count": max(int(args.repeat_count), int(args.prediction_repeats), int(args.missing_present_task_repeats), 1),
        "head_variants": _normalize_head_variants(getattr(args, "head_variants", None)),
        "concurrency": resolve_concurrency(args),
        "max_active_jobs": int(args.max_active_jobs),
        "max_active_gpu_jobs": int(args.max_active_gpu_jobs),
        "job_count": len(jobs),
        "host_ram_stop_avail_mib": float(args.host_ram_stop_avail_mib),
        "host_ram_resume_avail_mib": float(args.host_ram_resume_avail_mib),
        "gpu_stop_avail_mib": float(args.gpu_stop_avail_mib),
        "gpu_resume_avail_mib": float(args.gpu_resume_avail_mib),
        "launch_host_reserve_mib": float(args.launch_host_reserve_mib),
        "launch_gpu_reserve_mib": float(args.launch_gpu_reserve_mib),
        "gpu_child_gate_close_buffer_mib": float(args.gpu_child_gate_close_buffer_mib),
        "host_child_gate_close_buffer_mib": float(args.host_child_gate_close_buffer_mib),
        "pressure_poll_interval_sec": float(args.pressure_poll_interval_sec),
        "post_launch_sample_delay_sec": float(args.post_launch_sample_delay_sec),
        "batch_backoff_factor": float(args.batch_backoff_factor),
        "gpu_device_index": int(args.gpu_device_index),
        "include_vanilla": bool(args.include_vanilla),
        "include_width_only": bool(args.include_width_only),
        "include_width_to_depth": bool(args.include_width_to_depth),
        "param_band": args.param_band,
        "legacy_architecture_grid": bool(args.legacy_architecture_grid),
        "include_small_stl": bool(args.include_small_stl),
        "vanilla_depths": [int(item) for item in str(args.vanilla_depths).split(",") if item.strip()],
        "vanilla_widths": [int(item) for item in str(args.vanilla_widths).split(",") if item.strip()],
        "vanilla_model": args.vanilla_model,
        "max_epochs_vanilla": int(args.max_epochs_vanilla),
        "missing_present_task_repeats": int(args.missing_present_task_repeats),
        "prediction_repeats": int(args.prediction_repeats),
    }
    run_metadata["manifest_signature"] = _load_json(run_root / "manifest.json", default={}).get("signature")
    run_metadata["seeded_from_source"] = dict(seeded)
    _atomic_write_json(run_root / "run_metadata.json", run_metadata)

    if args.dry_run:
        _atomic_write_json(run_root / "dry_run_manifest.json", [dataclasses.asdict(job) for job in jobs])
        print(json.dumps(run_metadata, indent=2, sort_keys=True))
        return

    progress_path = run_root / "run_progress.csv"
    batch_state = normalize_dual_gate_state(load_batch_backoff_state(run_root))
    # Each invocation starts from a clean scheduler baseline. Job-level
    # resumability stays intact, but transient backoff/gate state must not leak
    # across relaunches or fresh runs.
    batch_state["batch_scale"] = 1.0
    batch_state["pressure_backoff_pending"] = False
    batch_state["last_backoff_reason"] = None
    batch_state["last_backoff_at"] = None
    batch_state["gpu_launches_enabled"] = True
    batch_state["cpu_launches_enabled"] = True
    batch_state["gpu_gate_closed"] = False
    batch_state["host_gate_closed"] = False
    batch_state["gpu_gate_lock_job_id"] = None
    batch_state["host_gate_lock_job_id"] = None
    batch_state["gpu_hold_until"] = 0.0
    batch_state["cpu_hold_until"] = 0.0
    batch_state["gpu_peak_vram_mib"] = 0.0
    batch_state["cpu_peak_host_mib"] = 0.0
    batch_state["last_child_gpu_vram_mib"] = 0.0
    batch_state["last_child_host_rss_mib"] = 0.0
    batch_state["last_child_gpu_remaining_mib"] = 0.0
    batch_state["last_child_host_remaining_mib"] = 0.0
    batch_state["last_child_job_id"] = None
    batch_state["external_host_non_python_peak_mib"] = 0.0
    batch_state["external_gpu_non_python_peak_mib"] = 0.0
    batch_state["last_external_host_non_python_mib"] = 0.0
    batch_state["last_external_gpu_non_python_mib"] = 0.0
    batch_state["external_host_non_python_pids"] = []
    batch_state["external_gpu_non_python_pids"] = []
    batch_state["gpu_gate_close_buffer_mib"] = float(args.gpu_child_gate_close_buffer_mib)
    batch_state["host_gate_close_buffer_mib"] = float(args.host_child_gate_close_buffer_mib)
    update_batch_backoff_state(run_root, batch_state)
    gpu_launches_enabled = bool(batch_state.get("gpu_launches_enabled", True))
    cpu_launches_enabled = bool(batch_state.get("cpu_launches_enabled", True))
    gpu_gate_closed = bool(batch_state.get("gpu_gate_closed", False))
    host_gate_closed = bool(batch_state.get("host_gate_closed", False))
    gpu_gate_lock_job_id = batch_state.get("gpu_gate_lock_job_id")
    host_gate_lock_job_id = batch_state.get("host_gate_lock_job_id")
    gpu_hold_until = float(batch_state.get("gpu_hold_until", 0.0))
    cpu_hold_until = float(batch_state.get("cpu_hold_until", 0.0))
    gpu_peak_vram_mib = float(batch_state.get("gpu_peak_vram_mib", 0.0))
    cpu_peak_host_mib = float(batch_state.get("cpu_peak_host_mib", 0.0))
    last_child_gpu_vram_mib = float(batch_state.get("last_child_gpu_vram_mib", 0.0))
    last_child_host_rss_mib = float(batch_state.get("last_child_host_rss_mib", 0.0))
    last_child_gpu_remaining_mib = float(batch_state.get("last_child_gpu_remaining_mib", 0.0))
    last_child_host_remaining_mib = float(batch_state.get("last_child_host_remaining_mib", 0.0))
    last_child_job_id = batch_state.get("last_child_job_id")
    external_host_non_python_peak_mib = float(batch_state.get("external_host_non_python_peak_mib", 0.0))
    external_gpu_non_python_peak_mib = float(batch_state.get("external_gpu_non_python_peak_mib", 0.0))
    last_external_host_non_python_mib = float(batch_state.get("last_external_host_non_python_mib", 0.0))
    last_external_gpu_non_python_mib = float(batch_state.get("last_external_gpu_non_python_mib", 0.0))
    external_host_non_python_pids = set(int(pid) for pid in batch_state.get("external_host_non_python_pids", []) or [])
    external_gpu_non_python_pids = set(int(pid) for pid in batch_state.get("external_gpu_non_python_pids", []) or [])
    pressure_backoff_pending = bool(batch_state.get("pressure_backoff_pending", False))
    pressure_backoff_reason = batch_state.get("last_backoff_reason")

    runtimes: Dict[str, JobRuntime] = {}
    queue: Deque[JobRuntime] = deque()
    active: List[JobRuntime] = []
    suspended: List[JobRuntime] = []
    completed_jobs = 0
    failed_attempts = 0
    terminal_failed_jobs = 0
    retried_jobs = 0

    for spec in jobs:
        runtime = load_existing_runtime(run_root, spec)
        runtimes[spec.job_id] = runtime
        state = load_job_state(runtime.state_path)
        if state.get("status") == "completed" and _safe_int(state.get("last_exit_code"), 0) == 0:
            completed_jobs += 1
            continue
        if is_terminal_inexact_resume_failure(state):
            terminal_failed_jobs += 1
            continue
        if runtime.status == "running":
            active.append(runtime)
            continue
        if runtime.status == "paused":
            suspended.append(runtime)
            continue
        queue.append(runtime)

    last_status_log_ts = 0.0
    last_resource_snapshot: Dict[str, Any] = {}
    shutdown_requested = False

    def _shutdown_all_runtimes(reason: str) -> None:
        nonlocal shutdown_requested
        if shutdown_requested:
            return
        shutdown_requested = True
        logger.warning("[INTERRUPT] Shutdown requested by %s. Terminating active and suspended jobs...", reason)
        for runtime in list(active) + list(suspended):
            if runtime.proc is not None and runtime.proc.poll() is None:
                try:
                    _terminate_process_tree(runtime.proc)
                except Exception:
                    pass
            _close_runtime_log_handle(runtime)
            runtime.proc = None
            runtime.last_exit_code = None
            runtime.ended_at = None
            runtime.paused = runtime in suspended
            runtime.pause_reason = runtime.pause_reason if runtime.paused else ""
            save_job_state(
                runtime.state_path,
                {
                    **runtime.to_state(),
                    "status": "paused" if runtime.paused else "queued",
                    "pid": None,
                    "last_exit_code": None,
                    "ended_at": None,
                    "reason": f"interrupted:{reason}",
                    "updated_at": _now_stamp(),
                },
            )
            write_compat_state_views(run_root, runtime.spec)
        try:
            _save_backoff_state()
        except Exception:
            pass

    def _handle_shutdown(signum, frame):
        try:
            signame = signal.Signals(signum).name
        except Exception:
            signame = str(signum)
        _shutdown_all_runtimes(signame)
        raise SystemExit(130)

    def _save_backoff_state() -> None:
        nonlocal gpu_gate_closed, host_gate_closed, gpu_gate_lock_job_id, host_gate_lock_job_id
        nonlocal last_child_gpu_vram_mib, last_child_host_rss_mib, last_child_gpu_remaining_mib, last_child_host_remaining_mib
        nonlocal last_child_job_id, external_host_non_python_peak_mib, external_gpu_non_python_peak_mib
        nonlocal last_external_host_non_python_mib, last_external_gpu_non_python_mib
        nonlocal external_host_non_python_pids, external_gpu_non_python_pids
        batch_state.update(
            {
                "batch_scale": _safe_float(batch_state.get("batch_scale"), 1.0),
                "pressure_backoff_pending": bool(pressure_backoff_pending),
                "last_backoff_reason": pressure_backoff_reason,
                "last_backoff_at": batch_state.get("last_backoff_at"),
                "gpu_launches_enabled": bool(gpu_launches_enabled),
                "cpu_launches_enabled": bool(cpu_launches_enabled),
                "gpu_gate_closed": bool(gpu_gate_closed),
                "host_gate_closed": bool(host_gate_closed),
                "gpu_gate_lock_job_id": gpu_gate_lock_job_id,
                "host_gate_lock_job_id": host_gate_lock_job_id,
                "gpu_gate_close_buffer_mib": float(args.gpu_child_gate_close_buffer_mib),
                "host_gate_close_buffer_mib": float(args.host_child_gate_close_buffer_mib),
                "gpu_hold_until": float(gpu_hold_until),
                "cpu_hold_until": float(cpu_hold_until),
                "gpu_peak_vram_mib": float(gpu_peak_vram_mib),
                "cpu_peak_host_mib": float(cpu_peak_host_mib),
                "last_child_gpu_vram_mib": float(last_child_gpu_vram_mib),
                "last_child_host_rss_mib": float(last_child_host_rss_mib),
                "last_child_gpu_remaining_mib": float(last_child_gpu_remaining_mib),
                "last_child_host_remaining_mib": float(last_child_host_remaining_mib),
                "last_child_job_id": last_child_job_id,
                "external_host_non_python_peak_mib": float(external_host_non_python_peak_mib),
                "external_gpu_non_python_peak_mib": float(external_gpu_non_python_peak_mib),
                "last_external_host_non_python_mib": float(last_external_host_non_python_mib),
                "last_external_gpu_non_python_mib": float(last_external_gpu_non_python_mib),
                "external_host_non_python_pids": sorted(int(pid) for pid in external_host_non_python_pids),
                "external_gpu_non_python_pids": sorted(int(pid) for pid in external_gpu_non_python_pids),
            }
        )
        update_batch_backoff_state(run_root, batch_state)

    def _apply_emergency_batch_backoff(reason: str) -> None:
        nonlocal pressure_backoff_pending, pressure_backoff_reason
        previous_scale = _safe_float(batch_state.get("batch_scale"), 1.0)
        next_scale = max(1e-6, previous_scale * float(args.batch_backoff_factor))
        if next_scale >= previous_scale:
            return
        pressure_backoff_pending = False
        pressure_backoff_reason = reason
        batch_state["batch_scale"] = float(next_scale)
        batch_state["pressure_backoff_pending"] = False
        batch_state["last_backoff_reason"] = reason
        batch_state["last_backoff_at"] = _now_stamp()
        batch_state["gpu_launches_enabled"] = bool(gpu_launches_enabled)
        batch_state["cpu_launches_enabled"] = bool(cpu_launches_enabled)
        batch_state["gpu_hold_until"] = float(gpu_hold_until)
        batch_state["cpu_hold_until"] = float(cpu_hold_until)
        batch_state["gpu_peak_vram_mib"] = float(gpu_peak_vram_mib)
        batch_state["cpu_peak_host_mib"] = float(cpu_peak_host_mib)
        update_batch_backoff_state(run_root, batch_state)
        _save_backoff_state()

    def record_event(event: str, runtime: JobRuntime, reason: str = "", exit_code: int = 0) -> None:
        logger.info(
            "[%s] %s case=%s repeat=%s phase=%s family=%s depth=%s width=%s attempt=%s device=%s reason=%s exit=%s",
            event.upper(),
            runtime.spec.job_id,
            runtime.spec.case_name,
            runtime.spec.repeat_index,
            runtime.spec.phase,
            runtime.spec.family,
            runtime.spec.depth,
            runtime.spec.width,
            runtime.attempt,
            runtime.device_mode,
            reason or "-",
            exit_code,
        )
        _write_csv_row(
            progress_path,
            [
                "timestamp",
                "event",
                "job_id",
                "phase",
                "family",
                "case_name",
                "repeat_index",
                "depth",
                "width",
                "attempt",
                "device_mode",
                "exit_code",
                "reason",
                "results_dir",
                "resource_snapshot",
            ],
            {
                "timestamp": _now_stamp(),
                "event": event,
                "job_id": runtime.spec.job_id,
                "phase": runtime.spec.phase,
                "family": runtime.spec.family,
                "case_name": runtime.spec.case_name,
                "repeat_index": runtime.spec.repeat_index,
                "depth": runtime.spec.depth,
                "width": runtime.spec.width,
                "attempt": runtime.attempt,
                "device_mode": runtime.device_mode,
                "exit_code": exit_code,
                "reason": reason,
                "results_dir": runtime.spec.results_dir,
                "resource_snapshot": json.dumps(last_resource_snapshot, sort_keys=True),
            },
        )
        write_compat_state_views(run_root, runtime.spec)

    def release_gate_locks_on_completion(runtime: JobRuntime) -> None:
        nonlocal gpu_gate_closed, host_gate_closed, gpu_gate_lock_job_id, host_gate_lock_job_id
        nonlocal gpu_launches_enabled, cpu_launches_enabled
        released: List[str] = []
        # Completion is keyed off the job's device mode, not strict lock ownership.
        # GPU jobs release both admission lanes because they can consume host RAM
        # and VRAM. CPU jobs release only the host/RAM lane.
        if runtime.device_mode == "gpu":
            gpu_gate_closed = False
            gpu_gate_lock_job_id = None
            gpu_launches_enabled = True
            host_gate_closed = False
            host_gate_lock_job_id = None
            cpu_launches_enabled = True
            released.extend(["gpu", "host"])
        elif runtime.device_mode == "cpu":
            host_gate_closed = False
            host_gate_lock_job_id = None
            cpu_launches_enabled = True
            released.append("host")
        if released:
            logger.info(
                "[GATE] admission reopened job=%s released=%s",
                runtime.spec.job_id,
                ",".join(released),
            )

    for _sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT):
        try:
            signal.signal(_sig, _handle_shutdown)
        except Exception:
            pass

    def log_status(force: bool = False) -> None:
        nonlocal last_status_log_ts
        now_ts = time.time()
        if not force and (now_ts - last_status_log_ts) < 30.0:
            return
        last_status_log_ts = now_ts
        telemetry_summary = summarize_suite_runtime_progress(run_root, jobs)
        logger.info(
            "[STATUS] queued=%d active=%d suspended=%d completed=%d failed_attempts=%d terminal_failed=%d retried=%d batches_done=%d epochs_done=%d live_batches_done=%d live_epochs_done=%d telemetry_jobs=%d telemetry_dirs=%d batch_scale=%.2f gpu_launches=%s cpu_launches=%s gpu_gate_closed=%s host_gate_closed=%s gpu_hold=%.1fs cpu_hold=%.1fs host_avail=%s host_rss=%s gpu_free=%s gpu_used=%s last_child_rss=%s last_child_vram=%s launch_cap=%s",
            len(queue),
            len(active),
            len(suspended),
            completed_jobs,
            failed_attempts,
            terminal_failed_jobs,
            retried_jobs,
            telemetry_summary["batches_done"],
            telemetry_summary["epochs_done"],
            telemetry_summary["live_batches_done"],
            telemetry_summary["live_epochs_done"],
            telemetry_summary["telemetry_jobs"],
            telemetry_summary["telemetry_dirs"],
            _safe_float(batch_state.get("batch_scale"), 1.0),
            bool(gpu_launches_enabled),
            bool(cpu_launches_enabled),
            bool(gpu_gate_closed),
            bool(host_gate_closed),
            max(0.0, float(gpu_hold_until - now_ts)),
            max(0.0, float(cpu_hold_until - now_ts)),
            last_resource_snapshot.get("host_available_effective_mib", "-"),
            last_resource_snapshot.get("active_rss_mib", "-"),
            last_resource_snapshot.get("gpu_free_effective_mib", "-"),
            last_resource_snapshot.get("gpu_used_effective_mib", "-"),
            last_resource_snapshot.get("last_child_host_rss_mib", "-"),
            last_resource_snapshot.get("last_child_gpu_vram_mib", "-"),
            last_resource_snapshot.get("launch_capacity", "-"),
        )

    try:
        while queue or active or suspended:
            log_status()
            concurrency = resolve_concurrency(args)
            effective_batch_scale = _safe_float(batch_state.get("batch_scale"), 1.0)
            launch_budget = max(1, int(math.ceil(concurrency * effective_batch_scale)))
            host_sample = sample_host_memory()
            gpu_sample = sample_gpu_memory(int(args.gpu_device_index))

            active_pids = current_pids(active)
            if active_pids:
                try:
                    active_rss = sum(psutil.Process(pid).memory_info().rss for pid in active_pids) / (1024 * 1024)
                except Exception:
                    active_rss = 0.0
            else:
                active_rss = 0.0

            # `psutil.virtual_memory().available` already reflects live process RSS.
            # Do not subtract active RSS again or paused jobs will appear to free RAM
            # instantly, which causes host pressure pause/resume ping-pong.
            host_available_effective = int(host_sample.available_mib)
            gpu_free_effective = gpu_sample.free_mib if gpu_sample is not None else None
            gpu_used_effective = None if gpu_sample is None else max(0.0, float(gpu_sample.total_mib - gpu_free_effective))
            host_used_effective = float(host_sample.total_mib - host_available_effective)
            active_gpu_count = sum(1 for job in active if job.device_mode == "gpu" and job.proc is not None and job.proc.poll() is None)
            active_cpu_count = sum(1 for job in active if job.device_mode == "cpu" and job.proc is not None and job.proc.poll() is None)
            active_count = sum(1 for job in active if job.proc is not None and job.proc.poll() is None)
            active_and_suspended_pids = current_pids(active) + current_pids(suspended)
            launch_capacity = resolve_launch_capacity(args, float(host_available_effective), gpu_free_effective)
            if active_count > 0 and resolve_concurrency(args) != UNBOUNDED_CONCURRENCY:
                launch_capacity = max(1, min(launch_capacity, active_count + 1))
            gpu_pressure = gpu_free_effective is not None and gpu_free_effective < float(args.gpu_stop_avail_mib)
            host_pressure = host_available_effective < float(args.host_ram_stop_avail_mib)
            # Swap occupancy is telemetry only. Admission and pause decisions use
            # available host RAM, GPU VRAM, and active-child estimates.
            now = time.time()
            if queue and not gpu_pressure and not host_pressure and not gpu_gate_closed and not host_gate_closed:
                if now >= gpu_hold_until and now >= cpu_hold_until:
                    launch_budget = max(launch_budget, min(concurrency, active_count + 1))
            latest_runtime = _latest_active_runtime(active)
            latest_gpu_runtime = _latest_active_runtime(active, device_mode="gpu")
            sampled_child_host_rss = 0.0
            sampled_child_gpu_vram = 0.0
            sampled_child_host_remaining = 0.0
            sampled_child_gpu_remaining = 0.0
            sampled_child_job_id = None
            if latest_runtime is not None:
                sampled_child_host_rss, sampled_child_gpu_vram, _ = sample_runtime_child_memory(
                    latest_runtime,
                    int(args.gpu_device_index),
                )
                sampled_child_host_remaining = max(0.0, float(host_available_effective) - float(sampled_child_host_rss))
                sampled_child_gpu_remaining = max(0.0, float(gpu_free_effective) - float(sampled_child_gpu_vram)) if gpu_free_effective is not None else 0.0
                sampled_child_job_id = latest_runtime.spec.job_id
            elif latest_gpu_runtime is not None:
                sampled_child_host_rss, sampled_child_gpu_vram, _ = sample_runtime_child_memory(
                    latest_gpu_runtime,
                    int(args.gpu_device_index),
                )
                sampled_child_host_remaining = max(0.0, float(host_available_effective) - float(sampled_child_host_rss))
                sampled_child_gpu_remaining = max(0.0, float(gpu_free_effective) - float(sampled_child_gpu_vram)) if gpu_free_effective is not None else 0.0
                sampled_child_job_id = latest_gpu_runtime.spec.job_id

            previous_external_host_non_python_pids = set(external_host_non_python_pids)
            previous_external_gpu_non_python_pids = set(external_gpu_non_python_pids)
            previous_external_host_non_python_mib = float(last_external_host_non_python_mib)
            previous_external_gpu_non_python_mib = float(last_external_gpu_non_python_mib)

            external_host_non_python_mib, external_host_non_python_pid_list = sample_non_python_host_memory(
                exclude_pids=active_and_suspended_pids
            )
            external_gpu_non_python_mib, external_gpu_non_python_pid_list, _ = sample_non_python_gpu_memory(
                int(args.gpu_device_index),
                exclude_pids=active_and_suspended_pids,
            )
            if external_host_non_python_mib > external_host_non_python_peak_mib:
                external_host_non_python_peak_mib = float(external_host_non_python_mib)
            if external_gpu_non_python_mib > external_gpu_non_python_peak_mib:
                external_gpu_non_python_peak_mib = float(external_gpu_non_python_mib)
            external_drop_threshold = float(args.external_memory_drop_reopen_mib)
            host_external_drop = (
                previous_external_host_non_python_mib > 0.0
                and (previous_external_host_non_python_mib - float(external_host_non_python_mib)) >= external_drop_threshold
                and len(previous_external_host_non_python_pids - set(external_host_non_python_pid_list)) > 0
            )
            gpu_external_drop = (
                previous_external_gpu_non_python_mib > 0.0
                and (previous_external_gpu_non_python_mib - float(external_gpu_non_python_mib)) >= external_drop_threshold
                and len(previous_external_gpu_non_python_pids - set(external_gpu_non_python_pid_list)) > 0
            )

            last_resource_snapshot = {
                "timestamp": _now_stamp(),
                "host_total_mib": int(host_sample.total_mib),
                "host_available_mib": int(host_sample.available_mib),
                "host_available_effective_mib": int(host_available_effective),
                "host_used_effective_mib": int(host_used_effective),
                "active_rss_mib": int(active_rss),
                "gpu_total_mib": int(gpu_sample.total_mib) if gpu_sample is not None else None,
                "gpu_free_effective_mib": int(gpu_free_effective) if gpu_free_effective is not None else None,
                "gpu_used_effective_mib": int(gpu_used_effective) if gpu_used_effective is not None else None,
                "host_swap_used_pct": float(host_sample.swap_used_pct),
                "active_count": int(active_count),
                "active_gpu_count": int(active_gpu_count),
                "active_cpu_count": int(active_cpu_count),
                "launch_capacity": int(launch_capacity),
                "gpu_pressure": bool(gpu_pressure),
                "host_pressure": bool(host_pressure),
                "gpu_gate_closed": bool(gpu_gate_closed),
                "host_gate_closed": bool(host_gate_closed),
                "gpu_gate_lock_job_id": gpu_gate_lock_job_id,
                "host_gate_lock_job_id": host_gate_lock_job_id,
                "last_child_job_id": sampled_child_job_id,
                "last_child_host_rss_mib": float(sampled_child_host_rss),
                "last_child_gpu_vram_mib": float(sampled_child_gpu_vram),
                "last_child_host_remaining_mib": float(sampled_child_host_remaining),
                "last_child_gpu_remaining_mib": float(sampled_child_gpu_remaining),
                "external_host_non_python_mib": float(external_host_non_python_mib),
                "external_gpu_non_python_mib": float(external_gpu_non_python_mib),
                "external_host_non_python_peak_mib": float(external_host_non_python_peak_mib),
                "external_gpu_non_python_peak_mib": float(external_gpu_non_python_peak_mib),
                "last_external_host_non_python_mib": float(external_host_non_python_mib),
                "last_external_gpu_non_python_mib": float(external_gpu_non_python_mib),
                "external_host_non_python_pids": external_host_non_python_pid_list,
                "external_gpu_non_python_pids": external_gpu_non_python_pid_list,
            }

            last_child_host_rss_mib = float(sampled_child_host_rss)
            last_child_gpu_vram_mib = float(sampled_child_gpu_vram)
            last_child_host_remaining_mib = float(sampled_child_host_remaining)
            last_child_gpu_remaining_mib = float(sampled_child_gpu_remaining)
            last_child_job_id = sampled_child_job_id
            external_host_non_python_pids = set(external_host_non_python_pid_list)
            external_gpu_non_python_pids = set(external_gpu_non_python_pid_list)
            last_external_host_non_python_mib = float(external_host_non_python_mib)
            last_external_gpu_non_python_mib = float(external_gpu_non_python_mib)

            logging_context_state.update(
                {
                    "opf_gpu_running": int(active_gpu_count),
                    "opf_cpu_running": int(active_cpu_count),
                    "opf_gpu_vram_used_mib": float(gpu_used_effective or 0.0),
                    "opf_gpu_vram_total_mib": float(gpu_sample.total_mib if gpu_sample is not None else 0.0),
                    "opf_host_ram_used_mib": float(host_used_effective),
                    "opf_host_ram_total_mib": float(host_sample.total_mib),
                }
            )

            if latest_gpu_runtime is not None and gpu_free_effective is not None:
                gpu_buffer = float(args.gpu_child_gate_close_buffer_mib)
                gpu_child_pressure = (last_child_gpu_vram_mib + gpu_buffer) > float(gpu_free_effective)
                if not gpu_gate_closed and (gpu_pressure or gpu_child_pressure):
                    gpu_gate_closed = True
                    gpu_gate_lock_job_id = latest_gpu_runtime.spec.job_id
                    gpu_launches_enabled = False
                    logger.info(
                        "[GATE] gpu admission closed job=%s child_vram=%.1f remaining_vram=%.1f buffer=%.1f gpu_pressure=%s child_pressure=%s",
                        latest_gpu_runtime.spec.job_id,
                        last_child_gpu_vram_mib,
                        float(gpu_free_effective),
                        gpu_buffer,
                        bool(gpu_pressure),
                        bool(gpu_child_pressure),
                    )

            if latest_runtime is not None:
                host_buffer = float(args.host_child_gate_close_buffer_mib)
                host_child_pressure = (last_child_host_rss_mib + host_buffer) > float(host_available_effective)
                if not host_gate_closed and (host_pressure or host_child_pressure):
                    host_gate_closed = True
                    host_gate_lock_job_id = latest_runtime.spec.job_id
                    cpu_launches_enabled = False
                    logger.info(
                        "[GATE] host admission closed job=%s child_rss=%.1f remaining_host=%.1f buffer=%.1f host_pressure=%s child_pressure=%s",
                        latest_runtime.spec.job_id,
                        last_child_host_rss_mib,
                        float(host_available_effective),
                        host_buffer,
                        bool(host_pressure),
                        bool(host_child_pressure),
                    )
            if gpu_gate_closed and gpu_external_drop:
                gpu_gate_closed = False
                gpu_gate_lock_job_id = None
                gpu_launches_enabled = True
                logger.info(
                    "[GATE] gpu admission reopened by external non-python memory drop host_free=%s gpu_free=%s",
                    int(host_available_effective),
                    int(gpu_free_effective) if gpu_free_effective is not None else -1,
                )
            if host_gate_closed and host_external_drop:
                host_gate_closed = False
                host_gate_lock_job_id = None
                cpu_launches_enabled = True
                logger.info(
                    "[GATE] host admission reopened by external non-python memory drop host_free=%d",
                    int(host_available_effective),
                )
            _save_backoff_state()

            # Resume suspended jobs when the relevant gate reopens.
            for runtime in list(suspended):
                pause_reason = str(runtime.pause_reason or "")
                reopened = pressure_pause_reopened(
                    pause_reason=pause_reason,
                    device_mode=runtime.device_mode,
                    host_available_mib=host_available_effective,
                    gpu_free_mib=gpu_free_effective,
                    host_used_mib=host_used_effective,
                    gpu_used_mib=gpu_used_effective,
                    host_resume_avail_mib=args.host_ram_resume_avail_mib,
                    gpu_resume_avail_mib=args.gpu_resume_avail_mib,
                    host_peak_used_mib=cpu_peak_host_mib,
                    gpu_peak_used_mib=gpu_peak_vram_mib,
                    external_drop_threshold_mib=external_drop_threshold,
                    external_host_drop_mib=(
                        previous_external_host_non_python_mib - float(external_host_non_python_mib)
                        if host_external_drop
                        else 0.0
                    ),
                    external_gpu_drop_mib=(
                        previous_external_gpu_non_python_mib - float(external_gpu_non_python_mib)
                        if gpu_external_drop
                        else 0.0
                    ),
                )
                if not reopened:
                    continue
                resume_device = runtime.device_mode
                if resume_device == "gpu":
                    if gpu_free_effective is None or gpu_free_effective < float(args.gpu_resume_avail_mib):
                        continue
                else:
                    if host_available_effective < float(args.host_ram_resume_avail_mib):
                        continue
                if resume_device is None:
                    continue
                try:
                    _resume_runtime_from_pause(
                        runtime,
                        args,
                        concurrency,
                        run_root,
                        device_mode=resume_device,
                        active=active,
                        launch_context={
                            "resumed_from_pause": True,
                            "pause_reason": pause_reason,
                            "resume_device": resume_device,
                            "checkpoint_last": str(job_directory(run_root, runtime.spec) / "checkpoint_last.pt"),
                            "checkpoint_best": str(job_directory(run_root, runtime.spec) / "checkpoint_best.pt"),
                            "checkpoint_best_batch": str(job_directory(run_root, runtime.spec) / "checkpoint_best_batch.pt"),
                        },
                    )
                    suspended.remove(runtime)
                    record_event("resume", runtime, reason="gate_reopened", exit_code=0)
                except Exception as exc:
                    logger.warning(
                        "[RESUME] relaunch failed job=%s attempt=%s reason=%s error=%s; job remains paused",
                        runtime.spec.job_id,
                        runtime.attempt,
                        pause_reason,
                        exc,
                    )

            # Poll active jobs and finalize exits.
            for runtime in list(active):
                proc = runtime.proc
                if proc is None:
                    # OOM handling can suspend a peer during this snapshot walk.
                    if runtime in active:
                        active.remove(runtime)
                    continue
                rc = proc.poll()
                if rc is None:
                    continue
                if _requeue_recovered_runtime_after_exit(runtime, active, queue):
                    logger.warning(
                        "[RECOVER] child disappeared after orchestrator restart; requeued job=%s for checkpointed continuation",
                        runtime.spec.job_id,
                    )
                    continue
                if runtime in active:
                    active.remove(runtime)
                runtime.last_exit_code = int(rc)
                runtime.ended_at = _now_stamp()
                if rc == 0:
                    completed_jobs += 1
                    finalize_job(args, runtime, rc, runtime.log_path, progress_path, run_root, last_resource_snapshot, "complete", "")
                    release_gate_locks_on_completion(runtime)
                    pressure_backoff_pending = False
                    pressure_backoff_reason = None
                    if runtime.device_mode == "gpu":
                        gpu_launches_enabled = True
                    cpu_launches_enabled = True
                    batch_state["pressure_backoff_pending"] = False
                    batch_state["last_backoff_reason"] = None
                    batch_state["last_backoff_at"] = None
                    _save_backoff_state()
                    continue

                failed_attempts += 1
                inexact_resume = is_inexact_resume_log(runtime.log_path)
                gpu_oom = is_gpu_oom_log(runtime.log_path) or ("cuda" in str(rc).lower() if isinstance(rc, str) else False)
                oom = gpu_oom or is_memory_oom_log(runtime.log_path) or ("memoryerror" in str(rc).lower() if isinstance(rc, str) else False)
                oom_reason = "inexact_checkpoint_resume" if inexact_resume else ("cuda_oom" if gpu_oom else ("memory_oom" if oom else "nonzero_exit"))
                if gpu_oom and runtime.device_mode == "gpu":
                    runtime.gpu_pause_count += 1
                    if gpu_sample is not None:
                        gpu_peak_vram_mib = float(gpu_sample.total_mib - gpu_free_effective) if gpu_free_effective is not None else gpu_peak_vram_mib
                    peer_gpu_jobs = [
                        job for job in active
                        if job.device_mode == "gpu" and job.proc is not None and job.proc.poll() is None
                    ]
                    if peer_gpu_jobs:
                        peer_gpu_jobs.sort(key=_job_pressure_score, reverse=True)
                        peer = peer_gpu_jobs[0]
                        try:
                            if _suspend_runtime_for_pressure(
                                peer,
                                reason="peer_cuda_oom",
                                suspended=suspended,
                                active=active,
                                run_root=run_root,
                                priority_snapshot=last_resource_snapshot,
                            ):
                                record_event("pause", peer, reason="peer_cuda_oom", exit_code=0)
                        except Exception:
                            pass
                if not inexact_resume and should_retry(runtime, max_retries=int(args.max_retries)):
                    retried_jobs += 1
                    runtime.retry_count += 1
                    release_gate_locks_on_completion(runtime)
                    if runtime.log_handle is not None:
                        try:
                            runtime.log_handle.write(f"=== RETRY {runtime.spec.job_id} exit={rc} ===\n")
                            runtime.log_handle.flush()
                            runtime.log_handle.close()
                        except Exception:
                            pass
                        runtime.log_handle = None
                    runtime.proc = None
                    runtime.paused = False
                    runtime.pause_reason = ""
                    runtime.last_exit_code = None
                    save_job_state(
                        runtime.state_path,
                        {
                            **runtime.to_state(),
                            "status": "retrying",
                            "last_exit_code": int(rc),
                            "reason": oom_reason,
                            "updated_at": _now_stamp(),
                        },
                    )
                    record_event("retry", runtime, reason=oom_reason, exit_code=int(rc))
                    queue.append(runtime)
                    if oom:
                        # OOM reduces batch scale without permanently disabling
                        # an admission lane that has no guaranteed reopen event.
                        _apply_emergency_batch_backoff(oom_reason)
                    else:
                        pressure_backoff_pending = False
                        pressure_backoff_reason = None
                        batch_state["pressure_backoff_pending"] = False
                        batch_state["last_backoff_reason"] = None
                        batch_state["last_backoff_at"] = None
                        _save_backoff_state()
                else:
                    terminal_failed_jobs += 1
                    finalize_job(args, runtime, rc, runtime.log_path, progress_path, run_root, last_resource_snapshot, "fail", oom_reason)
                    release_gate_locks_on_completion(runtime)

            if gpu_pressure or host_pressure:
                if gpu_pressure and gpu_sample is not None and gpu_free_effective is not None:
                    gpu_peak_vram_mib = float(gpu_sample.total_mib - gpu_free_effective)
                if host_pressure:
                    cpu_peak_host_mib = float(host_sample.total_mib - host_available_effective)

                pressure_kind = "gpu" if gpu_pressure else "host"
                pressure_reasons = []
                if gpu_pressure:
                    pressure_reasons.append("gpu_memory_pressure")
                if host_pressure:
                    pressure_reasons.append("host_ram_pressure")
                pressure_reason = "+".join(pressure_reasons)
                if pressure_kind == "gpu":
                    candidates = [
                        (
                            sample_runtime_child_memory(runtime, int(args.gpu_device_index)),
                            runtime,
                        )
                        for runtime in active
                        if runtime.proc is not None
                        and runtime.proc.poll() is None
                        and not runtime.paused
                        and runtime.device_mode == "gpu"
                    ]
                    candidates.sort(
                        key=lambda item: (
                            float(item[0][1]),
                            float(item[0][0]),
                            _job_pressure_score(item[1]),
                        ),
                        reverse=True,
                    )
                else:
                    candidates = [
                        (
                            sample_runtime_child_memory(runtime, int(args.gpu_device_index)),
                            runtime,
                        )
                        for runtime in active
                        if runtime.proc is not None and runtime.proc.poll() is None and not runtime.paused
                    ]
                    candidates.sort(
                        key=lambda item: (
                            float(item[0][0]),
                            float(item[0][1]),
                            _job_pressure_score(item[1]),
                        ),
                        reverse=True,
                    )

                if candidates:
                    _, pressure_runtime = candidates[0]
                    if _suspend_runtime_for_pressure(
                        pressure_runtime,
                        reason=pressure_reason,
                        suspended=suspended,
                        active=active,
                        run_root=run_root,
                        priority_snapshot=last_resource_snapshot,
                    ):
                        pressure_backoff_pending = False
                        pressure_backoff_reason = None
                        batch_state["pressure_backoff_pending"] = False
                        batch_state["last_backoff_reason"] = None
                        batch_state["last_backoff_at"] = None
                        record_event("pause", pressure_runtime, reason=pressure_reason, exit_code=0)
                        logger.warning(
                            "[PRESSURE] paused job=%s reason=%s active=%d suspended=%d host_avail=%s host_rss=%s gpu_free=%s gpu_used=%s",
                            pressure_runtime.spec.job_id,
                            pressure_reason,
                            len(active),
                            len(suspended),
                            last_resource_snapshot.get("host_available_effective_mib", "-"),
                            last_resource_snapshot.get("active_rss_mib", "-"),
                            last_resource_snapshot.get("gpu_free_effective_mib", "-"),
                            last_resource_snapshot.get("gpu_used_effective_mib", "-"),
                        )
                _save_backoff_state()

            # Pressure handling and exit polling can change membership after
            # the first resource sample. Recompute live counts before making
            # admission decisions so active/suspended state is not double
            # counted for one scheduler cycle.
            active_gpu_count = sum(
                1 for job in active
                if job.device_mode == "gpu" and job.proc is not None and job.proc.poll() is None
            )
            active_cpu_count = sum(
                1 for job in active
                if job.device_mode == "cpu" and job.proc is not None and job.proc.poll() is None
            )
            active_count = sum(
                1 for job in active
                if job.proc is not None and job.proc.poll() is None
            )
            last_resource_snapshot.update(
                {
                    "active_count": int(active_count),
                    "active_gpu_count": int(active_gpu_count),
                    "active_cpu_count": int(active_cpu_count),
                }
            )

            # Launch additional jobs if the gates permit.
            launches_this_round = 0
            while queue:
                now = time.time()
                effective_batch_scale = _safe_float(batch_state.get("batch_scale"), 1.0)
                launch_budget = max(1, int(math.ceil(resolve_concurrency(args) * effective_batch_scale)))
                launch_budget = min(launch_budget, max(1, int(last_resource_snapshot.get("launch_capacity", launch_budget))))
                if active_count >= launch_budget:
                    break
                runtime = queue[0]
                if now < gpu_hold_until and now < cpu_hold_until:
                    break

                gpu_intended = (
                    not bool(getattr(args, "cpu_only", False))
                    and gpu_launches_enabled
                    and not gpu_gate_closed
                    and not host_gate_closed
                    and gpu_sample is not None
                    and gpu_free_effective is not None
                    and gpu_free_effective >= float(args.gpu_resume_avail_mib)
                    and (int(args.max_active_gpu_jobs) <= 0 or active_gpu_count < int(args.max_active_gpu_jobs))
                )
                host_launchable = (
                    cpu_launches_enabled
                    and not host_gate_closed
                    and host_available_effective >= float(args.host_ram_resume_avail_mib)
                )
                gpu_can_launch = (
                    gpu_intended
                    and now >= gpu_hold_until
                    and host_available_effective >= float(args.host_ram_resume_avail_mib)
                )
                cpu_can_launch = (
                    host_launchable
                    and now >= cpu_hold_until
                )
                device_mode = select_launch_device(
                    runtime,
                    args,
                    host_sample,
                    gpu_sample,
                    active_gpu_count,
                    active_cpu_count,
                    host_admission_enabled=cpu_can_launch,
                    gpu_admission_enabled=gpu_can_launch,
                )
                if device_mode is None:
                    break
                queue.popleft()
                launch_context = {
                    "timestamp": _now_stamp(),
                    "host_total_mib": int(host_sample.total_mib),
                    "host_available_mib": int(host_sample.available_mib),
                    "host_available_effective_mib": int(host_available_effective),
                    "host_used_effective_mib": int(host_used_effective),
                    "active_rss_mib": int(active_rss),
                    "gpu_total_mib": int(gpu_sample.total_mib) if gpu_sample is not None else None,
                    "gpu_free_effective_mib": int(gpu_free_effective) if gpu_free_effective is not None else None,
                    "gpu_used_effective_mib": int(gpu_used_effective) if gpu_used_effective is not None else None,
                    "active_count": int(active_count),
                    "active_gpu_count": int(active_gpu_count),
                    "active_cpu_count": int(active_cpu_count),
                    "launch_capacity": int(launch_budget),
                    "batch_scale": float(effective_batch_scale),
                    "host_ram_stop_avail_mib": float(args.host_ram_stop_avail_mib),
                    "host_ram_resume_avail_mib": float(args.host_ram_resume_avail_mib),
                    "gpu_stop_avail_mib": float(args.gpu_stop_avail_mib),
                    "gpu_resume_avail_mib": float(args.gpu_resume_avail_mib),
                    "gpu_child_gate_close_buffer_mib": float(args.gpu_child_gate_close_buffer_mib),
                    "host_child_gate_close_buffer_mib": float(args.host_child_gate_close_buffer_mib),
                    "last_child_job_id": sampled_child_job_id,
                    "last_child_host_rss_mib": float(sampled_child_host_rss),
                    "last_child_gpu_vram_mib": float(sampled_child_gpu_vram),
                    "last_child_host_remaining_mib": float(sampled_child_host_remaining),
                    "last_child_gpu_remaining_mib": float(sampled_child_gpu_remaining),
                    "gpu_gate_closed": bool(gpu_gate_closed),
                    "host_gate_closed": bool(host_gate_closed),
                }
                logger.info(
                    "[RESOURCE] launch=%s queue=%d active=%d cap=%d host_avail=%d host_rss=%d gpu_free=%s gpu_used=%s last_child_rss=%s last_child_vram=%s gpu_hold=%.1fs cpu_hold=%.1fs",
                    runtime.spec.job_id,
                    len(queue) + 1,
                    active_count,
                    launch_budget,
                    int(host_available_effective),
                    int(active_rss),
                    int(gpu_free_effective) if gpu_free_effective is not None else -1,
                    int(gpu_used_effective) if gpu_used_effective is not None else -1,
                    int(sampled_child_host_rss),
                    int(sampled_child_gpu_vram),
                    max(0.0, float(gpu_hold_until - now)),
                    max(0.0, float(cpu_hold_until - now)),
                )
                try:
                    launch_job(args, runtime, launch_budget, device_mode, run_root, launch_context=launch_context)
                except Exception as exc:
                    # Popen failures are launch-attempt failures, not scheduler
                    # failures. Requeue within the configured retry budget so a
                    # transient executable/filesystem/resource error cannot
                    # silently drop the dequeued job or abort the whole suite.
                    failed_attempts += 1
                    logger.exception("[LAUNCH] failed job=%s error=%s", runtime.spec.job_id, exc)
                    runtime.proc = None
                    _close_runtime_log_handle(runtime)
                    runtime.ended_at = _now_stamp()
                    runtime.last_exit_code = 1
                    if should_retry(runtime, max_retries=int(args.max_retries)):
                        retried_jobs += 1
                        runtime.retry_count += 1
                        runtime.last_exit_code = None
                        runtime.ended_at = None
                        save_job_state(
                            runtime.state_path,
                            {
                                **runtime.to_state(),
                                "status": "retrying",
                                "last_exit_code": 1,
                                "reason": "launch_error",
                                "updated_at": _now_stamp(),
                            },
                        )
                        record_event("retry", runtime, reason="launch_error", exit_code=1)
                        queue.append(runtime)
                    else:
                        terminal_failed_jobs += 1
                        finalize_job(
                            args,
                            runtime,
                            1,
                            runtime.log_path,
                            progress_path,
                            run_root,
                            last_resource_snapshot,
                            "fail",
                            "launch_error",
                        )
                    continue
                active.append(runtime)
                active_count += 1
                if device_mode == "gpu":
                    active_gpu_count += 1
                else:
                    active_cpu_count += 1
                launches_this_round += 1
                record_event("launch", runtime, reason=device_mode, exit_code=0)
                if device_mode == "gpu":
                    gpu_hold_until = now + float(args.post_launch_sample_delay_sec)
                else:
                    cpu_hold_until = now + float(args.post_launch_sample_delay_sec)
                _save_backoff_state()
                delay_sec = float(args.post_launch_sample_delay_sec)
                logger.info(
                    "[HOLD] launched=%s device=%s waiting=%.1fs before next resource poll",
                    runtime.spec.job_id,
                    device_mode,
                    delay_sec,
                )
                if delay_sec > 0:
                    time.sleep(delay_sec)
                break

            if launches_this_round > 0:
                batch_state["batch_scale"] = 1.0
                batch_state["pressure_backoff_pending"] = False
                batch_state["last_backoff_reason"] = None
                batch_state["last_backoff_at"] = None
                batch_state["gpu_launches_enabled"] = bool(gpu_launches_enabled)
                batch_state["cpu_launches_enabled"] = bool(cpu_launches_enabled)
                batch_state["gpu_hold_until"] = float(gpu_hold_until)
                batch_state["cpu_hold_until"] = float(cpu_hold_until)
                batch_state["gpu_peak_vram_mib"] = float(gpu_peak_vram_mib)
                batch_state["cpu_peak_host_mib"] = float(cpu_peak_host_mib)
                update_batch_backoff_state(run_root, batch_state)

            if not queue and not active and not suspended:
                break

            time.sleep(max(0.1, float(args.pressure_poll_interval_sec)))

    except KeyboardInterrupt:
        _shutdown_all_runtimes("KeyboardInterrupt")
        raise

    summary = render_summary(run_root, results_root, jobs, completed_jobs, terminal_failed_jobs, retried_jobs)
    summary["failed_attempts"] = int(failed_attempts)
    summary.update(summarize_suite_runtime_progress(run_root, jobs))
    # The final status is emitted after active jobs have been removed; keep the
    # logging context consistent with that terminal state.
    logging_context_state.update({"opf_gpu_running": 0, "opf_cpu_running": 0})
    log_status(force=True)
    _atomic_write_json(run_root / "final_report.json", summary)
    _atomic_write_json(run_root / "suite_summary.json", summary)
    summary_rows: List[Dict[str, Any]] = []
    for spec in jobs:
        state = load_job_state(job_directory(run_root, spec) / "job_state.json")
        summary_rows.append(
            {
                "phase": spec.phase,
                "family": spec.family,
                "case_name": spec.case_name,
                "repeat_index": spec.repeat_index,
                "depth": spec.depth,
                "width": spec.width,
                "status": state.get("status", "queued"),
                "attempt": state.get("attempt", 0),
                "last_exit_code": state.get("last_exit_code", ""),
                "results_dir": spec.results_dir,
            }
        )
    task_summary_path = run_root / "task_summary.csv"
    with task_summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "phase",
                "family",
                "case_name",
                "repeat_index",
                "depth",
                "width",
                "status",
                "attempt",
                "last_exit_code",
                "results_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except Exception:
            pass
    _atomic_write_text(
        run_root / "final_report.md",
        "\n".join(
            [
                f"# OPF Massive Suite Final Report",
                "",
                f"- Run root: `{summary['run_root']}`",
                f"- Results dir: `{summary['results_dir']}`",
                f"- Total jobs: `{summary['total_jobs']}`",
                f"- Completed jobs: `{summary['completed_jobs']}`",
                f"- Terminal failed jobs: `{summary['failed_jobs']}`",
                f"- Failed attempts: `{summary['failed_attempts']}`",
                f"- Retried jobs: `{summary['retried_jobs']}`",
                f"- All-job batch records done: `{summary['batches_done']}`",
                f"- All-job epoch records done: `{summary['epochs_done']}`",
                f"- Live-job batch records done: `{summary['live_batches_done']}`",
                f"- Live-job epoch records done: `{summary['live_epochs_done']}`",
                f"- Timestamp: `{summary['timestamp']}`",
            ]
        )
        + "\n",
    )
    _atomic_write_text(
        run_root / "suite_summary.md",
        "\n".join(
            [
                f"# OPF Massive Suite Summary",
                "",
                f"- Run root: `{summary['run_root']}`",
                f"- Results dir: `{summary['results_dir']}`",
                f"- Total jobs: `{summary['total_jobs']}`",
                f"- Completed jobs: `{summary['completed_jobs']}`",
                f"- Terminal failed jobs: `{summary['failed_jobs']}`",
                f"- Failed attempts: `{summary['failed_attempts']}`",
                f"- Retried jobs: `{summary['retried_jobs']}`",
                f"- All-job batch records done: `{summary['batches_done']}`",
                f"- All-job epoch records done: `{summary['epochs_done']}`",
                f"- Live-job batch records done: `{summary['live_batches_done']}`",
                f"- Live-job epoch records done: `{summary['live_epochs_done']}`",
            ]
        )
        + "\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Caught KeyboardInterrupt. Launcher-owned jobs were cleaned up and marked resumable.")
        sys.exit(130)
