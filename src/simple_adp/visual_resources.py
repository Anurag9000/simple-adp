"""Non-invasive OS/torch allocator measurements around a visual fit call.

RSS is measured for the whole Python process. VmHWM and CUDA allocator maxima
are *process-lifetime* high water marks, not resettable per-job peaks. This
module never modifies RNG, model mode, CUDA allocator counters, or checkpoints.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def _linux_process_rss_bytes() -> tuple[int | None, int | None]:
    """Read current and lifetime peak RSS from Linux /proc (kB -> bytes).

    /proc may be missing or restricted in non-Linux containers. An unavailable
    metric is unknown (None), never a fabricated zero.
    """
    try:
        lines = Path("/proc/self/status").read_text(encoding="ascii").splitlines()
        stats: dict[str, int] = {}
        for line in lines:
            if line.startswith(("VmRSS:", "VmHWM:")):
                label, numeric, units = line.split()
                if units != "kB":
                    return None, None
                stats[label[:-1]] = int(numeric) * 1024
        return stats.get("VmRSS"), stats.get("VmHWM")
    except (OSError, ValueError):
        return None, None


def visual_process_resource_snapshot(device: str | torch.device = "cpu") -> dict[str, object]:
    """Observe current CPU RSS and CUDA allocator counters without resetting them."""
    target = torch.device(device)
    rss, rss_high = _linux_process_rss_bytes()
    cuda_stats: dict[str, int] | None = None
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA resource snapshot requested without available CUDA")
        # Current and high-water CUDA tensor allocator bytes. Does not include
        # all driver/library workspaces and is *not* GPU device-wide VRAM.
        cuda_stats = {
            "allocated_bytes": int(torch.cuda.memory_allocated(target)),
            "reserved_bytes": int(torch.cuda.memory_reserved(target)),
            "lifetime_max_allocated_bytes": int(torch.cuda.max_memory_allocated(target)),
            "lifetime_max_reserved_bytes": int(torch.cuda.max_memory_reserved(target)),
        }
    return {
        "format": "simple-adp-visual-process-resource-snapshot-v1",
        "pid": int(os.getpid()),
        "device": str(target),
        "cpu_process_rss_bytes": rss,
        "cpu_process_lifetime_peak_rss_bytes": rss_high,
        "cuda_pytorch_allocator": cuda_stats,
        "scope": "Entire current process; lifetime maxima include prior work",
        "excludes": (
            "other processes; unsampled transients; external GPU driver allocations; "
            "model-specific attribution; full hardware memory usage"
        ),
    }


def visual_fit_resource_window(
    before: dict[str, Any], after: dict[str, Any],
) -> dict[str, object]:
    """Compare two true snapshots without asserting an exact interval peak.

    If the process high-water mark increases, its final value was reached in
    the interval; otherwise the interval maximum is unknown. Current RSS
    before/after is a lower bound on the interval's peak. This is not
    an OOM-prevention mechanism or complete training-memory profile.
    """
    for snap in (before, after):
        if snap.get("format") != "simple-adp-visual-process-resource-snapshot-v1":
            raise ValueError("unrecognized process resource snapshot")
    if before.get("pid") != after.get("pid") or before.get("device") != after.get("device"):
        raise ValueError("resource window snapshots must share process and device")

    def safe_integer(value: object) -> int | None:
        return value if type(value) is int and value >= 0 else None

    current_before = safe_integer(before.get("cpu_process_rss_bytes"))
    current_after = safe_integer(after.get("cpu_process_rss_bytes"))
    peak_before = safe_integer(before.get("cpu_process_lifetime_peak_rss_bytes"))
    peak_after = safe_integer(after.get("cpu_process_lifetime_peak_rss_bytes"))
    if peak_before is not None and peak_after is not None and peak_after < peak_before:
        raise ValueError("process lifetime RSS high-water mark cannot decrease")
    cpu_lower_bound = (
        max(current_before, current_after)
        if current_before is not None and current_after is not None else None
    )
    new_peak = (
        peak_after if peak_before is not None and peak_after is not None
        and peak_after > peak_before else None
    )

    cuda_before = before.get("cuda_pytorch_allocator")
    cuda_after = after.get("cuda_pytorch_allocator")
    gpu_window: dict[str, int | None] | None = None
    if (cuda_before is None) != (cuda_after is None):
        raise ValueError("CUDA snapshot availability changed across fit")
    if isinstance(cuda_before, dict) and isinstance(cuda_after, dict):
        for key in ("allocated_bytes", "reserved_bytes", "lifetime_max_allocated_bytes", "lifetime_max_reserved_bytes"):
            if safe_integer(cuda_before.get(key)) is None or safe_integer(cuda_after.get(key)) is None:
                raise ValueError("invalid CUDA allocator counters")
        for key in ("lifetime_max_allocated_bytes", "lifetime_max_reserved_bytes"):
            if cuda_after[key] < cuda_before[key]:
                raise ValueError("CUDA allocator lifetime maxima cannot decrease")
        gpu_window = {
            "allocated_bytes_before": cuda_before["allocated_bytes"],
            "allocated_bytes_after": cuda_after["allocated_bytes"],
            "reserved_bytes_before": cuda_before["reserved_bytes"],
            "reserved_bytes_after": cuda_after["reserved_bytes"],
            "lifetime_max_allocated_bytes_before": cuda_before["lifetime_max_allocated_bytes"],
            "lifetime_max_allocated_bytes_after": cuda_after["lifetime_max_allocated_bytes"],
            "new_lifetime_max_allocated_bytes_during_fit": (
                cuda_after["lifetime_max_allocated_bytes"]
                if cuda_after["lifetime_max_allocated_bytes"] >
                cuda_before["lifetime_max_allocated_bytes"] else None
            ),
            "exact_fit_window_peak_cuda_bytes": None,
        }
    return {
        "format": "simple-adp-visual-fit-process-resources-v1",
        "pid": before["pid"],
        "device": before["device"],
        "rss_current_before_bytes": current_before,
        "rss_current_after_bytes": current_after,
        "rss_current_delta_bytes": (
            current_after - current_before
            if current_before is not None and current_after is not None else None
        ),
        "rss_window_peak_lower_bound_bytes": cpu_lower_bound,
        "lifetime_peak_rss_before_bytes": peak_before,
        "lifetime_peak_rss_after_bytes": peak_after,
        "new_lifetime_peak_rss_during_fit_bytes": new_peak,
        "exact_fit_window_peak_rss_bytes": None,
        "cuda_pytorch_allocator": gpu_window,
        "scope": (
            "Fit call including native search, stage training and checkpointing; "
            "RSS whole process, CUDA PyTorch allocator current device only"
        ),
        "limitations": (
            "Process lifetime maxima are not reset. Unchanged lifetime maxima "
            "cannot reveal a distinct fit-stage peak; no background sampling. "
            "Not total GPU VRAM, not attributable solely to model/optimizer, "
            "and not a peak-training-memory bound or OOM guarantee."
        ),
    }
