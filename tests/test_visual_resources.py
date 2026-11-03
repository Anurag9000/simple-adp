"""Observed process RSS and CUDA allocator evidence is explicitly nonintrusive."""
from __future__ import annotations

import os

import pytest
import torch

from simple_adp.visual_resources import (
    _linux_process_rss_bytes,
    visual_process_resource_snapshot,
    visual_fit_resource_window,
)


def snapshot(*, rss, peak, device="cpu", pid=1234, cuda=None):
    return {
        "format": "simple-adp-visual-process-resource-snapshot-v1",
        "pid": pid, "device": device,
        "cpu_process_rss_bytes": rss,
        "cpu_process_lifetime_peak_rss_bytes": peak,
        "cuda_pytorch_allocator": cuda,
    }


def test_real_cpu_snapshot_observes_process_rss_without_modifying_rng():
    before = torch.get_rng_state().clone()
    out = visual_process_resource_snapshot("cpu")
    assert out["pid"] == os.getpid()
    assert out["device"] == "cpu"
    assert out["cuda_pytorch_allocator"] is None
    if os.path.exists("/proc/self/status"):
        assert out["cpu_process_rss_bytes"] > 0
        assert out["cpu_process_lifetime_peak_rss_bytes"] >= out["cpu_process_rss_bytes"]
    else:
        assert out["cpu_process_rss_bytes"] is None
    assert torch.equal(before, torch.get_rng_state())


def test_rss_delta_and_lifetime_hwm_are_not_misreported_as_window_peak():
    out = visual_fit_resource_window(
        snapshot(rss=200, peak=300),
        snapshot(rss=250, peak=400),
    )
    assert out["rss_current_delta_bytes"] == 50
    assert out["rss_window_peak_lower_bound_bytes"] == 250
    assert out["new_lifetime_peak_rss_during_fit_bytes"] == 400
    assert out["exact_fit_window_peak_rss_bytes"] is None
    assert out["cuda_pytorch_allocator"] is None
    assert "not a peak-training-memory bound" in out["limitations"]


def test_highwater_unmodified_does_not_fabricate_interval_peak():
    out = visual_fit_resource_window(
        snapshot(rss=500, peak=900),
        snapshot(rss=200, peak=900),
    )
    assert out["rss_current_delta_bytes"] == -300
    assert out["rss_window_peak_lower_bound_bytes"] == 500
    assert out["new_lifetime_peak_rss_during_fit_bytes"] is None
    assert out["exact_fit_window_peak_rss_bytes"] is None


def test_missing_procfs_samples_remain_unknown():
    out = visual_fit_resource_window(
        snapshot(rss=None, peak=None),
        snapshot(rss=None, peak=None),
    )
    assert out["rss_current_delta_bytes"] is None
    assert out["rss_window_peak_lower_bound_bytes"] is None
    assert out["new_lifetime_peak_rss_during_fit_bytes"] is None


@pytest.mark.parametrize("difference", ["pid", "device", "format", "hwm"])
def test_mismatched_snapshot_provenance_rejected(difference):
    before = snapshot(rss=100, peak=200)
    after = snapshot(rss=150, peak=250)
    if difference == "pid":
        after["pid"] = 5678
    elif difference == "device":
        after["device"] = "cuda:0"
    elif difference == "format":
        after["format"] = "wrong-version"
    else:
        after["cpu_process_lifetime_peak_rss_bytes"] = 100
    with pytest.raises(ValueError):
        visual_fit_resource_window(before, after)


def test_explicit_cuda_allocator_samples_preserve_distinction_from_actual_gpu_vram():
    before = snapshot(
        rss=100, peak=300, device="cuda:0",
        cuda={
            "allocated_bytes": 50, "reserved_bytes": 100,
            "lifetime_max_allocated_bytes": 90,
            "lifetime_max_reserved_bytes": 110,
        },
    )
    after = snapshot(
        rss=140, peak=400, device="cuda:0",
        cuda={
            "allocated_bytes": 120, "reserved_bytes": 160,
            "lifetime_max_allocated_bytes": 180,
            "lifetime_max_reserved_bytes": 200,
        },
    )
    out = visual_fit_resource_window(before, after)
    gpu = out["cuda_pytorch_allocator"]
    assert gpu["allocated_bytes_before"] == 50
    assert gpu["allocated_bytes_after"] == 120
    assert gpu["new_lifetime_max_allocated_bytes_during_fit"] == 180
    assert gpu["exact_fit_window_peak_cuda_bytes"] is None
    assert "not total GPU VRAM" in out["limitations"]


def test_cuda_unavailable_or_mismatched_allocator_rejected():
    before = snapshot(
        rss=100, peak=200, device="cuda:0",
        cuda={
            "allocated_bytes": 10, "reserved_bytes": 10,
            "lifetime_max_allocated_bytes": 15,
            "lifetime_max_reserved_bytes": 20,
        },
    )
    after = snapshot(rss=200, peak=300, device="cuda:0", cuda=None)
    with pytest.raises(ValueError, match="availability"):
        visual_fit_resource_window(before, after)
    after["cuda_pytorch_allocator"] = dict(before["cuda_pytorch_allocator"])
    after["cuda_pytorch_allocator"]["lifetime_max_allocated_bytes"] = 5
    with pytest.raises(ValueError, match="cannot decrease"):
        visual_fit_resource_window(before, after)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="physical CUDA unavailable")
def test_real_cuda_allocator_counters_are_read_only_and_device_specific():
    device = torch.device("cuda:0")
    before = visual_process_resource_snapshot(device)
    tensor = torch.empty((10000,), device=device)
    during = visual_process_resource_snapshot(device)
    del tensor
    out = visual_fit_resource_window(before, during)
    assert during["cuda_pytorch_allocator"]["allocated_bytes"] >= (
        before["cuda_pytorch_allocator"]["allocated_bytes"]
    )
    assert out["cuda_pytorch_allocator"]["exact_fit_window_peak_cuda_bytes"] is None
    assert out["device"] == "cuda:0"


def test_procfs_parser_missing_file_remains_unknown(monkeypatch):
    from simple_adp import visual_resources as resources

    original = resources.Path.read_text

    def missing(self, *args, **kwargs):
        if str(self) == "/proc/self/status":
            raise FileNotFoundError("/proc unavailable")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(resources.Path, "read_text", missing)
    assert _linux_process_rss_bytes() == (None, None)
