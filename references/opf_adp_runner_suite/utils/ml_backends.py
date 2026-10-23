from __future__ import annotations

import os
import random
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as _numpy

def backend_pandas() -> Any:
    enable_optional_gpu_accelerators()
    import pandas as pd

    return pd


@dataclass(frozen=True)
class BackendStatus:
    gpu_available: bool
    cudf_pandas_enabled: bool
    cuml_accel_enabled: bool
    cupy_available: bool

_STATUS: Optional[BackendStatus] = None
_GLOBAL_SEED_ENV = "OPF_ADP_GLOBAL_SEED"

_CPU_FLAGS = ("CPU_ONLY", "TRAINING_CONTROL_CPU_ONLY",
              "OPF_ADP_DISABLE_GPU_ACCELERATORS")
_CPU_TRUE = {"1", "true", "yes", "on", "y"}
_STATUS_CONTEXT: Optional[tuple] = None
_PROBED_GPU_CONTEXT: Optional[tuple] = None


def _cpu_admitted() -> bool:
    """Match the runner's child admission before importing CUDA libraries."""
    return (
        os.environ.get("TRAINING_CONTROL_BACKEND", "").strip().lower() == "cpu"
        or any(os.environ.get(name, "").strip().lower() in _CPU_TRUE
               for name in _CPU_FLAGS)
        or ("CUDA_VISIBLE_DEVICES" in os.environ
            and os.environ["CUDA_VISIBLE_DEVICES"].strip() in {"", "-1"})
    )


def _gpu_admitted() -> bool:
    return os.environ.get("TRAINING_CONTROL_BACKEND", "").strip().lower() == "gpu"


def _check_admission() -> bool:
    """Reject both contradictory assignments and stale initialized GPU state."""
    cpu = _cpu_admitted()
    if cpu and _gpu_admitted():
        raise RuntimeError("conflicting CPU and GPU scheduler admission")
    if _PROBED_GPU_CONTEXT is not None and _PROBED_GPU_CONTEXT != _backend_context():
        raise RuntimeError(
            "CUDA context was initialized before admission/visibility changed; "
            "restart the worker with the new backend environment"
        )
    if (_STATUS is not None and _STATUS_CONTEXT != _backend_context()
            and (_STATUS.gpu_available or _STATUS.cudf_pandas_enabled
                 or _STATUS.cuml_accel_enabled or _STATUS.cupy_available)):
        raise RuntimeError(
            "accelerator admission/visibility changed after GPU initialization; "
            "restart the worker with the new backend environment"
        )
    return cpu


def _backend_context() -> tuple:
    return (_cpu_admitted(), _gpu_admitted(),
            os.environ.get("CUDA_VISIBLE_DEVICES"))


def _reject_masked_gpu_array(value: Any) -> None:
    if not _check_admission():
        return
    if getattr(getattr(value, "__class__", None), "__module__", "").startswith("cupy"):
        raise RuntimeError("CPU-admitted worker cannot inspect or convert a CuPy array")
    device = getattr(value, "device", None)
    if getattr(device, "type", None) in {"cuda", "mps", "xpu", "hip"}:
        raise RuntimeError("CPU-admitted worker cannot inspect an accelerator tensor")



def resolve_seed(seed: Optional[int] = None) -> int:
    if seed is not None:
        try:
            return int(seed)
        except Exception:
            pass
    raw = os.environ.get(_GLOBAL_SEED_ENV)
    if raw is not None and str(raw).strip():
        try:
            return int(raw)
        except Exception:
            pass
    return 42

def gpu_available() -> bool:
    """Return usable CUDA, never just a driver/framework visibility claim."""
    if _check_admission():
        return False
    try:
        import torch
        if torch.cuda.is_available():
            probe = torch.empty((1,), device="cuda")
            probe.fill_(1)
            torch.cuda.synchronize("cuda")
            del probe
            global _PROBED_GPU_CONTEXT
            _PROBED_GPU_CONTEXT = _backend_context()
            return True
    except (ImportError, RuntimeError, OSError, AttributeError):
        # A CPU-only PyTorch wheel does not rule out independently usable CuPy.
        pass
    return _try_import_cupy() is not None


def _try_enable_cudf_pandas() -> bool:
    if _check_admission():
        return False
    if "pandas" in sys.modules:
        warnings.warn("cuDF pandas unavailable: pandas was already imported", RuntimeWarning, stacklevel=2)
        return False
    try:
        import cudf.pandas as accelerator
        accelerator.install()
        return True
    except ImportError:
        return False
    except Exception as exc:
        warnings.warn(f"cuDF pandas activation failed: {exc}", RuntimeWarning, stacklevel=2)
        return False


def _try_enable_cuml_accel() -> bool:
    if _check_admission():
        return False
    if any(name in sys.modules for name in ("sklearn", "umap", "hdbscan")):
        warnings.warn("cuML accelerator unavailable: target library already imported", RuntimeWarning, stacklevel=2)
        return False
    try:
        import cuml.accel as accelerator
        accelerator.install()
        return True
    except ImportError:
        return False
    except Exception as exc:
        warnings.warn(f"cuML accelerator activation failed: {exc}", RuntimeWarning, stacklevel=2)
        return False


def _try_import_cupy() -> Any:
    # This helper is called directly by array_module(), as_numpy(), metrics,
    # and the seed path; each must respect parent CPU admission independently.
    if _check_admission():
        return None
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            return None
        probe = cp.empty((1,), dtype=cp.uint8)
        probe.fill(1)
        cp.cuda.runtime.deviceSynchronize()
        del probe
        global _PROBED_GPU_CONTEXT
        _PROBED_GPU_CONTEXT = _backend_context()
        return cp
    except Exception:
        return None

def backend_numpy() -> Any:
    backend_status = enable_optional_gpu_accelerators()
    cp = _try_import_cupy() if backend_status.cupy_available else None
    return cp if cp is not None else _numpy

def enable_optional_gpu_accelerators() -> BackendStatus:
    global _STATUS, _STATUS_CONTEXT
    forced_cpu = _check_admission()
    context = _backend_context()
    if _STATUS is not None and _STATUS_CONTEXT == context:
        return _STATUS

    if _STATUS is not None and _STATUS_CONTEXT != context:
        # Process-global cuDF/cuML import hooks cannot safely be undone, and
        # changing CUDA visibility after a GPU context was initialized cannot
        # reassign the physical GPU. A fresh child process is required.
        if _STATUS.gpu_available or _STATUS.cudf_pandas_enabled or _STATUS.cuml_accel_enabled or _STATUS.cupy_available:
            raise RuntimeError(
                "accelerator admission/visibility changed after GPU initialization; "
                "restart the worker with the new backend environment"
            )
        _STATUS = None

    if forced_cpu:
        _STATUS = BackendStatus(False, False, False, False)
        _STATUS_CONTEXT = context
        return _STATUS

    available = gpu_available()
    if _gpu_admitted() and not available:
        raise RuntimeError("GPU-admitted child has no usable Torch or CuPy CUDA runtime")
    cudf_enabled = False
    cuml_enabled = False
    cupy_available = False

    if available:
        cudf_enabled = _try_enable_cudf_pandas()
        cuml_enabled = _try_enable_cuml_accel()
        cp = _try_import_cupy()
        cupy_available = cp is not None

    _STATUS = BackendStatus(available, cudf_enabled, cuml_enabled, cupy_available)
    _STATUS_CONTEXT = context
    return _STATUS



def status() -> BackendStatus:
    return enable_optional_gpu_accelerators()

def array_module(*arrays: Any) -> Any:
    for array in arrays:
        _reject_masked_gpu_array(array)
    cp = _try_import_cupy()
    if cp is None:
        return _numpy
    if any(getattr(array, "__class__", None).__module__.startswith("cupy") for array in arrays if array is not None):
        return cp
    return _numpy

def as_numpy(array: Any) -> Any:
    _reject_masked_gpu_array(array)
    cp = _try_import_cupy()
    if cp is not None and hasattr(cp, "asnumpy") and getattr(array, "__class__", None).__module__.startswith("cupy"):
        return cp.asnumpy(array)
    return _numpy.asarray(array)

def as_cupy(array: Any) -> Any:
    _reject_masked_gpu_array(array)
    backend_status = enable_optional_gpu_accelerators()
    cp = _try_import_cupy() if backend_status.cupy_available else None
    return cp.asarray(array) if cp is not None else _numpy.asarray(array)

def seed_everything(seed: Optional[int] = None) -> int:
    resolved = resolve_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(int(resolved))
    random.seed(resolved)
    # Seed both independently; never monkeypatch NumPy's global random API.
    _numpy.random.seed(resolved)
    state = enable_optional_gpu_accelerators()
    if state.cupy_available:
        cp = _try_import_cupy()
        if cp is not None:
            cp.random.seed(resolved)
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        if _check_admission():
            # torch.manual_seed also seeds accelerator generators. Access only
            # the CPU default generator in centrally admitted CPU workers.
            torch.random.default_generator.manual_seed(resolved)
        else:
            torch.manual_seed(resolved)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(resolved)
    return int(resolved)


def backend_accuracy_score():
    enable_optional_gpu_accelerators()

    def _accuracy_score(y_true: Any, y_pred: Any) -> float:
        _reject_masked_gpu_array(y_true)
        _reject_masked_gpu_array(y_pred)
        if hasattr(y_true, "detach") or hasattr(y_pred, "detach"):
            import torch

            true_t = y_true.detach() if hasattr(y_true, "detach") else torch.as_tensor(y_true)
            pred_t = y_pred.detach() if hasattr(y_pred, "detach") else torch.as_tensor(y_pred)
            if pred_t.ndim > 1 and pred_t.shape[-1] > 1:
                pred_t = pred_t.argmax(dim=-1)
            if true_t.ndim > 1 and true_t.shape[-1] > 1:
                true_t = true_t.argmax(dim=-1)
            return true_t.reshape(-1).eq(pred_t.reshape(-1)).to(dtype=torch.float32).mean().item()

        cp = _try_import_cupy()
        if cp is not None and hasattr(y_true, "__class__") and hasattr(y_pred, "__class__"):
            if getattr(y_true.__class__, "__module__", "").startswith("cupy") or getattr(y_pred.__class__, "__module__", "").startswith("cupy"):
                true_a = cp.asarray(y_true)
                pred_a = cp.asarray(y_pred)
                if pred_a.ndim > 1 and pred_a.shape[-1] > 1:
                    pred_a = pred_a.argmax(axis=-1)
                if true_a.ndim > 1 and true_a.shape[-1] > 1:
                    true_a = true_a.argmax(axis=-1)
                return float(cp.mean(true_a.reshape(-1) == pred_a.reshape(-1)).item())

        true_a = _numpy.asarray(y_true)
        pred_a = _numpy.asarray(y_pred)
        if pred_a.ndim > 1 and pred_a.shape[-1] > 1:
            pred_a = pred_a.argmax(axis=-1)
        if true_a.ndim > 1 and true_a.shape[-1] > 1:
            true_a = true_a.argmax(axis=-1)
        return float((true_a.reshape(-1) == pred_a.reshape(-1)).mean())

    return _accuracy_score

def iter_backend_names() -> Iterable[str]:
    status_value = status()
    yield f"gpu_available={status_value.gpu_available}"
    yield f"cudf_pandas_enabled={status_value.cudf_pandas_enabled}"
    yield f"cuml_accel_enabled={status_value.cuml_accel_enabled}"
    yield f"cupy_available={status_value.cupy_available}"
