from __future__ import annotations

import logging
import sys
from typing import Any, Callable, Dict, Optional

try:
    import psutil
except Exception:  # pragma: no cover - psutil is expected but keep logging robust
    psutil = None

try:
    import torch
except Exception:  # pragma: no cover - torch is expected in training jobs
    torch = None


def _default_logging_context() -> Dict[str, Any]:
    host_total = 0.0
    host_used = 0.0
    host_available = 0.0
    if psutil is not None:
        try:
            mem = psutil.virtual_memory()
            host_total = float(mem.total) / (1024 * 1024)
            host_used = float(mem.used) / (1024 * 1024)
            host_available = float(mem.available) / (1024 * 1024)
        except Exception:
            pass

    gpu_running = 0
    gpu_vram_used = 0.0
    gpu_vram_total = 0.0
    if torch is not None:
        try:
            if torch.cuda.is_available():
                gpu_running = 1
                device_index = torch.cuda.current_device()
                free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
                gpu_vram_total = float(total_bytes) / (1024 * 1024)
                gpu_vram_used = float(total_bytes - free_bytes) / (1024 * 1024)
        except Exception:
            pass

    return {
        "opf_gpu_running": gpu_running,
        "opf_cpu_running": 0,
        "opf_gpu_vram_used_mib": gpu_vram_used,
        "opf_gpu_vram_total_mib": gpu_vram_total,
        "opf_host_ram_used_mib": host_used,
        "opf_host_ram_total_mib": host_total,
    }


_LOGGING_CONTEXT_PROVIDER: Callable[[], Dict[str, Any]] = _default_logging_context
_RECORD_FACTORY_INSTALLED = False


def set_logging_context_provider(provider: Optional[Callable[[], Dict[str, Any]]]) -> None:
    global _LOGGING_CONTEXT_PROVIDER
    _LOGGING_CONTEXT_PROVIDER = provider or _default_logging_context


def _install_record_factory() -> None:
    global _RECORD_FACTORY_INSTALLED
    if _RECORD_FACTORY_INSTALLED:
        return

    original_factory = logging.getLogRecordFactory()

    def record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = original_factory(*args, **kwargs)
        context = _default_logging_context()
        try:
            provider_context = _LOGGING_CONTEXT_PROVIDER() or {}
            if isinstance(provider_context, dict):
                context.update(provider_context)
        except Exception:
            pass
        for key, value in context.items():
            setattr(record, key, value)
        return record

    logging.setLogRecordFactory(record_factory)
    _RECORD_FACTORY_INSTALLED = True


def _build_formatter() -> logging.Formatter:
    _install_record_factory()
    return logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s | "
        "gpu_running=%(opf_gpu_running)s cpu_running=%(opf_cpu_running)s "
        "gpu_vram_used=%(opf_gpu_vram_used_mib).0fMiB gpu_vram_total=%(opf_gpu_vram_total_mib).0fMiB "
        "ram_used=%(opf_host_ram_used_mib).0fMiB ram_total=%(opf_host_ram_total_mib).0fMiB",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def configure_console_logging(level: int = logging.INFO, *, name: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(_build_formatter())
        logger.addHandler(handler)

    return logger


def configure_root_logging(level: int = logging.INFO) -> logging.Logger:
    _install_record_factory()
    root = logging.getLogger()
    root.setLevel(level)
    if not any(isinstance(handler, logging.StreamHandler) for handler in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(_build_formatter())
        root.addHandler(handler)
    return root
