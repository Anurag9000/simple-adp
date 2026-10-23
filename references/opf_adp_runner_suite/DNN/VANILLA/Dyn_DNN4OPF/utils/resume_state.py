"""Training state needed to continue a batch-level checkpoint reproducibly."""
from __future__ import annotations

import random
import os
import tempfile
from typing import Any

import torch


CHECKPOINT_SCHEMA_VERSION = 2


def load_exact_checkpoint(path: Any, *, map_location: Any) -> dict[str, Any]:
    """Load a checkpoint and classify unreadable state as a fatal resume error."""
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"exact-resume failure: checkpoint cannot be loaded: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"exact-resume failure: checkpoint payload is not a mapping: {path}")
    return payload


def atomic_checkpoint_save(payload: dict[str, Any], path: Any) -> None:
    """Replace a checkpoint atomically so an interrupted write keeps the prior file."""
    target = os.fspath(path)
    directory = os.path.dirname(target) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{os.path.basename(target)}.", suffix=".tmp", dir=directory)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        dir_fd = os.open(directory, os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_initial_training_checkpoint(
    path: Any,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    train_loader: Any,
    val_loader: Any,
    *,
    patience: int,
    overwrite: bool = False,
) -> None:
    """Persist the exact stage-start state before its first optimizer update."""
    if path is None or (os.path.exists(os.fspath(path)) and not overwrite):
        return
    snapshot = model.snapshot_arch_and_state()
    atomic_checkpoint_save(
        {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "epoch": 0,
            "global_batch": 0,
            "model_state": model.state_dict(),
            "opt_state": optimizer.state_dict(),
            "sched_state": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
            "best_val": float("inf"),
            "best_snap": snapshot,
            "best_batch_loss": float("inf"),
            "best_batch_snap": snapshot,
            "patience": int(patience),
            **capture_training_state(train_loader, val_loader),
        },
        path,
    )


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available() and torch.cuda.is_initialized()
        else [],
    }
    try:
        import numpy as np
        state["numpy"] = np.random.get_state()
    except ImportError:
        state["numpy"] = None
    return state


def restore_rng_state(state: dict[str, Any] | None) -> bool:
    if not state:
        return False
    random.setstate(state["python"])
    if state.get("numpy") is not None:
        import numpy as np
        np.random.set_state(state["numpy"])
    if state.get("torch_cpu") is not None:
        torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state.get("torch_cuda") or []
    if torch.cuda.is_available() and cuda_states:
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])
    return True


def capture_training_state(train_loader: Any, val_loader: Any = None) -> dict[str, Any]:
    sampler = getattr(train_loader, "sampler", None)
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "rng_state": capture_rng_state(),
        "train_sampler_state": sampler.state_dict() if hasattr(sampler, "state_dict") else None,
        "train_loader_generator_state": _generator_state(train_loader),
        "val_loader_generator_state": _generator_state(val_loader),
        "backend_state": {
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "loader_contract": _loader_contract(train_loader),
        "runtime_versions": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }


def restore_training_state(payload: dict[str, Any], train_loader: Any, val_loader: Any = None) -> bool:
    """Restore reproducibility state, refusing checkpoints that cannot be exact.

    Older checkpoints can still contain useful weights, but restoring them and
    continuing silently would change the sample/RNG sequence. Callers must not
    treat that as an exact resume.
    """
    schema = int(payload.get("checkpoint_schema_version", 0) or 0)
    if schema < CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"exact-resume failure: checkpoint schema {schema} lacks exact-resume guarantees "
            f"(required {CHECKPOINT_SCHEMA_VERSION})"
        )
    # Schema-2 checkpoints from an earlier writer captured the exact RNG and
    # sampler state but omitted the explicit batch cursor at epoch boundaries.
    # The checkpointable sampler marks itself exhausted only after the loader
    # has reached StopIteration. With a complete saved permutation, that proves
    # the next cursor is the next epoch (batch zero); never infer this for a
    # mid-epoch or incomplete sampler state.
    if "global_batch" not in payload:
        sampler_state = payload.get("train_sampler_state")
        sampler = getattr(train_loader, "sampler", None)
        order = sampler_state.get("order") if isinstance(sampler_state, dict) else None
        dataset = getattr(train_loader, "dataset", None)
        if (
            isinstance(sampler_state, dict)
            and sampler_state.get("exhausted") is True
            and isinstance(order, (list, tuple))
            and dataset is not None
            and hasattr(sampler, "load_state_dict")
            and len(order) == len(dataset)
            and sorted(order) == list(range(len(dataset)))
        ):
            payload["global_batch"] = 0
    adapter_without_scheduler = payload.get("training_kind") == "adp_external_adapter_no_scheduler"
    required = ("model_state", "opt_state", "epoch", "global_batch")
    if not adapter_without_scheduler:
        required = (*required, "sched_state")
    missing = [key for key in required if key not in payload or payload[key] is None]
    if missing:
        raise ValueError(f"exact-resume failure: checkpoint is missing training state fields: {', '.join(missing)}")
    if adapter_without_scheduler:
        adapter_required = (
            "best_val", "best_state", "es_counter", "history",
            "train_epoch_log_accum", "train_epoch_n",
        )
        adapter_missing = [key for key in adapter_required if key not in payload]
        if adapter_missing:
            raise ValueError(f"exact-resume failure: external ADP checkpoint is missing state fields: {', '.join(adapter_missing)}")
    elif "patience" in payload:
        adp_required = ("best_val", "best_snap", "best_batch_loss", "best_batch_snap")
        adp_missing = [key for key in adp_required if key not in payload]
        if adp_missing:
            raise ValueError(f"exact-resume failure: adaptive checkpoint is missing state fields: {', '.join(adp_missing)}")
    elif "early_stop_counter" in payload:
        native_required = ("best_val", "epoch_loss_total", "epoch_loss_count", "history")
        native_missing = [key for key in native_required if key not in payload]
        if native_missing:
            raise ValueError(f"exact-resume failure: native checkpoint is missing state fields: {', '.join(native_missing)}")
    rng_state = payload.get("rng_state")
    if not isinstance(rng_state, dict) or not rng_state.get("python") or rng_state.get("torch_cpu") is None:
        raise ValueError("exact-resume failure: checkpoint is missing Python or CPU RNG state")
    try:
        import numpy as np  # noqa: F401
    except ImportError:
        pass
    else:
        if rng_state.get("numpy") is None:
            raise ValueError("exact-resume failure: checkpoint is missing NumPy RNG state")
    if hasattr(train_loader, "generator") and getattr(train_loader, "generator") is not None:
        if not isinstance(payload.get("train_loader_generator_state"), torch.Tensor):
            raise ValueError("exact-resume failure: checkpoint is missing the training loader generator state")
    if val_loader is not None and getattr(val_loader, "generator", None) is not None:
        if not isinstance(payload.get("val_loader_generator_state"), torch.Tensor):
            raise ValueError("exact-resume failure: checkpoint is missing the validation loader generator state")
    cursor = int(payload.get("global_batch", 0) or 0)
    sampler = getattr(train_loader, "sampler", None)
    sampler_state = payload.get("train_sampler_state")
    if cursor > 0:
        if not isinstance(sampler_state, dict) or sampler_state.get("order") is None:
            raise ValueError("exact-resume failure: mid-epoch checkpoint is missing the shuffled sample order")
        if not hasattr(sampler, "load_state_dict"):
            raise ValueError("exact-resume failure: current sampler cannot restore the saved mid-epoch position")
    saved_contract = payload.get("loader_contract")
    current_contract = _loader_contract(train_loader)
    if current_contract is not None and saved_contract is None:
        raise ValueError("exact-resume failure: checkpoint is missing the training loader contract")
    if saved_contract is not None and saved_contract != current_contract:
        raise ValueError(f"exact-resume failure: checkpoint loader contract mismatch: saved={saved_contract}, current={current_contract}")
    sampler = getattr(train_loader, "sampler", None)
    try:
        if sampler_state is not None and hasattr(sampler, "load_state_dict"):
            sampler.load_state_dict(sampler_state, replay_current_epoch=cursor > 0)
        _restore_generator_state(train_loader, payload.get("train_loader_generator_state"))
        _restore_generator_state(val_loader, payload.get("val_loader_generator_state"))
        backend = payload.get("backend_state") or {}
        if backend:
            torch.backends.cudnn.deterministic = bool(backend.get("cudnn_deterministic", False))
            torch.backends.cudnn.benchmark = bool(backend.get("cudnn_benchmark", False))
            torch.use_deterministic_algorithms(bool(backend.get("deterministic_algorithms", False)), warn_only=True)
        if cursor > 0 and hasattr(train_loader, "defer_rng_restore"):
            # Creating a replacement DataLoader iterator consumes its base seed.
            train_loader.defer_rng_restore(rng_state)
            rng_restored = True
        else:
            rng_restored = restore_rng_state(rng_state)
    except Exception as exc:
        raise RuntimeError(f"exact-resume failure: saved training state could not be restored: {exc}") from exc
    if not rng_restored:
        raise ValueError("exact-resume failure: RNG restoration returned false")
    return True


def _generator_state(loader: Any) -> torch.Tensor | None:
    generator = getattr(loader, "generator", None) if loader is not None else None
    return generator.get_state() if generator is not None else None


def _loader_contract(loader: Any) -> dict[str, Any] | None:
    # Legacy entry points may accept concrete batch collections rather than a
    # torch DataLoader. RNG state still applies, but there is no loader contract.
    if loader is None or not hasattr(loader, "dataset"):
        return None
    sampler = getattr(loader, "sampler", None)
    dataset = loader.dataset
    return {
        "dataset_type": f"{type(dataset).__module__}.{type(dataset).__qualname__}",
        "dataset_length": len(dataset),
        "batch_size": getattr(loader, "batch_size", None),
        "drop_last": bool(getattr(loader, "drop_last", False)),
        "sampler_type": f"{type(sampler).__module__}.{type(sampler).__qualname__}",
    }


def _restore_generator_state(loader: Any, state: torch.Tensor | None) -> None:
    generator = getattr(loader, "generator", None) if loader is not None else None
    if generator is not None and state is not None:
        generator.set_state(state.cpu())
