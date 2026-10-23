from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Mapping, MutableMapping, Optional

import torch
import torch.nn as nn


def load_checkpoint_snapshot(
    checkpoint_path: object,
    *,
    device: torch.device,
) -> Optional[Mapping[str, object]]:
    """Load a best architecture snapshot without mutating a live model."""
    if not checkpoint_path or not Path(str(checkpoint_path)).is_file():
        return None
    try:
        try:
            checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        except TypeError:  # Older PyTorch releases do not accept weights_only.
            checkpoint = torch.load(str(checkpoint_path), map_location=device)
    except Exception:
        return None
    snapshot = checkpoint.get("best_snap") if isinstance(checkpoint, Mapping) else None
    return snapshot if isinstance(snapshot, Mapping) else None


def restore_stage_architecture(model: object, stage_state: Mapping[str, object]) -> bool:
    """Restore the architecture/cursor recorded before an interrupted stage."""
    widths = stage_state.get("current_hidden_widths")
    if not isinstance(widths, (list, tuple)) or not widths:
        raise RuntimeError("exact-resume failure: stage state lacks the current hidden-width vector")
    checkpoint_path = stage_state.get("stage_checkpoint")
    architecture_path = stage_state.get("stage_architecture_checkpoint")
    restore_arch_and_state = getattr(model, "restore_arch_and_state", None)
    source_path = checkpoint_path if checkpoint_path and Path(str(checkpoint_path)).is_file() else None
    if source_path is None and architecture_path:
        relative = Path(str(architecture_path))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("exact-resume failure: stage architecture checkpoint path is not relative")
        if not checkpoint_path:
            raise RuntimeError("exact-resume failure: active stage checkpoint path is missing")
        root = Path(str(checkpoint_path)).resolve().parents[2]
        candidate = root / relative
        if candidate.is_file():
            source_path = candidate
    if source_path and callable(restore_arch_and_state):
        try:
            checkpoint = torch.load(str(source_path), map_location="cpu", weights_only=False)
            snapshot = checkpoint.get("best_snap") if isinstance(checkpoint, Mapping) else None
            if snapshot is None and isinstance(checkpoint, Mapping):
                snapshot = checkpoint.get("architecture_snapshot")
            saved_widths = snapshot.get("hidden_widths") if isinstance(snapshot, Mapping) else None
            if not isinstance(saved_widths, (list, tuple)) or tuple(map(int, saved_widths)) != tuple(map(int, widths)):
                raise RuntimeError("checkpoint architecture does not match the stage sidecar")
            restore_arch_and_state(snapshot)
            _restore_controller_rng_after_architecture(stage_state, checkpoint_path)
            return True
        except Exception as exc:
            raise RuntimeError(f"exact-resume failure: saved stage architecture could not be restored: {exc}") from exc
    rebuild = getattr(model, "_rebuild_hidden_layers", None) or getattr(model, "_rebuild", None)
    if not callable(rebuild):
        raise RuntimeError("exact-resume failure: model cannot rebuild its saved stage architecture")
    try:
        rebuild([int(width) for width in widths])
        setattr(model, "_width_growth_cursor", int(stage_state.get("current_width_growth_cursor", 0)))
        _restore_controller_rng_after_architecture(stage_state, checkpoint_path)
        return True
    except Exception as exc:
        raise RuntimeError(f"exact-resume failure: saved stage architecture could not be restored: {exc}") from exc


def _restore_controller_rng_after_architecture(stage_state: Mapping[str, object], checkpoint_path: object) -> None:
    if bool(stage_state.get("active_stage")) and checkpoint_path and Path(str(checkpoint_path)).is_file():
        return  # the active training checkpoint is the RNG authority
    rng_path_value = stage_state.get("rng_checkpoint_path")
    rng_name = stage_state.get("rng_checkpoint")
    if not rng_path_value and not rng_name:
        return
    results_dir = Path(str(checkpoint_path)).resolve().parents[2]
    if rng_path_value:
        relative_rng_path = Path(str(rng_path_value))
        if relative_rng_path.is_absolute() or ".." in relative_rng_path.parts:
            raise RuntimeError("exact-resume failure: controller RNG checkpoint path is not relative")
        rng_path = results_dir / relative_rng_path
    else:
        if Path(str(rng_name)).name != str(rng_name):
            raise RuntimeError("exact-resume failure: controller RNG checkpoint name is not local")
        matches = list(results_dir.rglob(str(rng_name)))
        if len(matches) != 1:
            raise RuntimeError("exact-resume failure: controller RNG checkpoint path is ambiguous or missing")
        rng_path = matches[0]
    try:
        rng_state = torch.load(rng_path, map_location="cpu", weights_only=False)
        from Dyn_DNN4OPF.utils.resume_state import restore_rng_state

        if not restore_rng_state(rng_state):
            raise ValueError("saved controller RNG state is invalid")
    except Exception as exc:
        raise RuntimeError(f"exact-resume failure: controller RNG checkpoint could not be restored: {rng_path}") from exc


def checkpoint_architecture_matches(
    current_snapshot: Mapping[str, object],
    checkpoint_snapshot: object,
) -> bool:
    """Return whether a stage checkpoint belongs to the live architecture."""
    if not isinstance(checkpoint_snapshot, Mapping):
        # A missing snapshot cannot prove that a checkpoint belongs to the
        # current growth stage. Refuse it rather than partially loading stale
        # tensors into a different architecture.
        return False

    def widths(snapshot: Mapping[str, object]) -> Optional[tuple[int, ...]]:
        hidden = snapshot.get("hidden_widths")
        if isinstance(hidden, (list, tuple)) and hidden:
            return tuple(int(value) for value in hidden)
        width = snapshot.get("width")
        depth = snapshot.get("depth")
        if width is None or depth is None:
            return None
        return (int(width),) * int(depth)

    current_widths = widths(current_snapshot)
    checkpoint_widths = widths(checkpoint_snapshot)
    if current_widths is not None and checkpoint_widths is not None:
        return current_widths == checkpoint_widths
    return False


def discover_exact_active_depth_stage(
    results_dir: object,
    *,
    width_growth_strategy: Optional[str] = None,
) -> Optional[Mapping[str, object]]:
    """Recover a legacy depth controller from its last exact stage checkpoint.

    Older depth controllers wrote per-stage summaries only after each stage and
    did not mirror the active stage into the task-level sidecar. If the latest
    stage has a schema-2 batch checkpoint and its preceding stage has a valid
    controller summary, the search cursor and best-so-far decision state are
    reconstructible without changing the training trajectory.
    """
    root = Path(str(results_dir)) if results_dir else None
    if root is None or not root.is_dir():
        return None
    candidates = []
    for stage_dir in root.glob("stages/[0-9][0-9][0-9]_*"):
        match = re.match(r"^(\d{3})_(.+)$", stage_dir.name)
        if not match:
            continue
        checkpoint_path = stage_dir / "checkpoint_last.pt"
        if checkpoint_path.is_file():
            candidates.append((int(match.group(1)), match.group(2), stage_dir, checkpoint_path))
    for stage_index, stage_name, stage_dir, checkpoint_path in sorted(candidates, reverse=True):
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            required = (
                "checkpoint_schema_version", "model_state", "opt_state", "sched_state",
                "epoch", "global_batch", "rng_state", "train_sampler_state",
                "loader_contract", "backend_state", "best_snap", "best_val",
                "best_batch_loss", "best_batch_snap", "patience",
            )
            if not isinstance(checkpoint, dict) or any(key not in checkpoint for key in required):
                continue
            if int(checkpoint["checkpoint_schema_version"]) < 2:
                continue
            # A positive batch cursor identifies a mid-epoch stage checkpoint.
            # A zero cursor plus an exhausted saved sampler identifies the
            # post-validation epoch boundary checkpoint; it already contains
            # optimizer, scheduler, RNG, and patience state, so no sidecar is
            # needed to prove the cursor.
            cursor = int(checkpoint.get("global_batch", 0) or 0)
            if cursor <= 0:
                sampler_state = checkpoint.get("train_sampler_state")
                if not isinstance(sampler_state, dict) or not bool(sampler_state.get("exhausted")):
                    continue
            rng_state = checkpoint.get("rng_state")
            if (
                not isinstance(rng_state, dict)
                or not rng_state.get("python")
                or rng_state.get("torch_cpu") is None
                or rng_state.get("numpy") is None
                or not isinstance(checkpoint.get("loader_contract"), dict)
                or not isinstance(checkpoint.get("backend_state"), dict)
            ):
                continue
            sampler_state = checkpoint["train_sampler_state"]
            if cursor > 0 and (
                not isinstance(sampler_state, dict) or sampler_state.get("order") is None
            ):
                continue
            current_snap = checkpoint.get("best_snap")
            current_widths = current_snap.get("hidden_widths") if isinstance(current_snap, dict) else None
            if not isinstance(current_widths, (list, tuple)) or not current_widths:
                continue
            previous_dirs = list(root.glob(f"stages/{stage_index - 1:03d}_*"))
            if len(previous_dirs) != 1:
                continue
            previous_dir = previous_dirs[0]
            summary_path = previous_dir / "stage_state.json"
            if not summary_path.is_file():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            best_checkpoint = Path(str(summary.get("best_checkpoint") or (previous_dir / "checkpoint_best.pt")))
            if not best_checkpoint.is_absolute():
                best_checkpoint = Path.cwd() / best_checkpoint
            if not best_checkpoint.is_file():
                continue
            best_snapshot = load_checkpoint_snapshot(best_checkpoint, device=torch.device("cpu"))
            if not isinstance(summary, dict) or best_snapshot is None:
                continue
            if not all(key in summary for key in ("best_val", "best_depth")):
                continue
            if "depth_fail" not in summary and "width_fail" not in summary:
                continue
            width_cursor = summary.get("current_width_growth_cursor")
            if width_cursor is None:
                width_cursor = current_snap.get("width_growth_cursor") if isinstance(current_snap, dict) else None
            strategy = str(width_growth_strategy or "all_layers").strip().lower()
            if strategy in {"round_robin", "layer_wise", "round-robin"} and width_cursor is None:
                # Older depth controllers did not copy the cursor into the
                # best snapshot. While the saved controller is still in its
                # depth-growth chain, no width-growth call has happened yet;
                # the constructor cursor (zero) is therefore reconstructible.
                stage_names = [path.name.split("_", 1)[1] for path in root.glob("stages/[0-9][0-9][0-9]_*")]
                still_in_depth_chain = (
                    stage_name.startswith("depth_")
                    and "depth_fail" in summary
                    and all(name == "initial" or name.startswith("depth_") for name in stage_names)
                )
                if not still_in_depth_chain:
                    continue
                width_cursor = 0
            if not isinstance(best_snapshot.get("state_dict"), Mapping):
                continue
            return {
                "stage_index": stage_index,
                "stage_name": stage_name,
                "best_val": float(summary["best_val"]),
                "best_width": int(summary.get("best_width", summary.get("width", current_widths[0]))),
                "best_depth": int(summary["best_depth"]),
                "depth_fail": int(summary.get("depth_fail", 0)),
                "width_fail": int(summary["width_fail"]) if "width_fail" in summary else None,
                "controller_kind": "width" if "width_fail" in summary and "depth_fail" not in summary else "depth",
                "best_checkpoint": str(best_checkpoint),
                "completed": False,
                "active_stage": True,
                "current_hidden_widths": [int(value) for value in current_widths],
                "current_width_growth_cursor": int(width_cursor) if width_cursor is not None else 0,
                "stage_checkpoint": str(checkpoint_path),
                "stage_start_width": int(summary.get("width", current_widths[0])),
                "legacy_stage_recovery": True,
            }
        except Exception:
            continue
    return None


def load_partial_state_dict(
    module: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    logger: Optional[object] = None,
    context: str = "",
) -> None:
    target_state = module.state_dict()
    copied = 0
    skipped = 0

    for key, value in state_dict.items():
        if key not in target_state:
            skipped += 1
            continue
        target_tensor = target_state[key]
        if not isinstance(target_tensor, torch.Tensor) or not isinstance(value, torch.Tensor):
            skipped += 1
            continue

        if target_tensor.shape == value.shape:
            target_state[key] = value.detach().clone().to(device=target_tensor.device, dtype=target_tensor.dtype)
            copied += 1
            continue

        if target_tensor.ndim != value.ndim:
            skipped += 1
            continue

        merged = target_tensor.detach().clone()
        slices = tuple(slice(0, min(a, b)) for a, b in zip(merged.shape, value.shape))
        if all(s.stop > 0 for s in slices):
            merged[slices] = value.detach()[slices].to(device=merged.device, dtype=merged.dtype)
            target_state[key] = merged
            copied += 1
        else:
            skipped += 1

    module.load_state_dict(target_state, strict=False)

    if logger is not None:
        prefix = f"{context}: " if context else ""
        try:
            logger.info("%sloaded partial checkpoint state copied=%d skipped=%d", prefix, copied, skipped)
        except Exception:
            pass


def restore_stage_checkpoint(
    model: object,
    stage_state: Mapping[str, object],
    *,
    device: torch.device,
    logger: object,
    context: str,
) -> bool:
    """Restore a stage best snapshot, returning false for unusable state."""
    checkpoint_path = stage_state.get("best_checkpoint")
    if not checkpoint_path or not Path(str(checkpoint_path)).is_file():
        raise RuntimeError(f"exact-resume failure: {context} stage state has no readable best checkpoint")
    try:
        snapshot = load_checkpoint_snapshot(checkpoint_path, device=device)
        if snapshot is None:
            raise RuntimeError(f"exact-resume failure: {context} best checkpoint has no architecture snapshot")
        model.restore_arch_and_state(snapshot)
        return True
    except Exception as exc:
        raise RuntimeError(f"exact-resume failure: could not restore {context} stage checkpoint: {exc}") from exc


def restore_stage_last_checkpoint(
    model: object,
    checkpoint_path: object,
    *,
    device: torch.device,
    logger: object,
    context: str,
) -> bool:
    """Restore the latest model weights for a completed growth stage."""
    if not checkpoint_path or not Path(str(checkpoint_path)).is_file():
        return False
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location=device)
        model_state = checkpoint.get("model_state") if isinstance(checkpoint, Mapping) else None
        if not isinstance(model_state, Mapping):
            return False
        load_partial_state_dict(model, model_state, logger=logger, context=f"{context} last")
        return True
    except Exception:
        logger.exception("Could not restore %s latest checkpoint", context)
        return False


def restore_stage_last_checkpoint_exact(model: object, checkpoint_path: object, *, device: torch.device, context: str) -> bool:
    """Restore a completed stage's exact model architecture and weights."""
    if not checkpoint_path or not Path(str(checkpoint_path)).is_file():
        return False
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        if not isinstance(checkpoint, Mapping) or int(checkpoint.get("checkpoint_schema_version", 0)) < 2:
            return False
        if not checkpoint_architecture_matches(model.snapshot_arch_and_state(), checkpoint.get("best_snap")):
            return False
        model_state = checkpoint.get("model_state")
        if not isinstance(model_state, Mapping):
            return False
        model.load_state_dict(model_state, strict=True)
        return True
    except Exception as exc:
        raise RuntimeError(f"exact-resume failure: could not restore {context} stage checkpoint") from exc
