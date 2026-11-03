"""Exact OPF_ADP V01 evaluate and train_stage function slices.

Only imports and annotations are made available by this wrapper.
The function text following this header is copied verbatim from
Anurag9000/OPF_ADP/vision_adp/v01.py, blob b33a51e22a50df24877ee894c8a85a7e1d8b84a1.
This isolated certificate does NOT reproduce ADP_DEN geometry growth.
"""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F
from .shadow_bgd_variance import ShadowBGDVariance
from .v01_resume import capture_rng, restore_rng, loader_generator_state, restore_loader_generator

@torch.no_grad()
def evaluate(model: V01ImageHead, loader, device: torch.device):
    model.eval()
    loss_total, correct, total = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss_total += F.cross_entropy(logits, y, reduction="sum").item()
        correct += int((logits.argmax(dim=-1) == y).sum().item())
        total += int(y.numel())
    if not total:
        raise RuntimeError("empty evaluation loader")
    return {"cross_entropy": loss_total / total, "accuracy": correct / total, "n": total}


def train_stage(model, train_loader, val_loader, device, *, epochs, lr, weight_decay,
                shadow: ShadowBGDVariance | None = None,
                resume_state: dict | None = None, checkpoint_callback=None):
    """Stage-local AdamW training with opt-in exact minibatch cursor replay.

    A recovery checkpoint is committed after every fully processed batch,
    including its auxiliary observer step, and at every epoch boundary.
    Native training without checkpoint_callback/resume_state is unchanged.
    """
    if (checkpoint_callback is not None or resume_state is not None) and (
        train_loader.num_workers != 0
        or getattr(train_loader, "generator", None) is None
    ):
        raise ValueError("exact V01 minibatch resume requires workers=0 and an explicit loader generator")
    if resume_state is not None:
        # Rebuilding widened layers consumes RNG, so restore it *after*
        # reconstruction and after replaying the already-seen batches.
        model.head.restore_arch_and_state(resume_state["current_head_snapshot"])
        if shadow is not None:
            if resume_state.get("shadow") is None:
                raise ValueError("missing BGD posterior in variance resume")
            shadow.load_state_dict(resume_state["shadow"])
        elif resume_state.get("shadow") is not None:
            raise ValueError("loss-mode checkpoint unexpectedly contains BGD state")
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=weight_decay)
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        local_best_val = float(resume_state["local_best_val"])
        local_best_snapshot = resume_state["local_best_snapshot"]
        local_best_shadow = resume_state["local_best_shadow"]
        trace = list(resume_state["trace"])
        first_epoch = int(resume_state["epoch"])
        first_batch = int(resume_state["batch"])
        if (
            first_epoch < 0 or first_epoch > epochs
            or first_batch < 0
            or (first_epoch == epochs and first_batch != 0)
        ):
            raise ValueError("invalid V01 saved epoch/batch cursor")
        restore_loader_generator(train_loader, resume_state["loader_generator"])
        restore_rng(resume_state["rng"])
    else:
        local_best_val = float("inf")
        local_best_snapshot = model.head.snapshot_arch_and_state()
        local_best_shadow = None
        trace = []
        first_epoch = 0
        first_batch = 0

    for epoch in range(first_epoch, epochs):
        model.train()
        continuing_epoch = resume_state is not None and epoch == first_epoch
        if continuing_epoch:
            seen = int(resume_state["seen"])
            loss_total = float(resume_state["loss_total"])
            batch_cursor = first_batch
            epoch_generator_start = resume_state["epoch_generator_start"]
            if batch_cursor:
                # Regenerate the identical shuffle permutation and iterate
                # already processed batches without replaying optimizer steps.
                # Batch data loading may advance global RNG: restore the saved
                # RNG *after* advancing the iterator to the exact cursor.
                restore_loader_generator(train_loader, epoch_generator_start)
                iterator = iter(train_loader)
                for _ in range(batch_cursor):
                    try:
                        next(iterator)
                    except StopIteration as exc:
                        raise ValueError("checkpoint batch cursor exceeds data") from exc
                restore_loader_generator(train_loader, resume_state["loader_generator"])
                restore_rng(resume_state["rng"])
            else:
                iterator = iter(train_loader)
                # If there is no completed batch, the loader's generator
                # state in the checkpoint is the epoch-start state.
                epoch_generator_start = resume_state["loader_generator"]
        else:
            seen, loss_total, batch_cursor = 0, 0.0, 0
            epoch_generator_start = loader_generator_state(train_loader)
            iterator = iter(train_loader)
        for x, y in iterator:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite V01 training loss")
            loss.backward()
            optimizer.step()
            if shadow is not None:
                shadow.observe(model, x, y, F.cross_entropy)
                if shadow.contraction is None or not math.isfinite(shadow.contraction):
                    raise RuntimeError("visual ADP variance observer failed to measure contraction")
            loss_total += loss.item() * len(y)
            seen += len(y)
            batch_cursor += 1
            if checkpoint_callback is not None:
                checkpoint_callback({
                    "epoch": epoch, "batch": batch_cursor,
                    "seen": seen, "loss_total": loss_total,
                    "current_head_snapshot": model.head.snapshot_arch_and_state(),
                    "optimizer": optimizer.state_dict(),
                    "local_best_val": local_best_val,
                    "local_best_snapshot": local_best_snapshot,
                    "local_best_shadow": local_best_shadow,
                    "trace": trace,
                    "shadow": shadow.state_dict() if shadow is not None else None,
                    "epoch_generator_start": epoch_generator_start,
                    "loader_generator": loader_generator_state(train_loader),
                    "rng": capture_rng(),
                })
        if not seen:
            raise RuntimeError("empty train loader")
        val = evaluate(model, val_loader, device)
        trace.append({
            "epoch": epoch + 1,
            "train_cross_entropy": loss_total / seen,
            "val_cross_entropy": val["cross_entropy"],
            "val_accuracy": val["accuracy"],
        })
        if val["cross_entropy"] < local_best_val - 1e-12:
            local_best_val = val["cross_entropy"]
            local_best_snapshot = model.head.snapshot_arch_and_state()
            if shadow is not None:
                local_best_shadow = shadow.state_dict()
        if checkpoint_callback is not None:
            # Durable epoch boundary: the next epoch has no consumed batches.
            checkpoint_callback({
                "epoch": epoch + 1, "batch": 0,
                "seen": 0, "loss_total": 0.0,
                "current_head_snapshot": model.head.snapshot_arch_and_state(),
                "optimizer": optimizer.state_dict(),
                "local_best_val": local_best_val,
                "local_best_snapshot": local_best_snapshot,
                "local_best_shadow": local_best_shadow,
                "trace": trace,
                "shadow": shadow.state_dict() if shadow is not None else None,
                "epoch_generator_start": loader_generator_state(train_loader),
                "loader_generator": loader_generator_state(train_loader),
                "rng": capture_rng(),
            })
    model.head.restore_arch_and_state(local_best_snapshot)
    if shadow is not None:
        if local_best_shadow is None:
            raise RuntimeError("visual ADP best candidate lacks matching variance state")
        shadow.load_state_dict(local_best_shadow)
    return local_best_val, trace

