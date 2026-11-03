"""Side-effect-free, dtype-aware visual model tensor payload accounting.

This is a sum of logical parameter and buffer tensor sizes. It is not a
measurement of unique allocator storage, live activation footprints, CUDA
reserved memory, optimizer state or peak memory during training.
"""
from __future__ import annotations

from collections import defaultdict

from torch import nn


def visual_model_tensor_payload(model: nn.Module) -> dict[str, object]:
    """Return parameter/buffer logical bytes without moving or cloning tensors.

    Shared parameter *objects* are counted once, even if a module references
    them multiple times. Distinct tensor views sharing underlying storage
    may count more than the actual unique allocation; the cap therefore
    guards a conservatively accounted payload, not an exact VRAM peak.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    parameters = {id(t): t for t in model.parameters()}
    buffers = {id(t): t for t in model.buffers()}
    if set(parameters) & set(buffers):
        raise ValueError("model has the same object registered as parameter and buffer")

    def nbytes(tensor) -> int:
        return int(tensor.numel()) * int(tensor.element_size())

    trainable_bytes = sum(nbytes(p) for p in parameters.values() if p.requires_grad)
    frozen_bytes = sum(nbytes(p) for p in parameters.values() if not p.requires_grad)
    buffer_bytes = sum(nbytes(b) for b in buffers.values())
    dtype_bytes: dict[str, int] = defaultdict(int)
    device_bytes: dict[str, int] = defaultdict(int)
    for tensor in (*parameters.values(), *buffers.values()):
        size = nbytes(tensor)
        dtype_bytes[str(tensor.dtype)] += size
        device_bytes[str(tensor.device)] += size

    return {
        "format": "simple-adp-visual-logical-tensor-payload-v1",
        "parameter_tensors": len(parameters),
        "buffer_tensors": len(buffers),
        "trainable_parameter_bytes": int(trainable_bytes),
        "frozen_parameter_bytes": int(frozen_bytes),
        "parameter_payload_bytes": int(trainable_bytes + frozen_bytes),
        "buffer_payload_bytes": int(buffer_bytes),
        "total_model_tensor_payload_bytes": int(
            trainable_bytes + frozen_bytes + buffer_bytes
        ),
        # Under the explicitly hypothetical case that every trainable
        # parameter has a dense gradient and one Adam/AdamW first/second
        # moment, count those payloads without claiming live-allocator peak.
        "nominal_dense_gradient_payload_bytes": int(trainable_bytes),
        "nominal_adam_or_adamw_two_moment_payload_bytes": int(2 * trainable_bytes),
        "nominal_dense_adam_or_adamw_model_grad_moments_bytes": int(
            frozen_bytes + buffer_bytes + 4 * trainable_bytes
        ),
        "nominal_optimizer_assumption": (
            "Every requires_grad parameter participates with dense gradient "
            "and two same-dtype Adam/AdamW moments; excludes optimizer "
            "step scalars and implementation-specific fused workspaces"
        ),
        "logical_bytes_by_dtype": dict(sorted(dtype_bytes.items())),
        "logical_bytes_by_device": dict(sorted(device_bytes.items())),
        "memory_scope": (
            "Logical model parameter and buffer tensor payload, not "
            "unique backing storage or peak allocated CPU/GPU RAM"
        ),
        "excluded": [
            "training/validation batches and activations",
            "gradients, optimizer moments, temporary growth clones, RNG",
            "autograd intermediates, attention workspaces, CUDA allocator overhead",
            "checkpoint staging and CPU/GPU reserved memory",
        ],
        "peak_training_memory_bytes": None,
        "unique_allocator_storage_bytes": None,
    }


def projected_tabular_mlp_payload_bytes(
    model: nn.Module, hidden_widths: tuple[int, ...],
) -> int | None:
    """Predict bytes for a plain TabularMLP rebuild *without constructing it*.

    Returns None for unusual parameter layouts or an inconsistent default
    dtype: those cases must use the existing postconstruction check. This
    forecast covers nn.Linear weights/biases and optional BatchNorm1d
    affine parameters, running mean/variance, and int64 counters. It
    deliberately does not attempt to forecast custom/aliased modules.
    """
    from .models import TabularMLP
    import torch

    if not isinstance(model, TabularMLP):
        raise TypeError("projected model must be a TabularMLP")
    widths = tuple(int(w) for w in hidden_widths)
    if not widths or min(widths) <= 0:
        raise ValueError("projected hidden widths must be positive")
    parameters = tuple(model.parameters())
    if not parameters:
        return None
    dtype = torch.get_default_dtype()
    if any(p.dtype != dtype for p in parameters):
        return None
    elt_bytes = int(parameters[0].element_size())

    def count(target: tuple[int, ...]) -> int:
        size = 0
        previous = int(model.input_dim)
        for width in target:
            size += (previous + 1) * width * elt_bytes
            if model.batch_norm:
                # gamma/beta + running mean/var; each BN has int64 batches.
                size += 4 * width * elt_bytes + 8
            previous = width
        size += (previous + 1) * int(model.output_dim) * elt_bytes
        return size

    current_bytes = visual_model_tensor_payload(model)["total_model_tensor_payload_bytes"]
    if count(tuple(model.hidden_widths)) != current_bytes:
        return None
    return int(count(widths))


def prospective_tabular_growth_widths(
    model: nn.Module, *,
    operator: str, increment: int = 1, max_width: int = 512,
    max_depth: int = 5, seed_width: int = 1,
) -> tuple[int, ...] | None:
    """Mirror native width/depth *geometry*, never allocate growth weights.

    The caller still performs the source growth admission rules, neuron
    caps, source-specific structural validation and actual post-build
    byte check. This is only a preallocation memory screen.
    """
    from .models import TabularMLP

    if not isinstance(model, TabularMLP):
        raise TypeError("prospective growth requires a TabularMLP")
    if min(increment, max_width, max_depth, seed_width) <= 0:
        raise ValueError("growth geometry limits must be positive")
    widths = list(model.hidden_widths)
    cursor = int(getattr(model, "_width_cursor", 0)) % len(widths)
    if operator == "all_layers":
        grown = [min(max_width, x + increment) for x in widths]
        return tuple(grown) if grown != widths else None
    if operator in {"round_robin", "den_round_robin"}:
        target = min(widths) + increment
        for offset in range(len(widths)):
            index = (cursor + offset) % len(widths)
            if operator == "den_round_robin":
                if widths[index] >= target or widths[index] >= max_width:
                    continue
                if widths[index] + increment > max_width:
                    return None
                widths[index] += increment
            else:
                if widths[index] >= max_width:
                    continue
                widths[index] = min(max_width, widths[index] + increment)
            return tuple(widths)
        return None
    if operator == "D1":
        if len(widths) >= max_depth or len(set(widths)) > 1:
            return None
        return (*widths, seed_width)
    if operator == "Dcur":
        if len(widths) >= max_depth:
            return None
        return (*widths, widths[-1])
    raise ValueError(f"unsupported source growth operator {operator!r}")


def forecast_visual_growth_payload_bytes(
    model: nn.Module, new_hidden_widths: tuple[int, ...],
    *, other_hidden_widths: tuple[int, ...] | None = None,
) -> int | None:
    """Project candidate model payload by MLP deltas, before growth copies.

    Applies to native single-site and synchronous V07 mixer-both growth.
    The unknown result deliberately falls back to post-build admission.
    """
    if not isinstance(model, nn.Module) or not hasattr(model, "local_mlp"):
        raise TypeError("forecast requires a built native visual composite")
    local = model.local_mlp
    new_local = projected_tabular_mlp_payload_bytes(local, new_hidden_widths)
    if new_local is None:
        return None
    current_local = visual_model_tensor_payload(local)["total_model_tensor_payload_bytes"]
    predicted = (
        visual_model_tensor_payload(model)["total_model_tensor_payload_bytes"]
        - current_local + new_local
    )
    if other_hidden_widths is not None:
        if not hasattr(model, "other_mlp"):
            raise ValueError("synchronous growth requires other_mlp")
        other = model.other_mlp
        new_other = projected_tabular_mlp_payload_bytes(other, other_hidden_widths)
        if new_other is None:
            return None
        old_other = visual_model_tensor_payload(other)["total_model_tensor_payload_bytes"]
        predicted += new_other - old_other
    return int(predicted)
