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
