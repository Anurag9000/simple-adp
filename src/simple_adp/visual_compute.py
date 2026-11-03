"""Observed dense/conv inference MACs for one visual image, not full FLOPs.

Execute the actual selected composite once with forward hooks. This counts
nn.Linear and nn.Conv2d invocations, including their spatial/token reuse, and
does not infer convolution or attention expansion from an MLP route name.
Uninstrumented attention kernels, complex FFT, normalization, pooling and
elementwise operations are explicitly excluded. No accuracy claim or total
FLOP estimate may be inferred from the returned count.
"""
from __future__ import annotations

from collections import defaultdict

import torch
from torch import nn


def visual_linear_conv_macs(
    model: nn.Module, image: torch.Tensor,
) -> dict[str, object]:
    """Count observed Linear/Conv2d dense MACs per *one* NCHW image.

    Never changes parameters, buffers, trainable flags, or the caller's
    module training flags/RNG states. A forward with gradients disabled is
    nevertheless a real composite-model execution, not an isolated local
    MLP proxy. This function does not claim complete computational coverage.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be nn.Module")
    if not torch.is_tensor(image) or image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("provide exactly one tensor image [1,C,H,W]")
    if not image.is_floating_point() or not bool(torch.isfinite(image).all()):
        raise ValueError("inference image must contain finite floating-point pixels")
    if min(image.shape[1:]) <= 0:
        raise ValueError("image channels/resolution must be positive")
    named_modules = list(model.named_modules())
    parameters = tuple(model.parameters())
    buffers = tuple(model.buffers())
    reference = next(iter(parameters), None)
    if reference is None:
        reference = next(iter(buffers), None)
    device = reference.device if reference is not None else image.device
    if image.device != device:
        raise ValueError("image and model must already be on the same device")
    before_modes = [(module, bool(module.training)) for _, module in named_modules]
    before_trainability = [(p, bool(p.requires_grad)) for p in parameters]
    macs: dict[str, int] = defaultdict(int)
    calls: dict[str, int] = defaultdict(int)
    handles = []

    def on_linear(name: str):
        def observe(module: nn.Linear, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if not isinstance(output, torch.Tensor) or output.shape[-1] != module.out_features:
                raise ValueError(f"Linear output mismatch at {name}")
            macs[name] += int(output.numel() * module.in_features)
            calls[name] += 1
        return observe

    def on_conv(name: str):
        def observe(module: nn.Conv2d, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if not isinstance(output, torch.Tensor) or output.ndim != 4:
                raise ValueError(f"Conv2d output mismatch at {name}")
            kernel = int(module.kernel_size[0]) * int(module.kernel_size[1])
            cin_group = int(module.in_channels) // int(module.groups)
            macs[name] += int(output.numel() * cin_group * kernel)
            calls[name] += 1
        return observe

    for name, module in named_modules:
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(on_linear(name)))
        elif isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(on_conv(name)))

    cuda_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda" else []
    )
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            model.eval()
            with torch.no_grad():
                result = model(image)
            if not isinstance(result, torch.Tensor) or result.shape[0] != 1:
                raise ValueError("visual composite must return batch-aligned logits")
    finally:
        for handle in handles:
            handle.remove()
        # Set flags directly to avoid subclass train() reconfiguration.
        for module, flag in before_modes:
            module.training = flag
        for parameter, flag in before_trainability:
            if parameter.requires_grad != flag:
                parameter.requires_grad_(flag)
    items = {
        name: {"macs": int(macs[name]), "forward_calls": int(calls[name])}
        for name in sorted(macs)
    }
    return {
        "format": "simple-adp-observed-linear-conv2d-macs-v1",
        "input_shape": list(image.shape),
        "batch_size": 1,
        "observed_linear_conv_macs_per_image": int(sum(macs.values())),
        "observed_modules": items,
        "covered_ops": ["nn.Linear", "nn.Conv2d"],
        "excluded_ops": [
            "functional linear/matmul and MultiheadAttention attention projections",
            "attention score/value products and fused attention kernels",
            "FFT/IFFT and complex Fourier filtering",
            "pooling/normalization/activation/elementwise/residual/interpolation",
            "memory movement, backward passes, optimizer and search",
        ],
        "whole_model_flops": None,
        "coverage": "partial observed inference MACs; NOT whole-model FLOPs",
    }
