"""Observed dense/conv inference MACs for one visual image, not full FLOPs.

Execute the actual selected composite once with forward hooks. This counts
nn.Linear and nn.Conv2d invocations, including their spatial/token reuse, and
does not infer convolution or attention expansion from an MLP route name.
MultiheadAttention dense projection and score/value MACs are analytically
estimated from the observed query/key/value tensor shapes (not kernel
instrumentation). Complex FFT, normalization, pooling and elementwise
operations are explicitly excluded. No accuracy claim or total
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
    attention_macs: dict[str, dict[str, int]] = {}
    attention_calls: dict[str, int] = defaultdict(int)
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

    def on_attention(name: str):
        def observe(
            module: nn.MultiheadAttention,
            inputs: tuple[torch.Tensor, ...],
            output: tuple[torch.Tensor, ...],
        ) -> None:
            # This is a shape-based standard *dense* attention MAC estimate;
            # fused scaled_dot_product_attention does not expose its kernel.
            # Masking, softmax, dropout, bias addition and implementation
            # choice are not counted, and sparse-kernel savings are unknown.
            query, key, value = inputs[:3]
            if query.ndim == 3:
                if module.batch_first:
                    batch, q_len, q_dim = query.shape
                    k_batch, kv_len, k_dim = key.shape
                    v_batch, v_len, v_dim = value.shape
                else:
                    q_len, batch, q_dim = query.shape
                    kv_len, k_batch, k_dim = key.shape
                    v_len, v_batch, v_dim = value.shape
                if k_batch != batch or v_batch != batch:
                    raise ValueError("MHA batch sizes must agree")
            elif query.ndim == 2:
                batch = 1
                q_len, q_dim = query.shape
                kv_len, k_dim = key.shape
                v_len, v_dim = value.shape
            else:
                raise ValueError("MHA probe expects batched or unbatched query")
            if v_len != kv_len or q_dim != module.embed_dim:
                raise ValueError("MHA probe tensor geometry is inconsistent")
            embed_dim = int(module.embed_dim)
            # Separate Q/K/V projections and the output dense projection.
            proj = int(batch) * (
                int(q_len) * int(q_dim) * embed_dim
                + int(kv_len) * int(k_dim) * embed_dim
                + int(kv_len) * int(v_dim) * embed_dim
                + int(q_len) * embed_dim * embed_dim
            )
            # Synthetic key/value slots are not learned dense projections.
            effective_kv_len = int(kv_len) + int(module.add_zero_attn) + (
                1 if module.bias_k is not None else 0
            )
            score_value = 2 * int(batch) * int(q_len) * effective_kv_len * embed_dim
            row = attention_macs.setdefault(name, {
                "qkv_and_output_projection_macs": 0,
                "score_and_value_product_macs": 0,
                "total_dense_attention_macs": 0,
            })
            row["qkv_and_output_projection_macs"] += proj
            row["score_and_value_product_macs"] += score_value
            row["total_dense_attention_macs"] += proj + score_value
            attention_calls[name] += 1
        return observe

    for name, module in named_modules:
        if isinstance(module, nn.Linear):
            # PyTorch MultiheadAttention executes output projection using
            # torch.nn.functional.linear. Count its projection analytically
            # with the parent MHA, never as a second nn.Linear call.
            parent = name.rsplit(".", 1)[0] if "." in name else ""
            parent_module = dict(named_modules).get(parent)
            if isinstance(parent_module, nn.MultiheadAttention) and name.endswith(".out_proj"):
                continue
            handles.append(module.register_forward_hook(on_linear(name)))
        elif isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(on_conv(name)))
        elif isinstance(module, nn.MultiheadAttention):
            handles.append(module.register_forward_hook(on_attention(name)))

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
    mha_items = {
        name: {**attention_macs[name], "forward_calls": int(attention_calls[name])}
        for name in sorted(attention_macs)
    }
    mha_total = sum(item["total_dense_attention_macs"] for item in mha_items.values())
    return {
        "format": "simple-adp-observed-linear-conv2d-plus-analytical-mha-macs-v2",
        "input_shape": list(image.shape),
        "batch_size": 1,
        "observed_linear_conv_macs_per_image": int(sum(macs.values())),
        "observed_modules": items,
        "analytical_multihead_attention_macs_per_image": int(mha_total),
        "analytical_multihead_attention_modules": mha_items,
        "partial_dense_attention_inclusive_macs_per_image": int(sum(macs.values()) + mha_total),
        "covered_ops": [
            "observed nn.Linear", "observed nn.Conv2d",
            "shape-analytical dense nn.MultiheadAttention Q/K/V/out projections and QK^T/AV",
        ],
        "excluded_ops": [
            "other functional linear/matmul and fused attention kernels not declared as nn.MultiheadAttention",
            "attention softmax, mask and sparse-kernel optimizations",
            "FFT/IFFT and complex Fourier filtering",
            "pooling/normalization/activation/elementwise/residual/interpolation",
            "memory movement, backward passes, optimizer and search",
        ],
        "whole_model_flops": None,
        "coverage": "partial observed inference MACs; NOT whole-model FLOPs",
    }
