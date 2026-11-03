"""Count *executed training forwards* without performing a second forward.

Module hooks observe actual per-minibatch nn.Linear/nn.Conv2d work and use
query/key/value shapes for a separately labelled dense MultiheadAttention
estimate. They do not inspect gradients, alter training modes, consume RNG,
profile accelerator kernels, or claim backward/optimizer/full-model FLOPs.
Only forwards explicitly bracketed by record_batch count: validation,
checkpoint reconstruction and shadow-variance extra passes are excluded.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn


_FORMAT = "simple-adp-actual-training-forward-partial-macs-v1"
_KEYS = (
    "observed_linear_conv_training_forward_macs",
    "analytical_mha_training_forward_macs",
    "recorded_training_forward_batches",
    "recorded_training_samples",
)


class VisualTrainingForwardMACCounter:
    """Hooks onto one stage model, with rollback on failed forwards.

    Counter resumes from *signed stage state*. An older checkpoint lacking
    counters is incomplete rather than silently treating past work as free.
    """

    def __init__(
        self, model: nn.Module, *, resume: Mapping[str, object] | None = None,
        prior_batches: int = 0,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("training forward observer requires nn.Module")
        if type(prior_batches) is not int or prior_batches < 0:
            raise ValueError("prior_batches must be a nonnegative integer")
        self.model = model
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._started = False
        self._active = False
        self._pending_linear_conv = 0
        self._pending_mha = 0
        self._counts = {key: 0 for key in _KEYS}
        self.complete = True
        if resume is None and prior_batches:
            self.complete = False
        elif resume is not None:
            if resume.get("format") != _FORMAT or type(resume.get("complete")) is not bool:
                raise ValueError("invalid visual training forward MAC checkpoint format")
            self.complete = bool(resume["complete"])
            for key in _KEYS:
                value = resume.get(key)
                if type(value) is not int or value < 0:
                    raise ValueError(f"invalid signed training MAC counter {key}")
                self._counts[key] = value
            if self.complete and self._counts["recorded_training_forward_batches"] != prior_batches:
                raise ValueError("visual training MAC count does not match resumed optimizer batches")
            if self._counts["recorded_training_forward_batches"] > prior_batches:
                raise ValueError("visual training MAC count exceeds resumed optimizer batches")

    def __enter__(self) -> "VisualTrainingForwardMACCounter":
        if self._started:
            raise RuntimeError("training MAC observer is already attached")
        self._started = True
        modules = dict(self.model.named_modules())
        for name, module in modules.items():
            if isinstance(module, nn.Linear):
                parent_name = name.rsplit(".", 1)[0] if "." in name else ""
                if name.endswith(".out_proj") and isinstance(modules.get(parent_name), nn.MultiheadAttention):
                    # PyTorch MHA runs its out projection functionally; the
                    # parent MHA hook accounts for it exactly once.
                    continue

                def on_linear(layer, inputs, output, *, _name=name):
                    if not self._active:
                        return
                    if not isinstance(output, torch.Tensor) or output.shape[-1] != layer.out_features:
                        raise ValueError(f"unexpected Linear output at {_name}")
                    self._pending_linear_conv += int(output.numel() * layer.in_features)
                self._handles.append(module.register_forward_hook(on_linear))

            elif isinstance(module, nn.Conv2d):
                def on_conv(layer, inputs, output, *, _name=name):
                    if not self._active:
                        return
                    if not isinstance(output, torch.Tensor) or output.ndim != 4:
                        raise ValueError(f"unexpected Conv2d output at {_name}")
                    kernel = layer.kernel_size[0] * layer.kernel_size[1]
                    self._pending_linear_conv += int(
                        output.numel() * (layer.in_channels // layer.groups) * kernel
                    )
                self._handles.append(module.register_forward_hook(on_conv))

            elif isinstance(module, nn.MultiheadAttention):
                def on_mha(layer, inputs, output):
                    if not self._active:
                        return
                    query, key, value = inputs[:3]
                    if query.ndim == 3:
                        if layer.batch_first:
                            batch, qlen, qdim = query.shape
                            kb, klen, kdim = key.shape
                            vb, vlen, vdim = value.shape
                        else:
                            qlen, batch, qdim = query.shape
                            klen, kb, kdim = key.shape
                            vlen, vb, vdim = value.shape
                        if batch != kb or batch != vb:
                            raise ValueError("MultiheadAttention batch mismatch")
                    elif query.ndim == 2:
                        batch, qlen, qdim = 1, *query.shape
                        klen, kdim = key.shape
                        vlen, vdim = value.shape
                    else:
                        raise ValueError("unsupported MultiheadAttention sequence rank")
                    if qdim != layer.embed_dim or klen != vlen:
                        raise ValueError("inconsistent MultiheadAttention geometry")
                    e = int(layer.embed_dim)
                    q, k, b = int(qlen), int(klen), int(batch)
                    projections = b * (q * int(qdim) * e + k * int(kdim) * e
                                       + k * int(vdim) * e + q * e * e)
                    keff = k + int(layer.add_zero_attn) + (1 if layer.bias_k is not None else 0)
                    self._pending_mha += projections + 2 * b * q * keff * e
                self._handles.append(module.register_forward_hook(on_mha))
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._active = False
        self._started = False
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @contextmanager
    def record_batch(self, samples: int) -> Iterator[None]:
        """Commit exactly one successful training *forward*, never validation.

        No forward is executed here: the caller performs its normal forward
        once inside the context. Exceptions discard that partial batch count.
        """
        if not self._started or self._active:
            raise RuntimeError("attach observer first; nested training batches are invalid")
        if type(samples) is not int or samples <= 0:
            raise ValueError("training forward batch size must be a positive integer")
        self._pending_linear_conv = 0
        self._pending_mha = 0
        self._active = True
        try:
            yield
        except BaseException:
            raise
        else:
            self._counts["observed_linear_conv_training_forward_macs"] += self._pending_linear_conv
            self._counts["analytical_mha_training_forward_macs"] += self._pending_mha
            self._counts["recorded_training_forward_batches"] += 1
            self._counts["recorded_training_samples"] += samples
        finally:
            self._active = False
            self._pending_linear_conv = 0
            self._pending_mha = 0

    def snapshot(self) -> dict[str, object]:
        observed = self._counts["observed_linear_conv_training_forward_macs"]
        analytical = self._counts["analytical_mha_training_forward_macs"]
        return {
            "format": _FORMAT,
            **dict(self._counts),
            "partial_training_forward_macs": observed + analytical,
            "complete": self.complete,
            "scope": "only actual main model(xb) training forwards, not validation/shadow/replay",
            "backward_flops": None,
            "optimizer_flops": None,
            "full_training_flops": None,
            "attention_kind": "shape-based dense MHA, not observed fused CUDA kernel cost",
        }


def summarize_visual_training_forward_trace(
    records: list[Mapping[str, object]], *, training_stages: int,
    complete: bool,
) -> dict[str, object]:
    """Aggregate only observed stage forwards, with no invented prior work."""
    if type(training_stages) is not int or training_stages < 0 or type(complete) is not bool:
        raise ValueError("invalid signed visual training stage-count provenance")
    if len(records) > training_stages or (complete and len(records) != training_stages):
        raise ValueError("observed forward trace does not match signed training stage count")
    total = {key: 0 for key in _KEYS}
    for row in records:
        if row.get("format") != _FORMAT or type(row.get("complete")) is not bool:
            raise ValueError("visual training forward trace contains invalid format")
        if complete and not row["complete"]:
            raise ValueError("incomplete stage cannot establish complete training work")
        for key in _KEYS:
            x = row.get(key)
            if type(x) is not int or x < 0:
                raise ValueError(f"invalid observed training forward trace field {key}")
            total[key] += x
        if row.get("partial_training_forward_macs") != (
            row["observed_linear_conv_training_forward_macs"]
            + row["analytical_mha_training_forward_macs"]
        ):
            raise ValueError("visual forward MAC components do not close")
    return {
        "format": "simple-adp-native-actual-training-forward-mac-sum-v1",
        **total,
        "partial_training_forward_macs": (
            total["observed_linear_conv_training_forward_macs"]
            + total["analytical_mha_training_forward_macs"]
        ),
        "observed_training_stages": len(records),
        "expected_training_stages": training_stages,
        "complete": complete,
        "full_training_flops": None,
        "backward_flops": None,
        "optimizer_flops": None,
        "training_compute_scope": (
            "actual main model(xb) forward calls only; nn.Linear/Conv2d "
            "plus analytically estimated dense nn.MultiheadAttention"
        ),
    }
