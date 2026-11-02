"""Distinct parameter-free spatial shift operators for V06a and V06b.

The shifts are spatial, preserve channel count and resolution and use zero
boundary conditions (never circularly wrap neighboring patches). There are no
trainable weights, running buffers or randomized operations in either mixer.
"""
from __future__ import annotations

import torch
from torch import nn


def _zero_shift(x: torch.Tensor, delta_h: int, delta_w: int) -> torch.Tensor:
    """Shift spatial positions with zeros rather than torch.roll wraparound."""
    if x.ndim != 4:
        raise ValueError("spatial-shift mixer expects [B,C,H,W]")
    h, w = x.shape[-2:]
    if abs(delta_h) >= h or abs(delta_w) >= w:
        return torch.zeros_like(x)
    out = torch.zeros_like(x)
    src_h = slice(0, h - delta_h) if delta_h >= 0 else slice(-delta_h, h)
    dst_h = slice(delta_h, h) if delta_h >= 0 else slice(0, h + delta_h)
    src_w = slice(0, w - delta_w) if delta_w >= 0 else slice(-delta_w, w)
    dst_w = slice(delta_w, w) if delta_w >= 0 else slice(0, w + delta_w)
    out[:, :, dst_h, dst_w] = x[:, :, src_h, src_w]
    return out


class S2SpatialShift(nn.Module):
    """S²-MLP: split channel groups and shift up/down/left/right by one."""

    def __init__(self, channels: int):
        super().__init__()
        if channels <= 0 or channels % 4:
            raise ValueError("S2 shift requires a positive multiple-of-four channel count")
        self.channels = int(channels)

    def extra_repr(self) -> str:
        return f"channels={self.channels},groups=4,offset=1,boundary=zero"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError("S2 shift expects fixed [B,C,H,W] channels")
        groups = x.chunk(4, dim=1)
        return torch.cat((
            _zero_shift(groups[0], -1, 0),
            _zero_shift(groups[1], 1, 0),
            _zero_shift(groups[2], 0, -1),
            _zero_shift(groups[3], 0, 1),
        ), dim=1)


class AxialSpatialShift(nn.Module):
    """AS-MLP-style distinct height and width branches with equal mixing.

    The first half of channels moves negatively and the second positively.
    Height- and width-axis branches are averaged, unlike V06a's one-direction
    per channel. Original AS-MLP trainable surrounding projections are frozen
    or absent in this bounded from-scratch control.
    """

    def __init__(self, channels: int):
        super().__init__()
        if channels <= 0 or channels % 2:
            raise ValueError("axial shift requires a positive even channel count")
        self.channels = int(channels)

    def extra_repr(self) -> str:
        return f"channels={self.channels},axes=height+width,offset=1,boundary=zero"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError("axial shift expects fixed [B,C,H,W] channels")
        first, second = x.chunk(2, dim=1)
        vertical = torch.cat((
            _zero_shift(first, -1, 0), _zero_shift(second, 1, 0),
        ), dim=1)
        horizontal = torch.cat((
            _zero_shift(first, 0, -1), _zero_shift(second, 0, 1),
        ), dim=1)
        return (vertical + horizontal) * 0.5
