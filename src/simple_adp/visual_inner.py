"""V02 experimental CNN residual MLP adapter with genuine *local* ADP mutation.

Unlike V01 pooled embeddings, trainable dense hidden layers here sit inside
the image backbone. Frozen downstream operations remain in autograd. This
module supplies candidate construction, native growth and StageTrainer
compatibility; full canonical controller/runner integration remains pending.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from .growth import (
    expand_depth, expand_depth_match_width,
    expand_width_all_layers, expand_width_round_robin,
    expand_width_den_round_robin,
)
from .models import TabularMLP

WidthOperator = Literal["round_robin", "all_layers", "den_round_robin"]
DepthOperator = Literal["D1", "Dcur"]


@dataclass(frozen=True)
class ResidualAdapterSpec:
    input_channels: int
    backbone_channels: int
    local_dim: int
    classes: int
    image_height: int
    image_width: int
    fixed_seed: int = 17
    site_id: str = "adapter:block1"

    def __post_init__(self):
        if min(
            self.input_channels, self.backbone_channels, self.local_dim,
            self.classes, self.image_height, self.image_width,
        ) <= 0:
            raise ValueError("positive image channels, interface dims, classes and resolution required")
        if not self.site_id.startswith("adapter:") or self.site_id == "adapter:":
            raise ValueError("residual adapter site_id must begin with adapter:")


class VisualResidualAdapter(nn.Module):
    """Frozen CNN with local trainable dense MLP inside residual adapter.

    Computation: h=pre(x); h'=h + P_out(MLP(P_in(h))); y=head(pool(post(h'))).
    P_in,P_out and both CNN fragments/head are exactly frozen. A nonzero
    fixed outgoing projection is used to avoid zero first-step MLP gradients;
    this does *not* promise initial function preservation.
    """

    def __init__(
        self, spec: ResidualAdapterSpec,
        hidden_widths: tuple[int, ...] = (4,),
    ) -> None:
        super().__init__()
        if not hidden_widths or min(hidden_widths) <= 0:
            raise ValueError("local hidden widths must be positive")
        self.spec = spec
        # Isolate initial fixed topology weights from the source ADP RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(spec.fixed_seed))
            self.pre = nn.Sequential(
                nn.Conv2d(spec.input_channels, spec.backbone_channels, 3, padding=1),
                nn.BatchNorm2d(spec.backbone_channels), nn.ReLU(),
            )
            self.project_in = nn.Linear(spec.backbone_channels, spec.local_dim)
            self.project_out = nn.Linear(spec.local_dim, spec.backbone_channels)
            self.post = nn.Sequential(
                nn.Conv2d(spec.backbone_channels, spec.backbone_channels, 3, padding=1),
                nn.BatchNorm2d(spec.backbone_channels), nn.ReLU(),
            )
            self.classifier = nn.Linear(spec.backbone_channels, spec.classes)
        self.local_mlp = TabularMLP(
            spec.local_dim, tuple(int(x) for x in hidden_widths), spec.local_dim,
            batch_norm=False,
        )
        for module in self._fixed_modules():
            module.requires_grad_(False)
            module.eval()
        self.local_mlp.train(True)
        self._frozen_baseline_sha256 = self.frozen_state_sha256()
        self._birth_generation = 0

    def _fixed_modules(self) -> tuple[nn.Module, ...]:
        return self.pre, self.project_in, self.project_out, self.post, self.classifier

    @property
    def growth_targets(self) -> dict[str, TabularMLP]:
        return {self.spec.site_id: self.local_mlp}

    @property
    def hidden_widths(self) -> list[int]:
        return list(self.local_mlp.hidden_widths)

    @property
    def input_dim(self) -> int:
        """Local interface dimension, *not* image C*H*W."""
        return self.local_mlp.input_dim

    @property
    def output_dim(self) -> int:
        """Local MLP output dimension; supervised logits have size spec.classes."""
        return self.local_mlp.output_dim

    @property
    def batch_norm(self) -> bool:
        return self.local_mlp.batch_norm

    @property
    def _width_cursor(self) -> int:
        return int(getattr(self.local_mlp, "_width_cursor", 0))

    @_width_cursor.setter
    def _width_cursor(self, value: int) -> None:
        setattr(self.local_mlp, "_width_cursor", int(value))

    def train(self, mode: bool = True) -> "VisualResidualAdapter":
        super().train(mode)
        # Frozen weights alone do not freeze BN statistics/dropout mode.
        for module in self._fixed_modules():
            module.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != self.spec.input_channels:
            raise ValueError("V02 images must have [B,C,H,W] with declared channels")
        if tuple(images.shape[-2:]) != (self.spec.image_height, self.spec.image_width):
            raise ValueError("V02 image resolution differs from fixed architecture")
        x = self.pre(images)
        x = x.permute(0, 2, 3, 1).contiguous()
        delta = self.project_out(self.local_mlp(self.project_in(x)))
        x = (x + delta).permute(0, 3, 1, 2).contiguous()
        # Do not wrap frozen *downstream* post/classifier in no_grad(): their
        # Jacobians must carry supervised gradients to the inner MLP.
        x = self.post(x)
        return self.classifier(x.mean(dim=(-2, -1)))

    def frozen_state_sha256(self) -> str:
        digest = hashlib.sha256()
        for group, module in zip(
            ("pre", "pin", "pout", "post", "classifier"),
            self._fixed_modules(),
        ):
            for name, tensor in sorted(module.state_dict().items()):
                arr = tensor.detach().cpu().contiguous().numpy()
                digest.update((group + ":" + name + ":" + str(arr.dtype) + ":" + str(arr.shape)).encode())
                digest.update(arr.tobytes())
        return digest.hexdigest()

    def verify_frozen(self) -> None:
        if self.frozen_state_sha256() != self._frozen_baseline_sha256:
            raise ValueError("V02 frozen modules or their buffers were mutated")
        if any(p.requires_grad for m in self._fixed_modules() for p in m.parameters()):
            raise ValueError("V02 frozen modules became trainable")
        if any(m.training for m in self._fixed_modules()):
            raise ValueError("V02 frozen modules must remain eval")

    def grow_width(self, *, site_id: str, operator: WidthOperator,
                   increment: int, max_width: int) -> "VisualResidualAdapter | None":
        if site_id != self.spec.site_id:
            raise ValueError(f"unknown or noneligible V02 growth site {site_id!r}")
        if operator == "round_robin":
            local = expand_width_round_robin(self.local_mlp, increment, max_width)
        elif operator == "all_layers":
            local = expand_width_all_layers(self.local_mlp, increment, max_width)
        elif operator == "den_round_robin":
            local = expand_width_den_round_robin(self.local_mlp, increment, max_width)
        else:
            raise ValueError(f"invalid source width operator {operator!r}")
        if local is None:
            return None
        candidate = copy.deepcopy(self)
        candidate.local_mlp = local
        candidate._birth_generation = self._birth_generation + 1
        candidate.train(self.training)
        candidate.verify_frozen()
        if candidate.input_dim != self.input_dim or candidate.output_dim != self.output_dim:
            raise AssertionError("V02 mutation changed a fixed local interface")
        return candidate

    def grow_depth(self, *, site_id: str, operator: DepthOperator,
                   max_depth: int, seed_width: int = 1) -> "VisualResidualAdapter | None":
        if site_id != self.spec.site_id:
            raise ValueError(f"unknown V02 growth site {site_id!r}")
        if operator == "D1":
            local = expand_depth(self.local_mlp, max_depth, min_new_layer_width=seed_width)
        elif operator == "Dcur":
            local = expand_depth_match_width(self.local_mlp, max_depth)
        else:
            raise ValueError(f"invalid source depth operator {operator!r}")
        if local is None:
            return None
        candidate = copy.deepcopy(self)
        candidate.local_mlp = local
        candidate._birth_generation = self._birth_generation + 1
        candidate.train(self.training)
        candidate.verify_frozen()
        if candidate.input_dim != self.input_dim or candidate.output_dim != self.output_dim:
            raise AssertionError("V02 depth changed a fixed local interface")
        return candidate

    def snapshot(self, *, cursor: int = 0) -> dict[str, object]:
        self.verify_frozen()
        return {
            "kind": "visual-residual-adapter-v02",
            "spec": {
                "input_channels": self.spec.input_channels,
                "backbone_channels": self.spec.backbone_channels,
                "local_dim": self.spec.local_dim,
                "classes": self.spec.classes,
                "image_height": self.spec.image_height,
                "image_width": self.spec.image_width,
                "fixed_seed": self.spec.fixed_seed,
                "site_id": self.spec.site_id,
            },
            "input_dim": self.input_dim, "output_dim": self.output_dim,
            "hidden_widths": list(self.hidden_widths), "batch_norm": self.batch_norm,
            "growth_site": self.spec.site_id, "cursor": int(cursor),
            "generation": int(self._birth_generation),
            "frozen_sha256": self._frozen_baseline_sha256,
            "state_dict": copy.deepcopy(self.state_dict()),
        }

    @classmethod
    def from_snapshot(cls, payload: dict[str, object]) -> "VisualResidualAdapter":
        if payload.get("kind") != "visual-residual-adapter-v02":
            raise ValueError("unsupported V02 snapshot kind")
        spec = ResidualAdapterSpec(**payload["spec"])
        saved_rng = torch.get_rng_state()
        try:
            model = cls(spec, tuple(int(x) for x in payload["hidden_widths"]))
        finally:
            torch.set_rng_state(saved_rng)
        model.load_state_dict(payload["state_dict"], strict=True)
        model._width_cursor = int(payload.get("cursor", 0))
        model._birth_generation = int(payload.get("generation", 0))
        model._frozen_baseline_sha256 = str(payload["frozen_sha256"])
        model.train(False)
        model.verify_frozen()
        if model.input_dim != int(payload["input_dim"]) or model.output_dim != int(payload["output_dim"]):
            raise ValueError("restored V02 interface is incompatible")
        return model
