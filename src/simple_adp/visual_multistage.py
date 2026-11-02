"""V08 fixed early/middle/late residual adapters with source-local ADP growth.

Every job declares exactly one preinstalled site. Other site modules and the
original 3-stage CNN remain fixed; all operate in the differentiable image
forward. Multi-site simultaneous selection is separately documented as future
work and is never silently implied by ADP-W-All.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .models import TabularMLP
from .visual_inner import VisualResidualAdapter


SITES = ("early", "middle", "late")


@dataclass(frozen=True)
class MultistageSpec:
    input_channels: int
    backbone_channels: int
    classes: int
    image_height: int
    image_width: int
    site_id: str
    fixed_seed: int = 41
    fixed_other_width: int = 6

    def __post_init__(self):
        if self.site_id not in {f"adapter:{x}" for x in SITES}:
            raise ValueError("V08 growth site must be adapter:early/middle/late")
        for name in ("input_channels", "backbone_channels", "classes",
                     "image_height", "image_width", "fixed_other_width"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


class VisualMultistage(VisualResidualAdapter):
    """Fixed three-stage CNN with one eligible internal residual MLP."""

    SNAPSHOT_KIND = "visual-multistage-v08-single-site-v1"

    def __init__(self, spec: MultistageSpec, hidden_widths: tuple[int, ...] = (4,)):
        nn.Module.__init__(self)
        if not hidden_widths or min(hidden_widths) <= 0:
            raise ValueError("V08 local hidden widths must be positive")
        self.spec = spec
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(spec.fixed_seed)
            self.stem = nn.Conv2d(spec.input_channels, spec.backbone_channels, 3, padding=1)
            self.stage_convs = nn.ModuleList([
                nn.Conv2d(spec.backbone_channels, spec.backbone_channels, 3, padding=1)
                for _ in SITES
            ])
            self.fixed_adapters = nn.ModuleDict({
                site: TabularMLP(
                    spec.backbone_channels, (spec.fixed_other_width,),
                    spec.backbone_channels,
                )
                for site in SITES if f"adapter:{site}" != spec.site_id
            })
            self.classifier = nn.Linear(spec.backbone_channels, spec.classes)
        self.local_mlp = TabularMLP(
            spec.backbone_channels, hidden_widths, spec.backbone_channels,
        )
        self._gelu_hidden()
        for module in self._fixed_modules():
            module.requires_grad_(False)
            module.eval()
        self._birth_generation = 0
        self._frozen_baseline_sha256 = self.frozen_state_sha256()

    def _gelu_hidden(self) -> None:
        for i, layer in enumerate(self.local_mlp.hidden_layers):
            if isinstance(layer, nn.ReLU):
                self.local_mlp.hidden_layers[i] = nn.GELU()
        for mlp in self.fixed_adapters.values():
            for i, layer in enumerate(mlp.hidden_layers):
                if isinstance(layer, nn.ReLU):
                    mlp.hidden_layers[i] = nn.GELU()

    def _fixed_modules(self) -> tuple[nn.Module, ...]:
        return (self.stem, self.stage_convs, self.fixed_adapters, self.classifier)

    def frozen_state_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update("V08:early-middle-late-3x3-conv".encode())
        for name, module in zip(
            ("stem", "convs", "other_adapters", "classifier"), self._fixed_modules(),
        ):
            for key, tensor in sorted(module.state_dict().items()):
                arr = tensor.detach().cpu().contiguous().numpy()
                digest.update(f"{name}:{key}:{arr.dtype}:{arr.shape}".encode())
                digest.update(arr.tobytes())
        return digest.hexdigest()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if (images.ndim != 4 or images.shape[1] != self.spec.input_channels
                or tuple(images.shape[-2:]) !=
                (self.spec.image_height, self.spec.image_width)):
            raise ValueError("V08 images must match fixed NCHW channels and resolution")
        h = self.stem(images)
        for index, site in enumerate(SITES):
            h = torch.nn.functional.gelu(self.stage_convs[index](h))
            local = self.local_mlp if f"adapter:{site}" == self.spec.site_id else self.fixed_adapters[site]
            channels_last = h.permute(0, 2, 3, 1).contiguous()
            delta = local(channels_last).permute(0, 3, 1, 2).contiguous()
            h = h + 0.1 * delta
        return self.classifier(h.mean(dim=(-2, -1)))

    def grow_width(self, *, site_id: str, operator: str, increment: int,
                   max_width: int) -> "VisualMultistage | None":
        candidate = super().grow_width(
            site_id=site_id, operator=operator,
            increment=increment, max_width=max_width,
        )
        if candidate is not None:
            candidate._gelu_hidden()
            candidate.verify_frozen()
        return candidate

    def grow_depth(self, *, site_id: str, operator: str, max_depth: int,
                   seed_width: int = 1) -> "VisualMultistage | None":
        candidate = super().grow_depth(
            site_id=site_id, operator=operator,
            max_depth=max_depth, seed_width=seed_width,
        )
        if candidate is not None:
            candidate._gelu_hidden()
            candidate.verify_frozen()
        return candidate

    def snapshot(self, *, cursor: int = 0) -> dict[str, object]:
        self.verify_frozen()
        return {
            "kind": self.SNAPSHOT_KIND,
            "spec": asdict(self.spec),
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_widths": list(self.hidden_widths),
            "batch_norm": self.batch_norm,
            "cursor": int(cursor), "generation": int(self._birth_generation),
            "growth_site": self.spec.site_id,
            "frozen_sha256": self._frozen_baseline_sha256,
            "state_dict": copy.deepcopy(self.state_dict()),
        }

    @classmethod
    def from_snapshot(cls, record: dict[str, object]) -> "VisualMultistage":
        if record.get("kind") != cls.SNAPSHOT_KIND:
            raise ValueError("unsupported V08 multistage snapshot kind")
        spec = MultistageSpec(**record["spec"])
        with torch.random.fork_rng(devices=[]):
            model = cls(spec, tuple(int(v) for v in record["hidden_widths"]))
        model.load_state_dict(record["state_dict"], strict=True)
        model._width_cursor = int(record.get("cursor", 0))
        model._birth_generation = int(record.get("generation", 0))
        model._frozen_baseline_sha256 = str(record["frozen_sha256"])
        model.train(False)
        model.verify_frozen()
        if (model.input_dim != int(record["input_dim"]) or
                model.output_dim != int(record["output_dim"]) or
                record.get("growth_site") != model.spec.site_id):
            raise ValueError("V08 checkpoint changed local fixed interface or site")
        return model
