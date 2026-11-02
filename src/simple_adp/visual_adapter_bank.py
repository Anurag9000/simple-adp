"""V09 strictly isolated fixed-router bank of local residual adapters.

This is a small frozen-random CNN control, not a pretrained bank or a learned
router. Every job declares one bank member as the only trainable/expandable
MLP. A fixed, label-free hard router chooses exactly one member per image.
Nonselected adapters, projections, routing weights and backbone are immutable.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .models import TabularMLP
from .visual_inner import VisualResidualAdapter


@dataclass(frozen=True)
class BankSpec:
    input_channels: int
    backbone_channels: int
    local_dim: int
    classes: int
    image_height: int
    image_width: int
    site_id: str
    bank_size: int = 3
    fixed_other_width: int = 5
    fixed_seed: int = 47

    def __post_init__(self) -> None:
        for name in (
            "input_channels", "backbone_channels", "local_dim", "classes",
            "image_height", "image_width", "bank_size", "fixed_other_width",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"V09 {name} must be positive")
        if self.bank_size < 2 or self.backbone_channels < 2:
            raise ValueError("V09 hard router requires at least two members/features")
        if self.site_id not in {f"bank:{i}" for i in range(self.bank_size)}:
            raise ValueError("V09 site_id must select one existing bank:N member")

    @property
    def selected(self) -> int:
        return int(self.site_id.split(":")[1])


class VisualAdapterBank(VisualResidualAdapter):
    """Fixed shared CNN and K adapters, with a deterministic hard gate.

    Routing is independent of labels, targets, gradients and batch order:
    frozen feature means -> frozen K-way linear router -> argmax. No learned
    gating or test-label routing is used. Selected member's gradients flow
    through frozen downstream CNN; all other bank weights stay fixed.
    """

    SNAPSHOT_KIND = "visual-adapter-bank-v09-hard-router-v1"

    def __init__(self, spec: BankSpec, hidden_widths: tuple[int, ...] = (4,)):
        nn.Module.__init__(self)
        if not hidden_widths or min(hidden_widths) <= 0:
            raise ValueError("V09 requires nonempty positive local hidden widths")
        self.spec = spec
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(spec.fixed_seed)
            self.pre = nn.Conv2d(
                spec.input_channels, spec.backbone_channels, 3, padding=1,
            )
            self.project_in = nn.Linear(spec.backbone_channels, spec.local_dim)
            self.project_out = nn.Linear(spec.local_dim, spec.backbone_channels)
            self.router = nn.Linear(spec.backbone_channels, spec.bank_size, bias=False)
            # Fixed, evenly spaced 2D gate directions make each hard-routed
            # member reachable on an appropriate frozen feature vector.
            with torch.no_grad():
                self.router.weight.zero_()
                angles = torch.arange(spec.bank_size, dtype=torch.float32) * (
                    2.0 * torch.pi / spec.bank_size
                )
                self.router.weight[:, 0] = torch.cos(angles)
                self.router.weight[:, 1] = torch.sin(angles)
            self.post = nn.Sequential(
                nn.Conv2d(spec.backbone_channels, spec.backbone_channels, 3, padding=1),
                nn.GELU(),
            )
            self.classifier = nn.Linear(spec.backbone_channels, spec.classes)
            self.other_adapters = nn.ModuleDict({
                f"bank{i}": TabularMLP(
                    spec.local_dim, (spec.fixed_other_width,), spec.local_dim,
                    batch_norm=False,
                )
                for i in range(spec.bank_size) if i != spec.selected
            })
        self.local_mlp = TabularMLP(
            spec.local_dim, hidden_widths, spec.local_dim, batch_norm=False,
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
        for adapter in self.other_adapters.values():
            for i, layer in enumerate(adapter.hidden_layers):
                if isinstance(layer, nn.ReLU):
                    adapter.hidden_layers[i] = nn.GELU()

    def _fixed_modules(self) -> tuple[nn.Module, ...]:
        return (
            self.pre, self.project_in, self.project_out, self.router,
            self.post, self.classifier, self.other_adapters,
        )

    def frozen_state_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(f"V09:router=hard-argmax-cosine-v1:bank={self.spec.bank_size}".encode())
        for prefix, module in zip(
            ("pre", "pin", "pout", "router", "post", "classifier", "other_bank"),
            self._fixed_modules(),
        ):
            for key, tensor in sorted(module.state_dict().items()):
                array = tensor.detach().cpu().contiguous().numpy()
                digest.update(f"{prefix}:{key}:{array.dtype}:{array.shape}".encode())
                digest.update(array.tobytes())
        return digest.hexdigest()

    def _validate_images(self, images: torch.Tensor) -> None:
        if (images.ndim != 4 or images.shape[1] != self.spec.input_channels
                or tuple(images.shape[-2:]) !=
                (self.spec.image_height, self.spec.image_width)):
            raise ValueError("V09 images must match fixed NCHW channels and resolution")

    def _pre_and_route(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_images(images)
        features = self.pre(images)
        scores = self.router(features.mean(dim=(-2, -1)))
        # Torch argmax has deterministic tie-breaking. No label or task ID.
        return features, scores.argmax(dim=-1)

    @torch.no_grad()
    def route_indices(self, images: torch.Tensor) -> torch.Tensor:
        _, chosen = self._pre_and_route(images)
        return chosen

    @torch.no_grad()
    def routing_histogram(self, images: torch.Tensor) -> list[int]:
        chosen = self.route_indices(images)
        return torch.bincount(chosen, minlength=self.spec.bank_size).cpu().tolist()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features, chosen = self._pre_and_route(images)
        z = self.project_in(features.permute(0, 2, 3, 1).contiguous())
        # Each member processes the same fixed interface. A *hard* mask
        # ensures no cross-adapter gradient leakage; the router stays frozen.
        candidates = []
        for i in range(self.spec.bank_size):
            adapter = (self.local_mlp if i == self.spec.selected
                       else self.other_adapters[f"bank{i}"])
            candidates.append(adapter(z))
        bank_outputs = torch.stack(candidates, dim=1)  # [B,K,H,W,D]
        index = chosen.reshape(-1, 1, 1, 1, 1).expand(
            -1, 1, z.shape[1], z.shape[2], z.shape[3],
        )
        selected_delta = bank_outputs.gather(dim=1, index=index).squeeze(1)
        delta = self.project_out(selected_delta).permute(0, 3, 1, 2)
        h = self.post(features + delta)
        return self.classifier(h.mean(dim=(-2, -1)))

    def grow_width(self, *, site_id: str, operator: str, increment: int,
                   max_width: int) -> "VisualAdapterBank | None":
        result = super().grow_width(
            site_id=site_id, operator=operator, increment=increment,
            max_width=max_width,
        )
        if result is not None:
            result._gelu_hidden()
            result.verify_frozen()
        return result

    def grow_depth(self, *, site_id: str, operator: str, max_depth: int,
                   seed_width: int = 1) -> "VisualAdapterBank | None":
        result = super().grow_depth(
            site_id=site_id, operator=operator, max_depth=max_depth,
            seed_width=seed_width,
        )
        if result is not None:
            result._gelu_hidden()
            result.verify_frozen()
        return result

    def snapshot(self, *, cursor: int = 0) -> dict[str, object]:
        self.verify_frozen()
        return {
            "kind": self.SNAPSHOT_KIND,
            "spec": asdict(self.spec),
            "input_dim": self.input_dim, "output_dim": self.output_dim,
            "hidden_widths": list(self.hidden_widths), "batch_norm": self.batch_norm,
            "cursor": int(cursor), "generation": int(self._birth_generation),
            "growth_site": self.spec.site_id,
            "router_kind": "hard-argmax-cosine-v1",
            "frozen_sha256": self._frozen_baseline_sha256,
            "state_dict": copy.deepcopy(self.state_dict()),
        }

    @classmethod
    def from_snapshot(cls, record: dict[str, object]) -> "VisualAdapterBank":
        if record.get("kind") != cls.SNAPSHOT_KIND:
            raise ValueError("unsupported V09 adapter bank snapshot")
        if record.get("router_kind") != "hard-argmax-cosine-v1":
            raise ValueError("V09 checkpoint router semantics changed")
        spec = BankSpec(**record["spec"])
        with torch.random.fork_rng(devices=[]):
            model = cls(spec, tuple(int(w) for w in record["hidden_widths"]))
        model.load_state_dict(record["state_dict"], strict=True)
        model._width_cursor = int(record.get("cursor", 0))
        model._birth_generation = int(record.get("generation", 0))
        model._frozen_baseline_sha256 = str(record["frozen_sha256"])
        model.train(False)
        model.verify_frozen()
        if (model.input_dim != int(record["input_dim"])
                or model.output_dim != int(record["output_dim"])
                or model.spec.site_id != record.get("growth_site")):
            raise ValueError("V09 restored local interface or selected member changed")
        return model
