"""V08 explicitly predeclared multi-site round-robin adapter scheduling.

A frozen three-stage CNN contains three real independently stateful residual
MLPs. One selected site is trainable for each candidate's optimization stage;
the *site* cycles early -> middle -> late by completed growth proposal count.
Within that site, original source ADP RR/all/D1/Dcur policies are unchanged.
This is a noncanonical schedule ablation, not a learned site selector.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .growth import (
    expand_width_round_robin, expand_width_all_layers,
    expand_width_den_round_robin, expand_depth, expand_depth_match_width,
    total_neurons,
)
from .models import TabularMLP
from .visual_inner import VisualResidualAdapter
from .visual_multistage import SITES


@dataclass(frozen=True)
class ScheduledMultistageSpec:
    input_channels: int
    backbone_channels: int
    classes: int
    image_height: int
    image_width: int
    site_id: str = "adapter:round_robin_sites"
    site_order: tuple[str, ...] = SITES
    fixed_seed: int = 43

    def __post_init__(self) -> None:
        if self.site_id != "adapter:round_robin_sites":
            raise ValueError("V08 scheduled site must be adapter:round_robin_sites")
        if tuple(self.site_order) != SITES:
            raise ValueError("V08 scheduled site order must be early/middle/late")
        for name in ("input_channels", "backbone_channels", "classes",
                     "image_height", "image_width"):
            if getattr(self, name) <= 0:
                raise ValueError(f"V08 {name} must be positive")


class VisualMultistageScheduled(VisualResidualAdapter):
    """A fixed predetermined site sequence with checkpointed site MLP states.

    No candidate can change convolutional width, stage count, channel interfaces,
    classifier or the order of sites. Only the currently selected local MLP is
    modified and trained. Earlier grown MLPs remain committed for later passes.
    """

    SNAPSHOT_KIND = "visual-multistage-v08-rr-sites-v1"

    def __init__(
        self, spec: ScheduledMultistageSpec,
        hidden_widths: tuple[int, ...] = (4,),
        *, site_widths: dict[str, tuple[int, ...]] | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.spec = spec
        if not hidden_widths or min(hidden_widths) <= 0:
            raise ValueError("V08 scheduled local widths must be positive")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(spec.fixed_seed)
            self.stem = nn.Conv2d(spec.input_channels, spec.backbone_channels, 3, padding=1)
            self.stage_convs = nn.ModuleList([
                nn.Conv2d(spec.backbone_channels, spec.backbone_channels, 3, padding=1)
                for _ in SITES
            ])
            self.classifier = nn.Linear(spec.backbone_channels, spec.classes)
        widths = site_widths or {site: hidden_widths for site in SITES}
        if set(widths) != set(SITES):
            raise ValueError("V08 scheduled checkpoint must specify all three sites")
        self.site_mlps = nn.ModuleDict()
        for site in SITES:
            layer_widths = tuple(int(w) for w in widths[site])
            if not layer_widths or min(layer_widths) <= 0:
                raise ValueError("V08 scheduled site widths must be positive")
            self.site_mlps[site] = TabularMLP(
                spec.backbone_channels, layer_widths, spec.backbone_channels,
                batch_norm=False,
            )
            self._gelu_site(site)
        self._active = "early"
        self._birth_generation = 0
        for fixed in self._fixed_modules():
            fixed.requires_grad_(False)
            fixed.eval()
        self._frozen_baseline_sha256 = self.frozen_state_sha256()
        self.activate("early")

    @property
    def local_mlp(self) -> TabularMLP:
        return self.site_mlps[self._active]

    @local_mlp.setter
    def local_mlp(self, model: TabularMLP) -> None:
        self.site_mlps[self._active] = model

    @property
    def growth_targets(self) -> dict[str, TabularMLP]:
        return {f"adapter:{site}": self.site_mlps[site] for site in SITES}

    def _gelu_site(self, site: str) -> None:
        mlp = self.site_mlps[site]
        for i, layer in enumerate(mlp.hidden_layers):
            if isinstance(layer, nn.ReLU):
                mlp.hidden_layers[i] = nn.GELU()

    def _gelu_hidden(self) -> None:
        self._gelu_site(self._active)

    def _fixed_modules(self) -> tuple[nn.Module, ...]:
        return (self.stem, self.stage_convs, self.classifier)

    def frozen_state_sha256(self) -> str:
        digest = hashlib.sha256(b"V08:fixed3stage:predeclared-rr-sites-v1")
        for prefix, module in zip(("stem", "convs", "classifier"), self._fixed_modules()):
            for key, tensor in sorted(module.state_dict().items()):
                arr = tensor.detach().cpu().contiguous().numpy()
                digest.update(f"{prefix}:{key}:{arr.dtype}:{arr.shape}".encode())
                digest.update(arr.tobytes())
        return digest.hexdigest()

    def activate(self, site: str) -> "VisualMultistageScheduled":
        if site not in SITES:
            raise ValueError("V08 cannot activate an undeclared stage")
        self._active = site
        self.train(self.training)
        self.verify_frozen()
        return self

    def train(self, mode: bool = True) -> "VisualMultistageScheduled":
        nn.Module.train(self, mode)
        for module in self._fixed_modules():
            module.eval()
        for site, module in self.site_mlps.items():
            module.requires_grad_(site == self._active)
            module.train(mode and site == self._active)
        return self

    def verify_frozen(self) -> None:
        super().verify_frozen()
        for site, module in self.site_mlps.items():
            eligible = site == self._active
            if any(p.requires_grad != eligible for p in module.parameters()):
                raise ValueError("V08 scheduled frozen/unselected site became trainable")
            if not eligible and module.training:
                raise ValueError("V08 scheduled unselected site must remain eval")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if (images.ndim != 4 or images.shape[1] != self.spec.input_channels
                or tuple(images.shape[-2:]) !=
                (self.spec.image_height, self.spec.image_width)):
            raise ValueError("V08 scheduled images must match fixed NCHW channels/resolution")
        h = self.stem(images)
        for i, site in enumerate(SITES):
            h = torch.nn.functional.gelu(self.stage_convs[i](h))
            z = h.permute(0, 2, 3, 1).contiguous()
            delta = self.site_mlps[site](z).permute(0, 3, 1, 2).contiguous()
            h = h + 0.1 * delta
        return self.classifier(h.mean(dim=(-2, -1)))

    def grow_width(self, *, site_id: str, operator: str, increment: int,
                   max_width: int) -> "VisualMultistageScheduled | None":
        if site_id != self.spec.site_id:
            raise ValueError("V08 scheduled growth requires adapter:round_robin_sites")
        choices = {
            "round_robin": expand_width_round_robin,
            "all_layers": expand_width_all_layers,
            "den_round_robin": expand_width_den_round_robin,
        }
        if operator not in choices:
            raise ValueError("invalid V08 scheduled source width operator")
        selected = choices[operator](self.local_mlp, increment, max_width)
        if selected is None:
            return None
        candidate = copy.deepcopy(self)
        # Explicit ModuleDict replacement: nn.Module.__setattr__ does not
        # dispatch Python property setters for child nn.Module instances.
        candidate.site_mlps[candidate._active] = selected
        candidate._gelu_hidden()
        candidate._birth_generation += 1
        candidate.train(self.training)
        candidate.verify_frozen()
        return candidate

    def grow_depth(self, *, site_id: str, operator: str,
                   max_depth: int, seed_width: int = 1) -> "VisualMultistageScheduled | None":
        if site_id != self.spec.site_id:
            raise ValueError("V08 scheduled growth requires adapter:round_robin_sites")
        if operator == "D1":
            mlp = expand_depth(self.local_mlp, max_depth,
                               min_new_layer_width=seed_width)
        elif operator == "Dcur":
            mlp = expand_depth_match_width(self.local_mlp, max_depth)
        else:
            raise ValueError("invalid V08 scheduled source depth operator")
        if mlp is None:
            return None
        candidate = copy.deepcopy(self)
        candidate.site_mlps[candidate._active] = mlp
        candidate._gelu_hidden()
        candidate._birth_generation += 1
        candidate.train(self.training)
        candidate.verify_frozen()
        return candidate

    def total_local_neurons(self) -> int:
        return sum(total_neurons(self.site_mlps[site]) for site in SITES)

    def snapshot(self, *, cursor: int = 0) -> dict[str, object]:
        self.verify_frozen()
        return {
            "kind": self.SNAPSHOT_KIND, "spec": asdict(self.spec),
            "input_dim": self.input_dim, "output_dim": self.output_dim,
            "hidden_widths": list(self.hidden_widths),
            "site_widths": {
                site: list(self.site_mlps[site].hidden_widths) for site in SITES
            },
            "site_cursors": {
                site: int(getattr(self.site_mlps[site], "_width_cursor", 0))
                for site in SITES
            },
            "active_site": self._active,
            "batch_norm": self.batch_norm,
            "cursor": int(cursor), "generation": int(self._birth_generation),
            "growth_site": self.spec.site_id,
            "frozen_sha256": self._frozen_baseline_sha256,
            "state_dict": copy.deepcopy(self.state_dict()),
        }

    @classmethod
    def from_snapshot(cls, record: dict[str, object]) -> "VisualMultistageScheduled":
        if record.get("kind") != cls.SNAPSHOT_KIND:
            raise ValueError("unsupported V08 scheduled snapshot kind")
        spec = ScheduledMultistageSpec(**record["spec"])
        widths = record["site_widths"]
        if set(widths) != set(SITES):
            raise ValueError("V08 scheduled snapshot omitted declared sites")
        active = record["active_site"]
        if active not in SITES or list(record["hidden_widths"]) != list(widths[active]):
            raise ValueError("V08 scheduled snapshot active site/widths mismatch")
        with torch.random.fork_rng(devices=[]):
            model = cls(spec, tuple(widths[active]), site_widths={
                site: tuple(int(x) for x in widths[site]) for site in SITES
            })
        model._active = active
        model.load_state_dict(record["state_dict"], strict=True)
        for site in SITES:
            model.site_mlps[site]._width_cursor = int(record["site_cursors"][site])
        model._birth_generation = int(record.get("generation", 0))
        model._frozen_baseline_sha256 = str(record["frozen_sha256"])
        model.train(False)
        model.verify_frozen()
        if (model.input_dim != int(record["input_dim"])
                or model.output_dim != int(record["output_dim"])
                or model.spec.site_id != record["growth_site"]):
            raise ValueError("V08 scheduled snapshot changed fixed interfaces")
        return model
