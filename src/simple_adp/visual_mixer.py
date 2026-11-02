"""V07 MLP-Mixer: independently eligible token- and channel-mixing MLP sites.

These are tiny random-initialized controlled architectures, NOT pretrained
MLP-Mixer checkpoints. Token-only and channel-only freeze the opposite MLP.
A separately named "both" ablation grows both internal MLPs synchronously,
without redefining canonical single-site ADP methods. V08 backbone multi-stage
site scheduling remains separate and is not claimed here.
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
class MixerSpec:
    input_channels: int
    local_dim: int
    classes: int
    image_height: int
    image_width: int
    patch_size: int
    fixed_seed: int = 37
    site_id: str = "token:block0"
    fixed_other_width: int = 6

    def __post_init__(self) -> None:
        if self.site_id not in {"token:block0", "channel:block0", "both:block0"}:
            raise ValueError("V07 supports token:block0, channel:block0 or both:block0")
        for name in ("input_channels", "local_dim", "classes", "image_height",
                     "image_width", "patch_size", "fixed_other_width"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.image_height % self.patch_size or self.image_width % self.patch_size:
            raise ValueError("V07 patch size must divide both image dimensions")

    @property
    def tokens(self) -> int:
        return self.image_height // self.patch_size * (self.image_width // self.patch_size)

    @property
    def target_dim(self) -> int:
        return self.tokens if self.site_id in {"token:block0", "both:block0"} else self.local_dim


def _gelu_mlp(module: TabularMLP) -> None:
    for i, layer in enumerate(module.hidden_layers):
        if isinstance(layer, nn.ReLU):
            module.hidden_layers[i] = nn.GELU()


class VisualMixer(VisualResidualAdapter):
    """One-block residual token MLP followed by residual channel MLP.

    Token site acts on the fixed T-axis of [B,D,T], channel site on the fixed
    D-axis of [B,T,D]. The both-site condition synchronously grows the two
    independent MLP hidden architectures without changing either interface.
    """

    SNAPSHOT_KIND = "visual-mixer-v07-single-site-v1"
    BOTH_SNAPSHOT_KIND = "visual-mixer-v07-sync-both-v1"

    def __init__(self, spec: MixerSpec, hidden_widths: tuple[int, ...] = (4,)):
        nn.Module.__init__(self)
        if not hidden_widths or min(hidden_widths) <= 0:
            raise ValueError("V07 requires positive nonempty hidden widths")
        self.spec = spec
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(spec.fixed_seed)
            self.stem = nn.Conv2d(
                spec.input_channels, spec.local_dim,
                kernel_size=spec.patch_size, stride=spec.patch_size,
            )
            self.token_norm = nn.LayerNorm(spec.local_dim)
            self.channel_norm = nn.LayerNorm(spec.local_dim)
            self.output_norm = nn.LayerNorm(spec.local_dim)
            self.classifier = nn.Linear(spec.local_dim, spec.classes)
            other_dim = (spec.local_dim if spec.site_id in {"token:block0", "both:block0"}
                         else spec.tokens)
            other_widths = (hidden_widths if spec.site_id == "both:block0"
                            else (spec.fixed_other_width,))
            self.other_mlp = TabularMLP(
                other_dim, other_widths, other_dim, batch_norm=False,
            )
        # The eligible site alone is initialized by the ordinary controller RNG.
        self.local_mlp = TabularMLP(
            spec.target_dim, hidden_widths, spec.target_dim, batch_norm=False,
        )
        _gelu_mlp(self.local_mlp)
        _gelu_mlp(self.other_mlp)
        for module in self._fixed_modules():
            module.requires_grad_(False)
            module.eval()
        self._birth_generation = 0
        self._frozen_baseline_sha256 = self.frozen_state_sha256()

    def _gelu_hidden(self) -> None:
        _gelu_mlp(self.local_mlp)
        if self.spec.site_id == "both:block0":
            _gelu_mlp(self.other_mlp)

    def _fixed_modules(self) -> tuple[nn.Module, ...]:
        common = (self.stem, self.token_norm, self.channel_norm)
        if self.spec.site_id != "both:block0":
            return (*common, self.other_mlp, self.output_norm, self.classifier)
        return (*common, self.output_norm, self.classifier)

    def frozen_state_sha256(self) -> str:
        digest = hashlib.sha256()
        names = ("stem", "token_norm", "channel_norm",
                 "output_norm", "classifier") if self.spec.site_id == "both:block0" else (
                 "stem", "token_norm", "channel_norm", "other_mlp",
                 "output_norm", "classifier")
        for name, module in zip(names, self._fixed_modules()):
            for key, value in sorted(module.state_dict().items()):
                arr = value.detach().cpu().contiguous().numpy()
                digest.update(f"{name}:{key}:{arr.dtype}:{arr.shape}".encode())
                digest.update(arr.tobytes())
        return digest.hexdigest()

    @property
    def growth_targets(self) -> dict[str, TabularMLP]:
        if self.spec.site_id == "both:block0":
            return {"token:block0": self.local_mlp,
                    "channel:block0": self.other_mlp}
        return {self.spec.site_id: self.local_mlp}

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if (images.ndim != 4 or images.shape[1] != self.spec.input_channels
                or tuple(images.shape[-2:]) !=
                (self.spec.image_height, self.spec.image_width)):
            raise ValueError("V07 images must match fixed NCHW channels and resolution")
        x = self.stem(images).flatten(2).transpose(1, 2).contiguous()  # B,T,D
        if x.shape[1] != self.spec.tokens or x.shape[2] != self.spec.local_dim:
            raise AssertionError("V07 patch/token/channel interface changed")
        # Token mixing: fixed channel identity, MLP operates over tokens T.
        t = self.token_norm(x).transpose(1, 2).contiguous()  # B,D,T
        t = self.local_mlp(t) if self.spec.site_id in {"token:block0", "both:block0"} else self.other_mlp(t)
        x = x + t.transpose(1, 2)
        # Channel mixing: fixed token identity, MLP operates over channel D.
        c = self.channel_norm(x)
        c = self.local_mlp(c) if self.spec.site_id == "channel:block0" else self.other_mlp(c)
        x = x + c
        pooled = self.output_norm(x).mean(dim=1)
        return self.classifier(pooled)

    def grow_width(self, *, site_id: str, operator: str,
                   increment: int, max_width: int) -> "VisualMixer | None":
        if self.spec.site_id != "both:block0":
            result = super().grow_width(
                site_id=site_id, operator=operator,
                increment=increment, max_width=max_width,
            )
        else:
            if site_id != "both:block0":
                raise ValueError("V07 both scope requires explicit both:block0 growth policy")
            from .growth import (
                expand_width_all_layers, expand_width_round_robin,
                expand_width_den_round_robin,
            )
            operators = {
                "all_layers": expand_width_all_layers,
                "round_robin": expand_width_round_robin,
                "den_round_robin": expand_width_den_round_robin,
            }
            if operator not in operators:
                raise ValueError("unknown V07 both-site width operator")
            grow = operators[operator]
            local = grow(self.local_mlp, increment, max_width)
            other = grow(self.other_mlp, increment, max_width)
            if local is None or other is None:
                return None
            result = copy.deepcopy(self)
            result.local_mlp = local
            result.other_mlp = other
            result._birth_generation = self._birth_generation + 1
            result.train(self.training)
        if result is not None:
            result._gelu_hidden()
            result.verify_frozen()
        return result

    def grow_depth(self, *, site_id: str, operator: str, max_depth: int,
                   seed_width: int = 1) -> "VisualMixer | None":
        if self.spec.site_id != "both:block0":
            result = super().grow_depth(
                site_id=site_id, operator=operator, max_depth=max_depth,
                seed_width=seed_width,
            )
        else:
            if site_id != "both:block0":
                raise ValueError("V07 both scope requires explicit both:block0 growth policy")
            from .growth import expand_depth, expand_depth_match_width
            if operator == "D1":
                grow = lambda mlp: expand_depth(
                    mlp, max_depth, min_new_layer_width=seed_width)
            elif operator == "Dcur":
                grow = lambda mlp: expand_depth_match_width(mlp, max_depth)
            else:
                raise ValueError("unknown V07 both-site depth operator")
            local, other = grow(self.local_mlp), grow(self.other_mlp)
            if local is None or other is None:
                return None
            result = copy.deepcopy(self)
            result.local_mlp, result.other_mlp = local, other
            result._birth_generation = self._birth_generation + 1
            result.train(self.training)
        if result is not None:
            result._gelu_hidden()
            result.verify_frozen()
        return result

    def snapshot(self, *, cursor: int = 0) -> dict[str, object]:
        self.verify_frozen()
        return {
            "kind": (self.BOTH_SNAPSHOT_KIND if self.spec.site_id == "both:block0"
                     else self.SNAPSHOT_KIND),
            "spec": asdict(self.spec),
            "input_dim": self.input_dim, "output_dim": self.output_dim,
            "hidden_widths": list(self.hidden_widths), "batch_norm": self.batch_norm,
            "cursor": int(cursor),
            "other_cursor": int(getattr(self.other_mlp, "_width_cursor", 0)),
            "other_widths": list(self.other_mlp.hidden_widths),
            "generation": int(self._birth_generation),
            "growth_site": self.spec.site_id,
            "frozen_sha256": self._frozen_baseline_sha256,
            "state_dict": copy.deepcopy(self.state_dict()),
        }

    @classmethod
    def from_snapshot(cls, payload: dict[str, object]) -> "VisualMixer":
        if payload.get("kind") not in {cls.SNAPSHOT_KIND, cls.BOTH_SNAPSHOT_KIND}:
            raise ValueError("unsupported V07 MLP-Mixer snapshot kind")
        spec = MixerSpec(**payload["spec"])
        with torch.random.fork_rng(devices=[]):
            model = cls(spec, tuple(int(w) for w in payload["hidden_widths"]))
        if model.spec.site_id == "both:block0" and payload.get("kind") != cls.BOTH_SNAPSHOT_KIND:
            raise ValueError("V07 both-scope checkpoint kind must be explicit")
        if model.spec.site_id != "both:block0" and payload.get("kind") != cls.SNAPSHOT_KIND:
            raise ValueError("V07 single-scope checkpoint kind must be explicit")
        if (model.spec.site_id == "both:block0" or "other_widths" in payload) and (
                list(model.other_mlp.hidden_widths) != list(payload.get("other_widths", []))):
            raise ValueError("V07 opposite-site architecture differs from checkpoint")
        model.load_state_dict(payload["state_dict"], strict=True)
        model._width_cursor = int(payload.get("cursor", 0))
        setattr(model.other_mlp, "_width_cursor", int(payload.get("other_cursor", 0)))
        model._birth_generation = int(payload.get("generation", 0))
        model._frozen_baseline_sha256 = str(payload["frozen_sha256"])
        model.train(False)
        model.verify_frozen()
        if (model.input_dim != int(payload["input_dim"])
                or model.output_dim != int(payload["output_dim"])
                or payload["growth_site"] != model.spec.site_id):
            raise ValueError("V07 snapshot changed the token/channel local interface")
        return model
