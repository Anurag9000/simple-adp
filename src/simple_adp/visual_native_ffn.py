"""Experimental V03 ViT FFN, V04 ConvNeXt FFN and V05 PoolFormer FFN.

These are *tiny from-scratch controls*, not torchvision pretrained backbones.
They use genuine frozen attention or depthwise spatial mixing and a native
channels-last / token-wise GELU dense FFN, with the fixed d -> d interface.
The only eligible parameters are the local FFN's TabularMLP Linear weights.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .models import TabularMLP
from .visual_inner import VisualResidualAdapter
from .visual_shift import S2SpatialShift, AxialSpatialShift


@dataclass(frozen=True)
class NativeFFNSpec:
    architecture_id: str
    input_channels: int
    local_dim: int
    classes: int
    image_height: int
    image_width: int
    patch_size: int = 2
    attention_heads: int = 2
    fixed_seed: int = 23
    site_id: str = ""

    def __post_init__(self) -> None:
        if self.architecture_id not in {"V03", "V04", "V05", "V06a", "V06b"}:
            raise ValueError("native FFN architecture must be V03, V04, V05, V06a or V06b")
        for key in ("input_channels", "local_dim", "classes", "image_height",
                    "image_width", "patch_size", "attention_heads"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if (self.image_height % self.patch_size != 0
                or self.image_width % self.patch_size != 0):
            raise ValueError("image resolution must be divisible by the fixed patch size")
        if self.architecture_id == "V03" and self.local_dim % self.attention_heads:
            raise ValueError("ViT embedding dimension must be divisible by attention heads")
        if self.architecture_id == "V06a" and self.local_dim % 4:
            raise ValueError("S2-MLP channel count must be divisible by four")
        if self.architecture_id == "V06b" and self.local_dim % 2:
            raise ValueError("AS-MLP channel count must be even")
        wanted = {"V03": "ffn:block0", "V04": "channel:stage0.block0",
                  "V05": "channel:poolformer.block0",
                  "V06a": "channel:s2.block0",
                  "V06b": "channel:axial.block0"}[self.architecture_id]
        if self.site_id != wanted:
            raise ValueError(f"{self.architecture_id} eligible growth site must be {wanted!r}")

    @property
    def token_count(self) -> int:
        return (self.image_height // self.patch_size) * (self.image_width // self.patch_size)


class VisualNativeFFN(VisualResidualAdapter):
    """Single real internal FFN in a frozen ViT / ConvNeXt / PoolFormer block.

    V03: frozen patch embedding, CLS/pos embeddings, MHA, norms, classifier;
         token-wise GELU FFN on [batch, patches+1, d] with residual.
    V04: frozen strided stem and depthwise 7x7 conv, channel LayerNorm,
         fixed layer scale, classifier; GELU FFN on [batch, h, w, d].
    V05: frozen AvgPool2d - identity mixer, channel FFN GELU and classifier.
    V06a/b: parameter-free S2 / dual-axial zero-padded spatial shifts and FFN.
    All local FFN hidden layers are eligible for native ADP operators.
    """

    SNAPSHOT_KIND = "visual-native-ffn-v1"

    def __init__(self, spec: NativeFFNSpec, hidden_widths: tuple[int, ...] = (8,)):
        nn.Module.__init__(self)
        self.spec = spec
        if not hidden_widths or min(hidden_widths) <= 0:
            raise ValueError("local FFN hidden widths must be positive")
        dim = spec.local_dim
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(spec.fixed_seed)
            self.stem = nn.Conv2d(
                spec.input_channels, dim, kernel_size=spec.patch_size,
                stride=spec.patch_size,
            )
            if spec.architecture_id == "V03":
                self.cls_embedding = nn.Embedding(1, dim)
                self.position_embedding = nn.Embedding(spec.token_count + 1, dim)
                self.pre_attention_norm = nn.LayerNorm(dim)
                self.attention = nn.MultiheadAttention(
                    dim, spec.attention_heads, dropout=0.0, batch_first=True,
                )
                self.pre_ffn_norm = nn.LayerNorm(dim)
                self.output_norm = nn.LayerNorm(dim)
            else:
                if spec.architecture_id == "V04":
                    self.depthwise = nn.Conv2d(
                        dim, dim, kernel_size=7, padding=3, groups=dim,
                    )
                elif spec.architecture_id == "V05":
                    self.pooling_mixer = nn.AvgPool2d(
                        kernel_size=3, stride=1, padding=1, count_include_pad=False,
                    )
                else:
                    self.spatial_shift = (
                        S2SpatialShift(dim) if spec.architecture_id == "V06a"
                        else AxialSpatialShift(dim)
                    )
                self.pre_ffn_norm = nn.LayerNorm(dim)
                # ConvNeXt's fixed per-channel layer scaling is noneligible.
                self.layer_scale = nn.ParameterList([
                    nn.Parameter(torch.full((dim,), 1e-3), requires_grad=False)
                ])
                self.output_norm = nn.LayerNorm(dim)
            self.classifier = nn.Linear(dim, spec.classes)
        self.local_mlp = TabularMLP(dim, hidden_widths, dim, batch_norm=False)
        self._gelu_hidden()
        for module in self._fixed_modules():
            module.requires_grad_(False)
            module.eval()
        self.local_mlp.train(True)
        self._frozen_baseline_sha256 = self.frozen_state_sha256()
        self._birth_generation = 0

    def _gelu_hidden(self) -> None:
        # Source growth constructors rebuild TabularMLP with ReLU modules.
        # Replacing *stateless* activations only preserves every source weight,
        # source overlap-copy operator and optimizer parameter identity.
        for index, layer in enumerate(self.local_mlp.hidden_layers):
            if isinstance(layer, nn.ReLU):
                self.local_mlp.hidden_layers[index] = nn.GELU()

    def _fixed_modules(self) -> tuple[nn.Module, ...]:
        if self.spec.architecture_id == "V03":
            return (
                self.stem, self.cls_embedding, self.position_embedding,
                self.pre_attention_norm, self.attention, self.pre_ffn_norm,
                self.output_norm, self.classifier,
            )
        mixer = (self.depthwise if self.spec.architecture_id == "V04" else
                 self.pooling_mixer if self.spec.architecture_id == "V05" else
                 self.spatial_shift)
        return (
            self.stem, mixer, self.pre_ffn_norm,
            self.layer_scale, self.output_norm, self.classifier,
        )

    def frozen_state_sha256(self) -> str:
        digest = hashlib.sha256()
        names = (
            ("stem", "cls", "position", "attn_norm", "attention", "ffn_norm",
             "out_norm", "classifier")
            if self.spec.architecture_id == "V03"
            else ("stem", "depthwise" if self.spec.architecture_id == "V04" else
                  "pool" if self.spec.architecture_id == "V05" else "shift",
                  "ffn_norm", "layer_scale", "out_norm", "classifier")
        )
        if self.spec.architecture_id == "V05":
            digest.update(repr(self.pooling_mixer).encode())
        if self.spec.architecture_id in {"V06a", "V06b"}:
            digest.update(repr(self.spatial_shift).encode())
        for prefix, module in zip(names, self._fixed_modules()):
            for name, tensor in sorted(module.state_dict().items()):
                arr = tensor.detach().cpu().contiguous().numpy()
                digest.update((prefix + ":" + name + ":" + str(arr.dtype)
                               + ":" + str(arr.shape)).encode())
                digest.update(arr.tobytes())
        return digest.hexdigest()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if (images.ndim != 4 or images.shape[1] != self.spec.input_channels
                or tuple(images.shape[-2:]) != (
                    self.spec.image_height, self.spec.image_width
                )):
            raise ValueError("native FFN images must match NCHW channels/resolution")
        value = self.stem(images)
        if self.spec.architecture_id == "V03":
            tokens = value.flatten(2).transpose(1, 2)
            cls = self.cls_embedding(
                torch.zeros((len(images), 1), dtype=torch.long, device=images.device)
            )
            tokens = torch.cat((cls, tokens), dim=1)
            positions = torch.arange(
                self.spec.token_count + 1, device=images.device,
            )
            tokens = tokens + self.position_embedding(positions).unsqueeze(0)
            norm = self.pre_attention_norm(tokens)
            attended, _ = self.attention(norm, norm, norm, need_weights=False)
            tokens = tokens + attended
            tokens = tokens + self.local_mlp(self.pre_ffn_norm(tokens))
            representation = self.output_norm(tokens[:, 0])
        else:
            # The fixed depthwise mixer remains in the differentiable graph.
            if self.spec.architecture_id == "V04":
                mixed = self.depthwise(value)
                skip = value
            else:
                # PoolFormer: Pool(x)-x residual. Spatial shift families:
                # deterministic zero-padded rearrangement, not convolutions.
                result = (self.pooling_mixer(value) if self.spec.architecture_id == "V05"
                          else self.spatial_shift(value))
                skip = value + (result - value)
                mixed = skip
            mixed = mixed.permute(0, 2, 3, 1).contiguous()
            delta = self.local_mlp(self.pre_ffn_norm(mixed))
            delta = delta * self.layer_scale[0]
            value = skip + delta.permute(0, 3, 1, 2).contiguous()
            representation = self.output_norm(value.mean(dim=(-2, -1)))
        return self.classifier(representation)

    def grow_width(self, *, site_id: str, operator: str,
                   increment: int, max_width: int) -> "VisualNativeFFN | None":
        result = super().grow_width(site_id=site_id, operator=operator,
                                    increment=increment, max_width=max_width)
        if result is not None:
            result._gelu_hidden()
            result.verify_frozen()
        return result

    def grow_depth(self, *, site_id: str, operator: str, max_depth: int,
                   seed_width: int = 1) -> "VisualNativeFFN | None":
        result = super().grow_depth(site_id=site_id, operator=operator,
                                    max_depth=max_depth, seed_width=seed_width)
        if result is not None:
            result._gelu_hidden()
            result.verify_frozen()
        return result

    def snapshot(self, *, cursor: int = 0) -> dict[str, object]:
        self.verify_frozen()
        return {
            "kind": self.SNAPSHOT_KIND,
            "spec": asdict(self.spec),
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_widths": list(self.hidden_widths),
            "batch_norm": self.batch_norm,
            "growth_site": self.spec.site_id,
            "cursor": int(cursor),
            "generation": int(self._birth_generation),
            "frozen_sha256": self._frozen_baseline_sha256,
            "state_dict": copy.deepcopy(self.state_dict()),
        }

    @classmethod
    def from_snapshot(cls, payload: dict[str, object]) -> "VisualNativeFFN":
        if payload.get("kind") != cls.SNAPSHOT_KIND:
            raise ValueError("unsupported native FFN snapshot kind")
        spec = NativeFFNSpec(**payload["spec"])
        with torch.random.fork_rng(devices=[]):
            model = cls(spec, tuple(int(v) for v in payload["hidden_widths"]))
        model.load_state_dict(payload["state_dict"], strict=True)
        model._width_cursor = int(payload.get("cursor", 0))
        model._birth_generation = int(payload.get("generation", 0))
        model._frozen_baseline_sha256 = str(payload["frozen_sha256"])
        model.train(False)
        model.verify_frozen()
        if (model.input_dim != int(payload["input_dim"])
                or model.output_dim != int(payload["output_dim"])
                or payload.get("growth_site") != model.spec.site_id):
            raise ValueError("native FFN local interface/site changed during restore")
        return model


def build_visual_composite(spec, hidden_widths):
    """Explicit factory: never route V03/V04 through a V02 CNN snapshot."""
    if isinstance(spec, NativeFFNSpec):
        return VisualNativeFFN(spec, tuple(hidden_widths))
    from .visual_inner import ResidualAdapterSpec
    if isinstance(spec, ResidualAdapterSpec):
        return VisualResidualAdapter(spec, tuple(hidden_widths))
    raise TypeError("unknown visual composite model spec")


def restore_visual_composite(snapshot: dict[str, object]):
    if snapshot.get("kind") == VisualNativeFFN.SNAPSHOT_KIND:
        return VisualNativeFFN.from_snapshot(snapshot)
    if snapshot.get("kind") == "visual-residual-adapter-v02":
        return VisualResidualAdapter.from_snapshot(snapshot)
    raise ValueError("exact-resume failure: unknown visual composite kind")
