"""Source-faithful V03 torchvision ViT-B/16 first transformer MLP adaptation.

Only the actual encoder_layer_0.mlp channel FFN is eligible for ADP growth.
Full local weights are strict-loaded and SHA256-verified; all transformer
attention/other blocks remain frozen. No download or pretrained-origin claim
is inferred from a local state_dict. ViT positional embeddings require 224².
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from .growth import (
    expand_depth, expand_depth_match_width, expand_width_all_layers,
    expand_width_den_round_robin, expand_width_round_robin,
)
from .models import TabularMLP
from .visual_inner import VisualResidualAdapter
from .visual_source_convnext import file_sha256

_SOURCE_DIM = 768
_SOURCE_HIDDEN = 3072
_SOURCE_SITE_PREFIX = "encoder.layers.encoder_layer_0.mlp.0."
_SITE = "vit:encoder_layer_0.channel_ffn"


@dataclass(frozen=True)
class SourceViTSpec:
    classes: int
    image_height: int
    image_width: int
    weights_path: str
    weights_sha256: str
    input_channels: int = 3
    fixed_seed: int = 0
    site_id: str = _SITE
    normalization: str = "imagenet"

    def __post_init__(self) -> None:
        if self.classes < 2 or min(self.image_height, self.image_width) < 4:
            raise ValueError("source ViT requires >=2 classes and positive source image geometry")
        if self.input_channels not in {1, 3}:
            raise ValueError("source ViT requires grayscale or RGB images")
        if self.site_id != _SITE:
            raise ValueError("source ViT has one eligible first-transformer FFN site")
        if self.normalization not in {"imagenet", "none"}:
            raise ValueError("unsupported source ViT normalization")
        if not self.weights_path or len(self.weights_sha256) != 64 or any(
            c not in "0123456789abcdefABCDEF" for c in self.weights_sha256
        ):
            raise ValueError("complete local ViT source checkpoint path and SHA256 required")


class VisualSourceViTFFN(VisualResidualAdapter):
    """Frozen full ViT with trainable original FFN and new fixed-size task head.

    Source encoder_layer_0.mlp is:
        Linear(768, 3072) -> GELU -> Dropout -> Linear(3072, 768) -> Dropout.
    The two original source Linear tensors transfer exactly to a compatible
    GELU TabularMLP. Original pretrained Dropout is inactive in eval-mode
    source; keeping the entire source in eval makes initial features exactly
    equal. Candidate widening/deepening mutates ONLY the inner source MLP.
    """

    SNAPSHOT_KIND = "visual-v03-source-vit-b16-ffn-v1"
    SOURCE_WIDTH = _SOURCE_HIDDEN

    def __init__(self, spec: SourceViTSpec,
                 hidden_widths: tuple[int, ...] = (_SOURCE_HIDDEN,)) -> None:
        nn.Module.__init__(self)
        if not hidden_widths or min(hidden_widths) <= 0:
            raise ValueError("source ViT hidden widths must be positive")
        self.spec = spec
        checkpoint = Path(spec.weights_path)
        if not checkpoint.is_file():
            raise ValueError("source ViT checkpoint file is missing")
        if file_sha256(checkpoint).lower() != spec.weights_sha256.lower():
            raise ValueError("source ViT checkpoint SHA256 mismatch")
        try:
            from torchvision import models
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError("source ViT requires torchvision") from exc

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(spec.fixed_seed))
            source = models.vit_b_16(weights=None).eval()
        payload = torch.load(checkpoint, weights_only=True, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError("source ViT checkpoint must be a named tensor state_dict")
        if "state_dict" in payload and isinstance(payload["state_dict"], dict):
            payload = payload["state_dict"]
        if not payload or not all(isinstance(k, str) and torch.is_tensor(v)
                                  for k, v in payload.items()):
            raise ValueError("source ViT checkpoint must be a named tensor state_dict")
        source.load_state_dict(payload, strict=True)
        del payload

        block = source.encoder.layers[0]
        old = block.mlp
        if (not isinstance(old, nn.Sequential) or len(old) != 5
                or not isinstance(old[0], nn.Linear)
                or not isinstance(old[1], nn.GELU)
                or not isinstance(old[2], nn.Dropout)
                or not isinstance(old[3], nn.Linear)
                or not isinstance(old[4], nn.Dropout)
                or old[0].in_features != _SOURCE_DIM
                or old[0].out_features != _SOURCE_HIDDEN
                or old[3].in_features != _SOURCE_HIDDEN
                or old[3].out_features != _SOURCE_DIM
                or source.image_size != 224):
            raise ValueError("loaded torchvision ViT FFN/positional geometry differs from V03")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(spec.fixed_seed) + 101)
            source.heads.head = nn.Linear(
                source.heads.head.in_features, spec.classes,
            )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(spec.fixed_seed) + 102)
            local = TabularMLP(_SOURCE_DIM, hidden_widths, _SOURCE_DIM)
        with torch.no_grad():
            first = next(layer for layer in local.hidden_layers
                         if isinstance(layer, nn.Linear))
            rows = min(first.weight.shape[0], old[0].weight.shape[0])
            first.weight[:rows].copy_(old[0].weight[:rows])
            first.bias[:rows].copy_(old[0].bias[:rows])
            columns = min(local.output_layer.weight.shape[1],
                          old[3].weight.shape[1])
            local.output_layer.weight[:, :columns].copy_(
                old[3].weight[:, :columns])
            local.output_layer.bias.copy_(old[3].bias)
        for idx, layer in enumerate(local.hidden_layers):
            if isinstance(layer, nn.ReLU):
                local.hidden_layers[idx] = nn.GELU()

        # Preserve the original encoder block and all residual/attention
        # operations. The MLPBlock retains its original Sequential shape,
        # but slots 1-4 become stateless identities.
        old[0] = local
        for idx in range(1, 5):
            old[idx] = nn.Identity()
        self.source = source
        self.register_buffer(
            "input_mean",
            torch.tensor([0.485, 0.456, 0.406] if spec.normalization == "imagenet"
                         else [0., 0., 0.]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "input_std",
            torch.tensor([0.229, 0.224, 0.225] if spec.normalization == "imagenet"
                         else [1., 1., 1.]).view(1, 3, 1, 1),
        )
        self._birth_generation = 0
        self._source_equivalent_at_birth = tuple(hidden_widths) == (_SOURCE_HIDDEN,)
        self.source.requires_grad_(False)
        self.local_mlp.requires_grad_(True)
        self.task_head.requires_grad_(True)
        self._frozen_baseline_sha256 = self.frozen_state_sha256()
        self.train(False)

    @property
    def local_mlp(self) -> TabularMLP:
        return self.source.encoder.layers[0].mlp[0]

    @property
    def task_head(self) -> nn.Linear:
        return self.source.heads.head

    @property
    def source_ffn_exact_at_initialization(self) -> bool:
        return bool(self._source_equivalent_at_birth)

    def _fixed_modules(self) -> tuple[nn.Module, ...]:
        return ()

    def frozen_state_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(f"V03-source:{self.spec.site_id}:{self.spec.weights_sha256}".encode())
        for name, tensor in sorted(self.source.state_dict().items()):
            if name.startswith((_SOURCE_SITE_PREFIX, "heads.head.")):
                continue
            array = tensor.detach().cpu().contiguous().numpy()
            digest.update(f"{name}:{array.dtype}:{array.shape}".encode())
            digest.update(array.tobytes())
        for name, tensor in (("input_mean", self.input_mean),
                             ("input_std", self.input_std)):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def verify_frozen(self) -> None:
        if self.frozen_state_sha256() != self._frozen_baseline_sha256:
            raise ValueError("source ViT frozen parameters or buffers changed")
        for name, param in self.source.named_parameters():
            mutable = (name.startswith(_SOURCE_SITE_PREFIX)
                       or name.startswith("heads.head."))
            if bool(param.requires_grad) != mutable:
                raise ValueError("source ViT trainable scope was violated")
        trainable_modules = {
            id(child)
            for mutable in (self.local_mlp, self.task_head)
            for child in mutable.modules()
        }
        if self.source.training or any(
            child.training for child in self.source.modules()
            if id(child) not in trainable_modules
        ):
            raise ValueError("source ViT frozen blocks must stay in eval mode")

    def train(self, mode: bool = True) -> "VisualSourceViTFFN":
        nn.Module.train(self, False)
        self.local_mlp.train(mode)
        self.task_head.train(mode)
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if (images.ndim != 4 or images.shape[1] != self.spec.input_channels
                or tuple(images.shape[-2:]) !=
                (self.spec.image_height, self.spec.image_width)):
            raise ValueError("source ViT input must match fixed NCHW geometry")
        if not images.is_floating_point() or not torch.isfinite(images).all():
            raise ValueError("source ViT images must be finite float pixels")
        rgb = (images.repeat(1, 3, 1, 1)
               if self.spec.input_channels == 1 else images)
        if tuple(rgb.shape[-2:]) != (224, 224):
            rgb = nn.functional.interpolate(
                rgb, size=(224, 224), mode="bilinear", align_corners=False)
        return self.source((rgb - self.input_mean) / self.input_std)

    def _gelu_hidden(self) -> None:
        for idx, layer in enumerate(self.local_mlp.hidden_layers):
            if isinstance(layer, nn.ReLU):
                self.local_mlp.hidden_layers[idx] = nn.GELU()

    def grow_width(self, *, site_id: str, operator: str,
                   increment: int, max_width: int) -> "VisualSourceViTFFN | None":
        if site_id != self.spec.site_id:
            raise ValueError("noneligible source ViT FFN growth site")
        operators = {
            "round_robin": expand_width_round_robin,
            "den_round_robin": expand_width_den_round_robin,
            "all_layers": expand_width_all_layers,
        }
        if operator not in operators:
            raise ValueError("unknown source ViT width growth operator")
        new_ffn = operators[operator](self.local_mlp, increment, max_width)
        if new_ffn is None:
            return None
        candidate = copy.deepcopy(self)
        candidate.source.encoder.layers[0].mlp[0] = new_ffn
        candidate._gelu_hidden()
        candidate._birth_generation += 1
        candidate.train(self.training)
        candidate.verify_frozen()
        return candidate

    def grow_depth(self, *, site_id: str, operator: str, max_depth: int,
                   seed_width: int = 1) -> "VisualSourceViTFFN | None":
        if site_id != self.spec.site_id:
            raise ValueError("noneligible source ViT FFN growth site")
        if operator == "D1":
            new_ffn = expand_depth(
                self.local_mlp, max_depth, min_new_layer_width=seed_width)
        elif operator == "Dcur":
            new_ffn = expand_depth_match_width(self.local_mlp, max_depth)
        else:
            raise ValueError("unknown source ViT depth growth operator")
        if new_ffn is None:
            return None
        candidate = copy.deepcopy(self)
        candidate.source.encoder.layers[0].mlp[0] = new_ffn
        candidate._gelu_hidden()
        candidate._birth_generation += 1
        candidate.train(self.training)
        candidate.verify_frozen()
        return candidate

    def snapshot(self, *, cursor: int = 0) -> dict[str, object]:
        self.verify_frozen()
        return {
            "kind": self.SNAPSHOT_KIND, "spec": asdict(self.spec),
            "input_dim": self.input_dim, "output_dim": self.output_dim,
            "hidden_widths": list(self.hidden_widths),
            "batch_norm": False, "growth_site": self.spec.site_id,
            "cursor": int(cursor), "generation": self._birth_generation,
            "source_ffn_exact": self.source_ffn_exact_at_initialization,
            "frozen_sha256": self._frozen_baseline_sha256,
            "state_dict": copy.deepcopy(self.state_dict()),
        }

    @classmethod
    def from_snapshot(cls, record: dict[str, object]) -> "VisualSourceViTFFN":
        if record.get("kind") != cls.SNAPSHOT_KIND:
            raise ValueError("unknown source ViT signed snapshot kind")
        spec = SourceViTSpec(**record["spec"])
        with torch.random.fork_rng(devices=[]):
            model = cls(spec, tuple(int(v) for v in record["hidden_widths"]))
        model.load_state_dict(record["state_dict"], strict=True)
        model._width_cursor = int(record.get("cursor", 0))
        model._birth_generation = int(record.get("generation", 0))
        model._source_equivalent_at_birth = bool(record["source_ffn_exact"])
        model._frozen_baseline_sha256 = str(record["frozen_sha256"])
        model.train(False)
        model.verify_frozen()
        if (model.input_dim != int(record["input_dim"])
                or model.output_dim != int(record["output_dim"])
                or record.get("growth_site") != spec.site_id):
            raise ValueError("source ViT frozen FFN local interface/site changed")
        return model
