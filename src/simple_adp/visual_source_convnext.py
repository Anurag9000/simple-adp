"""Source-faithful V04 ConvNeXt-Tiny first CNBlock channel FFN adaptation.

Loads an exact complete local torchvision ConvNeXt-Tiny state_dict, transplants
the ORIGINAL pretrained two-layer GELU FFN inside features[1][0].block,
and provides source ADP width/depth mutation only for its internal hidden
neurons. Downstream frozen CNN operations remain in the gradient graph.
The fresh task classifier trains but NEVER grows. No implicit weight download.
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

_SOURCE_DIM = 96
_SOURCE_HIDDEN = 384
_ORIGINAL_FFN_SITE = "features.1.0.block.3"
_SITE = "convnext:features.1.0.channel_ffn"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


@dataclass(frozen=True)
class SourceConvNeXtSpec:
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
            raise ValueError("source ConvNeXt requires >=2 classes and positive image geometry")
        if self.input_channels not in {1, 3}:
            raise ValueError("source ConvNeXt requires grayscale or RGB images")
        if self.site_id != _SITE:
            raise ValueError("source ConvNeXt has one eligible inner FFN site")
        if self.normalization not in {"imagenet", "none"}:
            raise ValueError("unsupported source ConvNeXt normalization")
        if not self.weights_path or len(self.weights_sha256) != 64 or any(
            c not in "0123456789abcdefABCDEF" for c in self.weights_sha256
        ):
            raise ValueError("complete local source checkpoint path and SHA256 required")


class VisualSourceConvNeXtFFN(VisualResidualAdapter):
    """Frozen full ConvNeXt source with one source-weight-preserving trainable FFN.

    The first CNBlock channel FFN is C=96 -> H=384 -> C=96 with GELU,
    transplanted directly from the official source architecture's exact
    Linear/GELU/Linear tensors. Replacing its two source Linear layers with
    TabularMLP initially preserves the source feature computation. A fresh
    task head is separately trainable, with fixed input/output dimensions.
    """

    SNAPSHOT_KIND = "visual-v04-source-convnext-ffn-v1"
    SOURCE_WIDTH = _SOURCE_HIDDEN

    def __init__(
        self,
        spec: SourceConvNeXtSpec,
        hidden_widths: tuple[int, ...] = (_SOURCE_HIDDEN,),
    ) -> None:
        nn.Module.__init__(self)
        if not hidden_widths or min(hidden_widths) <= 0:
            raise ValueError("source ConvNeXt FFN hidden widths must be positive")
        self.spec = spec
        source_file = Path(spec.weights_path)
        if not source_file.is_file():
            raise ValueError("local ConvNeXt source checkpoint is missing")
        if file_sha256(source_file).lower() != spec.weights_sha256.lower():
            raise ValueError("source ConvNeXt checkpoint SHA256 mismatch")
        try:
            from torchvision import models
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError("source ConvNeXt requires torchvision") from exc
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(spec.fixed_seed))
            model = models.convnext_tiny(weights=None)
        payload = torch.load(source_file, weights_only=True, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError("source ConvNeXt checkpoint must be a named tensor state_dict")
        if "state_dict" in payload and isinstance(payload["state_dict"], dict):
            payload = payload["state_dict"]
        if not isinstance(payload, dict) or not payload or not all(
            isinstance(k, str) and torch.is_tensor(v) for k, v in payload.items()
        ):
            raise ValueError("source ConvNeXt checkpoint must be a named tensor state_dict")
        model.load_state_dict(payload, strict=True)
        block = model.features[1][0].block
        if (not isinstance(block[3], nn.Linear) or not isinstance(block[4], nn.GELU)
                or not isinstance(block[5], nn.Linear)
                or block[3].in_features != _SOURCE_DIM
                or block[3].out_features != _SOURCE_HIDDEN
                or block[5].in_features != _SOURCE_HIDDEN
                or block[5].out_features != _SOURCE_DIM):
            raise ValueError("loaded torchvision source FFN shape/layout differs from expected V04 site")
        # Source head was strict-loaded above. Replace only its *output
        # classifier*, not any CNN stage, with a newly seeded target head.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(spec.fixed_seed) + 101)
            model.classifier[-1] = nn.Linear(
                model.classifier[-1].in_features, spec.classes,
            )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(spec.fixed_seed) + 102)
            local = TabularMLP(_SOURCE_DIM, hidden_widths, _SOURCE_DIM)
        # An exact source-equivalent initialization occurs for [384]. For
        # any later growth width, the source's overlapping tensors transfer.
        with torch.no_grad():
            first = next(layer for layer in local.hidden_layers if isinstance(layer, nn.Linear))
            rows = min(first.weight.shape[0], block[3].weight.shape[0])
            first.weight[:rows].copy_(block[3].weight[:rows])
            first.bias[:rows].copy_(block[3].bias[:rows])
            columns = min(local.output_layer.weight.shape[1], block[5].weight.shape[1])
            local.output_layer.weight[:, :columns].copy_(block[5].weight[:, :columns])
            local.output_layer.bias.copy_(block[5].bias)
        for i, layer in enumerate(local.hidden_layers):
            if isinstance(layer, nn.ReLU):
                local.hidden_layers[i] = nn.GELU()
        block[3] = local
        block[4] = nn.Identity()
        block[5] = nn.Identity()
        self.source = model
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
    def task_head(self) -> nn.Linear:
        return self.source.classifier[-1]

    @property
    def local_mlp(self) -> TabularMLP:
        return self.source.features[1][0].block[3]

    @property
    def source_ffn_exact_at_initialization(self) -> bool:
        return bool(self._source_equivalent_at_birth)

    def _fixed_modules(self) -> tuple[nn.Module, ...]:
        # Frozen verification below acts over source state rather than a
        # container that accidentally includes the selected local module.
        return ()

    def frozen_state_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(f"V04-source:{self.spec.site_id}:{self.spec.weights_sha256}".encode())
        skip = (_ORIGINAL_FFN_SITE + ".", "classifier.2.")
        for name, tensor in sorted(self.source.state_dict().items()):
            if name.startswith(skip):
                continue
            array = tensor.detach().cpu().contiguous().numpy()
            digest.update(f"{name}:{array.dtype}:{array.shape}".encode())
            digest.update(array.tobytes())
        for name, tensor in (("input_mean", self.input_mean), ("input_std", self.input_std)):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def verify_frozen(self) -> None:
        if self.frozen_state_sha256() != self._frozen_baseline_sha256:
            raise ValueError("source ConvNeXt frozen state changed")
        for name, parameter in self.source.named_parameters():
            mutable = (name.startswith(_ORIGINAL_FFN_SITE + ".")
                       or name.startswith("classifier.2."))
            if bool(parameter.requires_grad) != mutable:
                raise ValueError("source ConvNeXt parameter isolation violated")
        allowed_training = {
            id(layer)
            for mutable in (self.local_mlp, self.task_head)
            for layer in mutable.modules()
        }
        if self.source.training or any(
            module.training for module in self.source.modules()
            if id(module) not in allowed_training
        ):
            raise ValueError("source ConvNeXt frozen modules must remain in eval mode")

    def train(self, mode: bool = True) -> "VisualSourceConvNeXtFFN":
        nn.Module.train(self, False)
        self.local_mlp.train(mode)
        self.task_head.train(mode)
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if (images.ndim != 4 or images.shape[1] != self.spec.input_channels
                or tuple(images.shape[-2:]) !=
                (self.spec.image_height, self.spec.image_width)):
            raise ValueError("source ConvNeXt input must match fixed NCHW geometry")
        if not images.is_floating_point() or not torch.isfinite(images).all():
            raise ValueError("source ConvNeXt images must be finite float pixels")
        image = images.repeat(1, 3, 1, 1) if self.spec.input_channels == 1 else images
        if min(image.shape[-2:]) < 32:
            image = nn.functional.interpolate(
                image, size=(max(32, image.shape[-2]), max(32, image.shape[-1])),
                mode="bilinear", align_corners=False,
            )
        return self.source((image - self.input_mean) / self.input_std)

    def grow_width(self, *, site_id: str, operator: str, increment: int,
                   max_width: int) -> "VisualSourceConvNeXtFFN | None":
        if site_id != self.spec.site_id:
            raise ValueError("noneligible source ConvNeXt growth site")
        ops = {
            "round_robin": expand_width_round_robin,
            "den_round_robin": expand_width_den_round_robin,
            "all_layers": expand_width_all_layers,
        }
        if operator not in ops:
            raise ValueError("unknown source ConvNeXt width growth operator")
        widened = ops[operator](self.local_mlp, increment, max_width)
        if widened is None:
            return None
        candidate = copy.deepcopy(self)
        candidate.source.features[1][0].block[3] = widened
        candidate._gelu_hidden()
        candidate._birth_generation += 1
        candidate.train(self.training)
        candidate.verify_frozen()
        return candidate

    def grow_depth(self, *, site_id: str, operator: str, max_depth: int,
                   seed_width: int = 1) -> "VisualSourceConvNeXtFFN | None":
        if site_id != self.spec.site_id:
            raise ValueError("noneligible source ConvNeXt growth site")
        if operator == "D1":
            local = expand_depth(self.local_mlp, max_depth,
                                 min_new_layer_width=seed_width)
        elif operator == "Dcur":
            local = expand_depth_match_width(self.local_mlp, max_depth)
        else:
            raise ValueError("unknown source ConvNeXt depth growth operator")
        if local is None:
            return None
        candidate = copy.deepcopy(self)
        candidate.source.features[1][0].block[3] = local
        candidate._gelu_hidden()
        candidate._birth_generation += 1
        candidate.train(self.training)
        candidate.verify_frozen()
        return candidate

    def _gelu_hidden(self) -> None:
        for i, layer in enumerate(self.local_mlp.hidden_layers):
            if isinstance(layer, nn.ReLU):
                self.local_mlp.hidden_layers[i] = nn.GELU()

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
    def from_snapshot(cls, record: dict[str, object]) -> "VisualSourceConvNeXtFFN":
        if record.get("kind") != cls.SNAPSHOT_KIND:
            raise ValueError("unknown source ConvNeXt signed checkpoint kind")
        spec = SourceConvNeXtSpec(**record["spec"])
        with torch.random.fork_rng(devices=[]):
            model = cls(spec, tuple(int(x) for x in record["hidden_widths"]))
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
            raise ValueError("source ConvNeXt source FFN interface/site changed")
        return model
