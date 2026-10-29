"""Concrete torchvision frozen encoders for V01 feature-head ADP.

Strictly local checkpoint loading; no implicit network access or hidden
pretraining. Original classifier heads are stripped only after exact weights
have been loaded into the complete source architecture.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from .visual import _encoder_digest

ArchitectureID = Literal["resnet18", "convnext_tiny", "vit_b_16"]
NormalizationID = Literal["none", "imagenet"]

_FEATURE_DIMS = {"resnet18": 512, "convnext_tiny": 768, "vit_b_16": 768}
_MIN_RESOLUTION = {"resnet18": 32, "convnext_tiny": 32, "vit_b_16": 224}


class FrozenVisionEncoder(nn.Module):
    """Validate exact interface, normalize pixels, and produce [B, D]."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        architecture_id: str,
        normalization: NormalizationID,
        expected_resolution: tuple[int, int] | None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.architecture_id = architecture_id
        self.normalization = normalization
        self.expected_resolution = expected_resolution
        if normalization == "imagenet":
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        elif normalization == "none":
            mean, std = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
        else:
            raise ValueError("normalization must be none or imagenet")
        self.register_buffer("input_mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("input_std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))
        self.eval()
        self.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("torchvision backbone expects [B,3,H,W] images")
        spatial_shape = tuple(images.shape[-2:])
        if self.expected_resolution is not None and spatial_shape != self.expected_resolution:
            raise ValueError(f"backbone resolution {spatial_shape} differs from {self.expected_resolution}")
        minimum = _MIN_RESOLUTION[self.architecture_id]
        if min(spatial_shape) < minimum:
            raise ValueError(f"{self.architecture_id} requires minimum dimension {minimum}")
        features = self.backbone((images - self.input_mean) / self.input_std)
        if features.ndim != 2 or features.shape[1] != _FEATURE_DIMS[self.architecture_id]:
            raise ValueError("backbone did not produce its declared feature-vector dimension")
        return features

    def train(self, mode: bool = True) -> "FrozenVisionEncoder":
        # A caller's train() must not drift frozen BatchNorm/dropout buffers.
        super().train(False)
        return self


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def create_frozen_vision_encoder(
    architecture_id: ArchitectureID,
    *,
    weights_path: str | Path | None = None,
    expected_weights_sha256: str | None = None,
    seed: int = 0,
    normalization: NormalizationID | None = None,
) -> tuple[FrozenVisionEncoder, dict[str, Any]]:
    """Return (frozen encoder, reproducible model identity).

    A local weights_path must be a complete **torchvision source model**
    state_dict saved before its classifier is replaced. Use weights_only=True
    and load_state_dict(strict=True). The function does not download weights.
    For ViT-B/16, resolution is fixed at 224x224; ConvNeXt and ResNet accept
    variable image shapes down to 32x32 with site-specific architectural limits.
    """
    if architecture_id not in _FEATURE_DIMS:
        raise ValueError(f"unknown frozen V01 encoder architecture {architecture_id!r}")
    if weights_path is None and expected_weights_sha256 is not None:
        raise ValueError("weights checksum requires a supplied checkpoint file")
    if normalization is None:
        normalization = "imagenet" if weights_path is not None else "none"
    if normalization not in {"none", "imagenet"}:
        raise ValueError("unknown vision input normalization")
    try:
        from torchvision import models
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("torchvision vision encoders require optional simple-adp[vision]") from exc

    constructors = {
        "resnet18": models.resnet18,
        "convnext_tiny": models.convnext_tiny,
        "vit_b_16": models.vit_b_16,
    }
    # torchvision initializes CPU weights; isolate its RNG draws from ADP's
    # candidate initialization. The saved pretrained state overwrites weights.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        backbone = constructors[architecture_id](weights=None)
    source_file_sha256: str | None = None
    if weights_path is not None:
        path = Path(weights_path)
        source_file_sha256 = _file_sha256(path)
        if expected_weights_sha256 is not None and source_file_sha256.lower() != expected_weights_sha256.lower():
            raise ValueError("pretrained weight file SHA256 does not match expected digest")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise TypeError("weights checkpoint must be a source state_dict mapping")
        if "state_dict" in payload and isinstance(payload["state_dict"], dict):
            state = payload["state_dict"]
        else:
            state = payload
        if not state or not all(isinstance(key, str) and isinstance(value, torch.Tensor)
                                for key, value in state.items()):
            raise TypeError("weights checkpoint must contain only named tensors")
        backbone.load_state_dict(state, strict=True)
    if architecture_id == "resnet18":
        backbone.fc = nn.Identity()
    elif architecture_id == "convnext_tiny":
        backbone.classifier[-1] = nn.Identity()
    else:
        backbone.heads = nn.Identity()

    resolution = (224, 224) if architecture_id == "vit_b_16" else None
    encoder = FrozenVisionEncoder(
        backbone,
        architecture_id=architecture_id,
        normalization=normalization,
        expected_resolution=resolution,
    )
    digest = _encoder_digest(encoder)
    local_checkpoint = source_file_sha256 is not None
    metadata = {
        "model_family": "vision",
        "architecture_id": architecture_id,
        "weights_id": ("local_state_sha256:" + source_file_sha256) if local_checkpoint else f"random_initialization_seed:{int(seed)}",
        "weights_file_sha256": source_file_sha256,
        "encoder_state_sha256": digest,
        "weights_source": "unverified_local_state_dict" if local_checkpoint else "random_initialization",
        "pretrained": None if local_checkpoint else False,
        "pretraining_dataset": None,
        "normalization": normalization,
        "normalization_mean": [0.485, 0.456, 0.406] if normalization == "imagenet" else [0, 0, 0],
        "normalization_std": [0.229, 0.224, 0.225] if normalization == "imagenet" else [1, 1, 1],
        "expected_resolution": list(resolution) if resolution else None,
        "feature_dim": _FEATURE_DIMS[architecture_id],
        "pool": "vector",
        "interface": "[B,3,H,W] to [B,D]; frozen encoder, trainable ADP head only",
    }
    return encoder, metadata
