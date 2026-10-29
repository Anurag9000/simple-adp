"""V01 frozen image encoder with canonical ADP-trained, growing MLP head.

The supplied encoder must return unclassified image features. For trainable
inner-backbone sites V02-V11 a composite candidate controller is required.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .config import SearchSettings
from .estimator import TabularADP
from .tasks import ClassificationTask, TaskAdapter


def _manifest_digest(manifest: Mapping[str, Any]) -> str:
    data = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(data.encode()).hexdigest()


def _encoder_digest(encoder: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(encoder.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        for part in (name, str(array.dtype), str(array.shape)):
            digest.update(part.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class VisualADPEstimator:
    """Frozen image features feed the original TabularADP controller.

    Use classifier-free ResNet/ConvNeXt/ViT feature extractors. The caller owns
    preprocessing and an immutable manifest describing sample IDs and splits.
    The head is a real, trainable TabularMLP; encoder features may be cached.
    """

    def __init__(
        self,
        encoder: nn.Module,
        *,
        encoder_id: str,
        weights_id: str,
        dataset_manifest: Mapping[str, Any],
        model_id: str = "ADP-W-RR",
        task: TaskAdapter | None = None,
        settings: SearchSettings | None = None,
        generic_profile: str = "faithful",
        pool: str = "vector",
        expected_resolution: tuple[int, int] | None = None,
        fixed_grid: Sequence[Sequence[int]] | None = None,
        encoder_batch_size: int = 64,
    ) -> None:
        if not isinstance(encoder, nn.Module):
            raise TypeError("encoder must be nn.Module")
        if not encoder_id.strip() or not weights_id.strip():
            raise ValueError("encoder_id and weights_id must be nonempty")
        if not dataset_manifest:
            raise ValueError("dataset_manifest must describe train/validation provenance")
        if pool not in {"vector", "cls", "tokens_mean", "spatial_mean"}:
            raise ValueError("unsupported encoder pooling mode")
        if expected_resolution is not None and (len(expected_resolution) != 2 or min(expected_resolution) <= 0):
            raise ValueError("invalid expected_resolution")
        if encoder_batch_size <= 0:
            raise ValueError("encoder_batch_size must be positive")
        self.encoder = encoder.eval()
        self.encoder.requires_grad_(False)
        self.encoder_id = encoder_id
        self.weights_id = weights_id
        self.dataset_digest = _manifest_digest(dataset_manifest)
        self.encoder_sha256 = _encoder_digest(encoder)
        self.pool = pool
        self.expected_resolution = tuple(expected_resolution) if expected_resolution else None
        self.encoder_batch_size = int(encoder_batch_size)
        self.head = TabularADP(
            model_id,
            task=task if task is not None else ClassificationTask(),
            settings=settings or SearchSettings(device="cuda" if torch.cuda.is_available() else "cpu"),
            generic_profile=generic_profile,
            fixed_grid=fixed_grid,
        )
        self.feature_dim_: int | None = None

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "format": "visual-adp-v01-v1",
            "encoder_id": self.encoder_id,
            "weights_id": self.weights_id,
            "encoder_sha256": self.encoder_sha256,
            "dataset_manifest_sha256": self.dataset_digest,
            "pool": self.pool,
            "expected_resolution": list(self.expected_resolution) if self.expected_resolution else None,
            "model_id": self.head.config.id,
            "generic_profile": self.head.generic_profile,
            "search_settings": self.head.settings.as_dict(),
            "fixed_grid": [list(v) for v in self.head.fixed_grid] if self.head.fixed_grid else None,
        }

    def _verify_frozen(self) -> None:
        if _encoder_digest(self.encoder) != self.encoder_sha256:
            raise ValueError("frozen encoder parameters or buffers changed")
        self.encoder.eval()
        if any(p.requires_grad for p in self.encoder.parameters()):
            raise ValueError("encoder must be frozen")

    def features(self, images: Any) -> np.ndarray:
        self._verify_frozen()
        x = images.detach() if torch.is_tensor(images) else torch.as_tensor(np.asarray(images))
        if x.ndim != 4 or len(x) == 0:
            raise ValueError("images must have nonempty [N,C,H,W] shape")
        if self.expected_resolution and tuple(x.shape[-2:]) != self.expected_resolution:
            raise ValueError("image resolution mismatch")
        if not torch.isfinite(x).all():
            raise ValueError("images must be finite")
        first = next(self.encoder.parameters(), None)
        if first is None:
            first = next(self.encoder.buffers(), None)
        device = first.device if first is not None else torch.device("cpu")
        chunks = []
        with torch.no_grad():
            for offset in range(0, len(x), self.encoder_batch_size):
                output = self.encoder(x[offset:offset + self.encoder_batch_size].to(device=device, dtype=torch.float32))
                if not torch.is_tensor(output):
                    raise TypeError("encoder must return one tensor; wrap tuple/dict outputs")
                if self.pool == "vector" and output.ndim == 2:
                    vectors = output
                elif self.pool == "cls" and output.ndim == 3:
                    vectors = output[:, 0]
                elif self.pool == "tokens_mean" and output.ndim == 3:
                    vectors = output.mean(dim=1)
                elif self.pool == "spatial_mean" and output.ndim == 4:
                    vectors = output.mean(dim=(-2, -1))
                else:
                    raise ValueError("encoder output rank incompatible with pooling")
                if vectors.shape[1] == 0 or not torch.isfinite(vectors).all():
                    raise ValueError("encoder features must be finite and nonempty")
                chunks.append(vectors.float().cpu().numpy())
        features = np.concatenate(chunks, axis=0)
        self._verify_frozen()
        if self.feature_dim_ is not None and features.shape[1] != self.feature_dim_:
            raise ValueError("encoder feature dimensionality changed")
        return features

    def fit(self, images: Any, labels: Any, *, validation_data: tuple[Any, Any],
            checkpoint_path: str | os.PathLike[str] | None = None) -> "VisualADPEstimator":
        if not isinstance(validation_data, tuple) or len(validation_data) != 2:
            raise ValueError("explicit validation_data=(images,labels) is required")
        train = self.features(images)
        valid = self.features(validation_data[0])
        if train.shape[1] != valid.shape[1]:
            raise ValueError("train/validation encoder feature dimensions differ")
        self.feature_dim_ = int(train.shape[1])
        if checkpoint_path is not None:
            sidecar = Path(str(checkpoint_path) + ".vision.json")
            expected = {**self.identity, "feature_dim": self.feature_dim_}
            if sidecar.exists():
                if json.loads(sidecar.read_text(encoding="utf-8")) != expected:
                    raise ValueError("visual checkpoint identity mismatch")
            else:
                if Path(checkpoint_path).exists():
                    raise ValueError("head checkpoint without visual identity")
                sidecar.parent.mkdir(parents=True, exist_ok=True)
                tmp = sidecar.with_name(sidecar.name + ".tmp")
                tmp.write_text(json.dumps(expected, sort_keys=True, indent=2), encoding="utf-8")
                os.replace(tmp, sidecar)
        self.head.fit(train, labels, validation_data=(valid, validation_data[1]),
                      checkpoint_path=checkpoint_path)
        self._verify_frozen()
        return self

    def predict(self, images: Any) -> np.ndarray:
        return self.head.predict(self.features(images))

    def predict_proba(self, images: Any) -> np.ndarray:
        return self.head.predict_proba(self.features(images))

    def save(self, path: str | os.PathLike[str]) -> None:
        if self.head.model_ is None:
            raise RuntimeError("fit before saving")
        self._verify_frozen()
        self.head.save(path)
        Path(str(path) + ".vision.json").write_text(
            json.dumps({**self.identity, "feature_dim": self.feature_dim_}, sort_keys=True, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str], *, encoder: nn.Module,
             encoder_id: str, weights_id: str, dataset_manifest: Mapping[str, Any],
             task: TaskAdapter | None = None, map_location: str = "cpu") -> "VisualADPEstimator":
        metadata = json.loads(Path(str(path) + ".vision.json").read_text(encoding="utf-8"))
        if metadata.get("format") != "visual-adp-v01-v1":
            raise ValueError("unsupported visual artifact")
        head = TabularADP.load(path, task=task, map_location=map_location)
        obj = cls(
            encoder, encoder_id=encoder_id, weights_id=weights_id,
            dataset_manifest=dataset_manifest, model_id=head.config.id,
            task=head.task, settings=head.settings, generic_profile=head.generic_profile,
            pool=metadata["pool"],
            expected_resolution=tuple(metadata["expected_resolution"]) if metadata["expected_resolution"] else None,
            fixed_grid=head.fixed_grid,
        )
        obj.feature_dim_ = int(metadata["feature_dim"])
        if obj.identity != {k: v for k, v in metadata.items() if k != "feature_dim"}:
            raise ValueError("encoder, weights, provenance or method identity mismatch")
        obj.head = head
        return obj
