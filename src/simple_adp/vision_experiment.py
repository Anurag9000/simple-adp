"""Opt-in, provenance-signed V01 real-image frozen-encoder experiment runner.

Only the official training partition is used to train and select candidates.
The official test partition is materialized and evaluated *after* ADP fit.
Real CIFAR/Fashion-MNIST files and optional complete local backbone weights
must already exist, unless download is explicitly enabled. Random encoder
experiments require explicit opt-in and are never labeled pretrained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from .checkpoint import stable_signature
from .config import SearchSettings
from .tasks import ClassificationTask
from .vision_data import ImagePartitions, load_torchvision_partitions
from .vision_encoders import VisionInputAdapter, create_frozen_vision_encoder
from .visual import VisualADPEstimator, _encoder_digest


def _atomic_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(obj, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_vision_experiment(
    *,
    dataset: str,
    root: str | Path,
    output_dir: str | Path,
    architecture_id: str,
    model_id: str = "ADP-W-RR",
    settings: SearchSettings | None = None,
    generic_profile: str = "faithful",
    fixed_grid: Sequence[Sequence[int]] | None = None,
    weights_path: str | Path | None = None,
    expected_weights_sha256: str | None = None,
    allow_random_backbone: bool = False,
    download: bool = False,
    train_limit: int | None = None,
    validation_limit: int | None = None,
    test_limit: int | None = None,
    validation_fraction: float = 0.2,
    split_seed: int = 11,
    encoder_seed: int = 0,
    encoder_batch_size: int = 32,
    # Synthetic fixture injection, for offline unit tests only. Not a CLI path.
    partitions: ImagePartitions | None = None,
    encoder: nn.Module | None = None,
    encoder_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if dataset not in {"cifar10", "cifar100", "fashion_mnist"}:
        raise ValueError("dataset must be one of cifar10, cifar100, fashion_mnist")
    if architecture_id not in {"resnet18", "convnext_tiny", "vit_b_16"}:
        raise ValueError("unknown vision frozen backbone")
    if weights_path is None and not allow_random_backbone:
        raise ValueError("local full-model weights required; explicitly allow random backbone for control")
    if encoder is not None and (encoder_metadata is None or weights_path is not None):
        raise ValueError("fixture encoder requires metadata and no weights_path")
    if encoder is None and encoder_metadata is not None:
        raise ValueError("encoder metadata cannot override the real source loader")
    if encoder_batch_size < 1:
        raise ValueError("encoder_batch_size must be positive")
    if settings is None:
        settings = SearchSettings(
            device="cuda" if torch.cuda.is_available() else "cpu",
            init_width=32, init_depth=1, max_width=64, max_depth=3,
            max_epochs=5, max_candidates=2, batch_size=64,
        )
    parts = (partitions if partitions is not None
             else load_torchvision_partitions(
                 dataset, root, download=download, seed=split_seed,
                 validation_fraction=validation_fraction,
                 train_limit=train_limit, validation_limit=validation_limit,
                 test_limit=test_limit,
             ))
    raw_shape = tuple(parts.train_images.shape[1:])
    if len(raw_shape) != 3 or raw_shape[0] not in {1, 3}:
        raise ValueError("image partition must contain raw [N,1|3,H,W] pixels")
    if not len(parts.fit_indices) or not len(parts.validation_indices) or not len(parts.evaluation_indices):
        raise ValueError("all train, validation and official-test partitions must be nonempty")
    if not set(parts.fit_indices.tolist()).isdisjoint(parts.validation_indices.tolist()):
        raise ValueError("training and validation source indices overlap")
    if encoder is None:
        encoder, encoder_metadata = create_frozen_vision_encoder(
            architecture_id, weights_path=weights_path,
            expected_weights_sha256=expected_weights_sha256,
            seed=encoder_seed,
        )
    assert encoder_metadata is not None
    if encoder_metadata.get("architecture_id") != architecture_id:
        raise ValueError("source encoder metadata does not identify the requested architecture")
    if encoder_metadata.get("weights_source") == "random_initialization" and not allow_random_backbone:
        raise ValueError("random backbone is not permitted without explicit opt-in")
    bridge = VisionInputAdapter(
        encoder, architecture_id=architecture_id, input_channels=int(raw_shape[0]),
        input_resolution=(int(raw_shape[1]), int(raw_shape[2])),
    ).to(settings.device)
    bridge.eval()
    bridge.requires_grad_(False)
    source_state_sha = _encoder_digest(bridge)
    manifest = {
        "format": "simple-adp-vision-v01-experiment-v1",
        "dataset": dataset,
        "partitions": parts.manifest,
        "source_encoder": encoder_metadata,
        "source_encoder_state_sha256": source_state_sha,
        "input_transform": bridge.preprocessing,
        "model_id": model_id,
        "generic_profile": generic_profile,
        "fixed_grid": ([list(map(int, row)) for row in fixed_grid]
                       if fixed_grid is not None else None),
        "settings": settings.as_dict(),
        "encoder_batch_size": int(encoder_batch_size),
        "split_policy": "fit/validation from official training; official test used only after fit",
    }
    manifest["run_signature"] = stable_signature(manifest)
    output = Path(output_dir)
    manifest_path = output / "manifest.json"
    result_path = output / "result.json"
    checkpoint_path = output / "estimator.checkpoint.pt"
    model_path = output / "estimator.model.pt"
    if manifest_path.exists():
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        if stored != manifest:
            raise ValueError("run manifest identity mismatch; use a fresh output directory")
    else:
        if any(path.exists() for path in (result_path, checkpoint_path, model_path)):
            raise ValueError("existing experiment artifacts without signed manifest")
        _atomic_json(manifest_path, manifest)
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("run_signature") != manifest["run_signature"]:
            raise ValueError("existing result identity mismatch")
        if not model_path.is_file() or not (output / "estimator.model.pt.vision.json").is_file():
            raise ValueError("completed result is missing its inference artifact")
        if result.get("model_file_sha256") != _file_sha256(model_path):
            raise ValueError("completed inference model checksum changed")
        if result.get("model_identity_sha256") != _file_sha256(output / "estimator.model.pt.vision.json"):
            raise ValueError("completed inference model identity checksum changed")
        return result

    train_x, train_y = parts.materialize("train")
    valid_x, valid_y = parts.materialize("validation")
    # Official test images/labels deliberately not materialized before fit.
    estimator = VisualADPEstimator(
        bridge,
        encoder_id=architecture_id,
        weights_id=str(encoder_metadata["weights_id"]),
        dataset_manifest=parts.manifest,
        model_id=model_id,
        settings=settings,
        generic_profile=generic_profile,
        fixed_grid=fixed_grid,
        pool="vector",
        expected_resolution=(int(raw_shape[1]), int(raw_shape[2])),
        encoder_batch_size=encoder_batch_size,
    )
    estimator.fit(
        train_x, train_y, validation_data=(valid_x, valid_y),
        checkpoint_path=checkpoint_path,
    )
    if estimator.encoder_sha256 != source_state_sha:
        raise ValueError("frozen encoder hash changed during architecture search")
    if estimator.head.model_ is None:
        raise RuntimeError("ADP fit completed without selected model")
    task: ClassificationTask = estimator.head.task
    valid_features = estimator.features(valid_x)
    valid_prediction = estimator.head.predict(valid_features)
    valid_proba = estimator.head.predict_proba(valid_features)
    validation_metrics = task.evaluation_metrics(valid_y, valid_prediction, valid_proba)
    # This is the ONLY point at which the official test partition is accessed.
    test_x, test_y = parts.materialize("test")
    test_features = estimator.features(test_x)
    test_prediction = estimator.head.predict(test_features)
    test_proba = estimator.head.predict_proba(test_features)
    test_metrics = task.evaluation_metrics(test_y, test_prediction, test_proba)
    estimator.save(model_path)
    result = {
        "format": "simple-adp-vision-v01-result-v1",
        "run_signature": manifest["run_signature"],
        "dataset": dataset, "source": parts.manifest["source"],
        "architecture_id": architecture_id, "model_id": model_id,
        "generic_profile": generic_profile,
        "weights_source": encoder_metadata.get("weights_source"),
        "pretrained": encoder_metadata.get("pretrained"),
        "source_encoder_state_sha256": source_state_sha,
        "selected_architecture": list(estimator.head.model_.hidden_widths),
        "history_entries": len(estimator.head.history_),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "sample_counts": {
            "train": int(len(train_y)),
            "validation": int(len(valid_y)),
            "official_test": int(len(test_y)),
        },
        "model_artifact": model_path.name,
        "checkpoint_artifact": checkpoint_path.name,
        "model_file_sha256": _file_sha256(model_path),
        "model_identity_sha256": _file_sha256(output / "estimator.model.pt.vision.json"),
        "status": "completed",
    }
    _atomic_json(result_path, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a provenance-signed frozen V01 ADP real-image experiment")
    parser.add_argument("--dataset", choices=("cifar10", "cifar100", "fashion_mnist"), required=True)
    parser.add_argument("--root", required=True, help="root containing official local torchvision datasets")
    parser.add_argument("--out", required=True, help="one isolated output folder per experiment")
    parser.add_argument("--architecture", choices=("resnet18", "convnext_tiny", "vit_b_16"), required=True)
    parser.add_argument("--weights", help="complete local source torchvision model state_dict")
    parser.add_argument("--weights-sha256", help="optional expected full source checkpoint SHA256")
    parser.add_argument("--allow-random-backbone", action="store_true")
    parser.add_argument("--download", action="store_true", help="explicitly permit official dataset download")
    parser.add_argument("--model-id", default="ADP-W-RR")
    parser.add_argument("--generic-profile", choices=("faithful", "corrected"), default="faithful")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--valid-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--init-width", type=int, default=32)
    parser.add_argument("--init-depth", type=int, default=1)
    parser.add_argument("--max-width", type=int, default=64)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--encoder-batch-size", type=int, default=32)
    parser.add_argument("--criterion", choices=("loss", "variance_stagnation"), default="loss")
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--fixed-width", type=int, help="required with MLP-Fixed")
    args = parser.parse_args(argv)
    if args.model_id == "MLP-Fixed" and args.fixed_width is None:
        parser.error("MLP-Fixed requires --fixed-width")
    if args.model_id != "MLP-Fixed" and args.fixed_width is not None:
        parser.error("--fixed-width is only supported for MLP-Fixed")
    config = SearchSettings(
        seed=args.seed, device=args.device,
        init_width=args.init_width, init_depth=args.init_depth,
        max_width=args.max_width, max_depth=args.max_depth,
        max_epochs=args.max_epochs, max_candidates=args.max_candidates,
        batch_size=args.batch_size, expansion_criterion=args.criterion,
    )
    result = run_vision_experiment(
        dataset=args.dataset, root=args.root, output_dir=args.out,
        architecture_id=args.architecture, model_id=args.model_id,
        generic_profile=args.generic_profile, settings=config,
        fixed_grid=((args.fixed_width,),) if args.fixed_width is not None else None,
        weights_path=args.weights, expected_weights_sha256=args.weights_sha256,
        allow_random_backbone=args.allow_random_backbone,
        download=args.download, split_seed=args.seed,
        train_limit=args.train_limit, validation_limit=args.valid_limit,
        test_limit=args.test_limit,
        encoder_batch_size=args.encoder_batch_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
