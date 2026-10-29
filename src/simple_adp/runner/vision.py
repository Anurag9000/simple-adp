"""Opt-in coordinated vision dataset/model adapters for V01 ADP heads.

Never selected by the default legacy suite. Explicit task group 'vision'
activates routes; only the 'vision' dataset provider pairs with these routes.
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from ..config import SearchSettings
from ..registry import MODEL_REGISTRY
from ..tasks import ClassificationTask
from ..vision_data import make_image_partitions, load_torchvision_partitions
from ..visual import VisualADPEstimator
from .artifacts import estimator_model_path
from .datasets import DatasetTaskRef, PreparedTaskData
from .plan import SplitSpec, _sha_indices
from .route import ModelRoute


class VisionDatasetProvider:
    """Separate raw-image data, including an offline deterministic smoke set."""

    name = "vision"
    suites = ("toy", "cifar10", "cifar100", "fashion_mnist")

    def list_tasks(self, suite: str) -> list[DatasetTaskRef]:
        if suite not in self.suites:
            raise ValueError(f"unknown vision suite {suite!r}")
        return [DatasetTaskRef(self.name, suite, suite + "-v01")]

    @staticmethod
    def _limit(name: str) -> int | None:
        value = os.environ.get(name, "").strip()
        if not value:
            return None
        parsed = int(value)
        if parsed <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return parsed

    @staticmethod
    def _toy():
        rng = np.random.default_rng(1101)
        x = rng.integers(0, 60, size=(48, 1, 8, 8), dtype=np.uint8)
        y = np.array([0, 1] * 24, dtype=np.int64)
        x[y == 1] += 150
        tx = rng.integers(0, 60, size=(16, 1, 8, 8), dtype=np.uint8)
        ty = np.array([0, 1] * 8, dtype=np.int64)
        tx[ty == 1] += 150
        return make_image_partitions(
            x, y, tx, ty, source="synthetic-vision-smoke",
            revision="toy-v01-seed1101", seed=11,
        )

    def load_task(self, ref: DatasetTaskRef) -> PreparedTaskData:
        if ref.provider != self.name or ref.suite not in self.suites:
            raise ValueError("task ref does not identify a supported vision dataset")
        if ref.task_id != ref.suite + "-v01":
            raise ValueError("unknown vision task ID")
        if ref.suite == "toy":
            parts = self._toy()
        else:
            root = os.environ.get("SIMPLE_ADP_VISION_ROOT", "data/vision")
            parts = load_torchvision_partitions(
                ref.suite, root, download=False, seed=11,
                train_limit=self._limit("SIMPLE_ADP_VISION_TRAIN_LIMIT"),
                validation_limit=self._limit("SIMPLE_ADP_VISION_VALID_LIMIT"),
                test_limit=self._limit("SIMPLE_ADP_VISION_TEST_LIMIT"),
            )
        train, valid, test = parts.fit_indices, parts.validation_indices, parts.evaluation_indices
        X = np.concatenate((
            parts.train_images[train], parts.train_images[valid], parts.test_images[test],
        ), axis=0)
        y = np.concatenate((
            parts.train_labels[train], parts.train_labels[valid], parts.test_labels[test],
        ), axis=0)
        official_source_id = np.concatenate((train, valid, test)).astype(np.int64)
        official_split_id = np.concatenate((
            np.zeros(len(train) + len(valid), dtype=np.int64),
            np.ones(len(test), dtype=np.int64),
        ))
        counts = (len(train), len(valid), len(test))
        identity = hashlib.sha256()
        identity.update(X.tobytes())
        identity.update(y.tobytes())
        identity.update(official_source_id.tobytes())
        identity.update(str(parts.manifest).encode())
        metadata = {
            "architecture_id": "v01-tiny-conv-frozen-seed7129",
            "pretraining": "none; frozen random CNN (separate control, not pretrained)",
            "normalization": "input raw uint8 / 255",
            "raw_source_sha256": parts.manifest["train_raw_sha256"],
            "raw_test_sha256": parts.manifest["test_raw_sha256"],
            "original_train_indices_sha256": hashlib.sha256(train.tobytes()).hexdigest(),
            "original_valid_indices_sha256": hashlib.sha256(valid.tobytes()).hexdigest(),
            "original_test_indices_sha256": hashlib.sha256(test.tobytes()).hexdigest(),
            "split_group_sizes": list(counts),
            "source": parts.manifest["source"],
            "revision": parts.manifest["revision"],
        }
        return PreparedTaskData(
            dataset_id=identity.hexdigest()[:20], name=ref.suite,
            target_name="class", X=X, y=y,
            feature_schema=None, content_sha256=identity.hexdigest(),
            dataset_metadata=metadata,
            task_kind="classification", task_adapter=ClassificationTask(),
            task_family="vision", task_variant=ref.suite,
            split_protocol="official train with stratified architecture validation; official test isolated",
            context={"source_index": official_source_id, "official_test_group": official_split_id},
        )

    def list_splits(self, ref: DatasetTaskRef, data: PreparedTaskData, settings: Any) -> list[SplitSpec]:
        if data.task_family != "vision":
            raise ValueError("not a vision task")
        if abs(float(settings.validation_fraction) - 0.2) > 1e-12:
            raise ValueError("vision v01 provider currently requires validation_fraction=0.2")
        n_train, n_valid, n_test = data.dataset_metadata["split_group_sizes"]
        train = tuple(range(n_train))
        valid = tuple(range(n_train, n_train + n_valid))
        test = tuple(range(n_train + n_valid, n_train + n_valid + n_test))
        return [SplitSpec(0, 0, 0, train, valid, test, _sha_indices(train, valid, test))]


class VisionModelAdapter:
    """V01 routes through the native ADP controllers, not a parallel search."""

    name = "vision_adp"

    def enumerate_routes(self, generic_profiles=("faithful", "corrected"), settings=None) -> list[ModelRoute]:
        if settings is None or "vision" not in settings.tasks:
            return []
        if not generic_profiles or set(generic_profiles) - {"faithful", "corrected"}:
            raise ValueError("unsupported generic profile")
        routes = []
        for model_id, config in MODEL_REGISTRY.items():
            if model_id == "MLP-Fixed":
                routes.append(ModelRoute(
                    model_id, label="V01:MLP-Fixed",
                    fixed_grid=((settings.width_to_depth_width,),), suite_group="vision",
                    init_width=settings.width_to_depth_width, init_depth=1,
                ))
                continue
            profiles = generic_profiles if config.generic else (None,)
            for profile in profiles:
                label = f"V01:{model_id}" + (f"@{profile}" if profile else "")
                routes.append(ModelRoute(
                    model_id, label=label, generic_profile=profile,
                    suite_group="vision",
                    init_width=settings.width_to_depth_width,
                    init_depth=settings.width_to_depth_depth,
                ))
        return routes

    @staticmethod
    def _frozen_encoder(channels: int, device: str) -> nn.Module:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7129)
            module = nn.Sequential(
                nn.Conv2d(channels, 8, kernel_size=3, padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
        return module.to(device).eval().requires_grad_(False)

    def fit_predict(
        self,
        route: ModelRoute,
        *,
        data: PreparedTaskData,
        split: SplitSpec,
        seed: int,
        settings: SearchSettings,
        checkpoint_path: str | None = None,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> dict[str, Any]:
        if data.task_family != "vision" or route.suite_group != "vision":
            raise ValueError("vision adapter requires a vision-only route and dataset")
        if route.model_id not in MODEL_REGISTRY:
            raise ValueError("unknown native ADP method")
        def take(rows):
            values = np.asarray(data.X[np.asarray(rows, dtype=np.int64)], dtype=np.float32)
            return values / 255.0
        train_x, valid_x, test_x = (take(split.train_indices), take(split.validation_indices),
                                   take(split.test_indices))
        train_y = np.asarray(data.y[list(split.train_indices)])
        valid_y = np.asarray(data.y[list(split.validation_indices)])
        test_y = np.asarray(data.y[list(split.test_indices)])
        actual_settings = SearchSettings(**{
            **settings.as_dict(),
            "seed": int(seed),
            "init_width": route.init_width or settings.init_width,
            "init_depth": route.init_depth or settings.init_depth,
        })
        encoder = self._frozen_encoder(int(train_x.shape[1]), actual_settings.device)
        manifest = {
            "dataset_id": data.dataset_id,
            "content_sha256": data.content_sha256,
            "dataset_metadata": dict(data.dataset_metadata),
            "split_sha256": split.split_sha256,
            "source_train_indices": np.asarray(data.context["source_index"])[list(split.train_indices)].tolist(),
            "source_validation_indices": np.asarray(data.context["source_index"])[list(split.validation_indices)].tolist(),
            "source_test_indices": np.asarray(data.context["source_index"])[list(split.test_indices)].tolist(),
            "preprocessing": "raw uint8 / 255",
        }
        estimator = VisualADPEstimator(
            encoder, encoder_id="v01-tiny-conv", weights_id="frozen-seed7129-random",
            dataset_manifest=manifest, model_id=route.model_id,
            task=ClassificationTask(), settings=actual_settings,
            generic_profile=route.generic_profile or "faithful",
            pool="vector", expected_resolution=tuple(train_x.shape[-2:]),
            fixed_grid=route.fixed_grid,
        )
        start = time.perf_counter()
        estimator.fit(train_x, train_y, validation_data=(valid_x, valid_y),
                      checkpoint_path=checkpoint_path,
                      progress_callback=progress_callback)
        duration = time.perf_counter() - start
        task = estimator.head.task
        valid_pred = estimator.predict(valid_x)
        valid_prob = estimator.predict_proba(valid_x)
        test_pred = estimator.predict(test_x)
        test_prob = estimator.predict_proba(test_x)
        valid_outputs = estimator.head._outputs(estimator.features(valid_x))
        valid_targets = task.encode_targets(valid_y, device=valid_outputs.device)
        valid_loss = task.validation_score(valid_outputs, valid_targets)
        artifact = None
        if checkpoint_path is not None:
            artifact = str(estimator_model_path(checkpoint_path))
            estimator.save(artifact)
        return {
            "fit_seconds": duration, "history_entries": len(estimator.head.history_),
            "selected_architecture": list(estimator.head.model_.hidden_widths),
            "validation_selection_score": float(valid_loss),
            "validation_metrics": task.evaluation_metrics(valid_y, valid_pred, valid_prob),
            "test_metrics": task.evaluation_metrics(test_y, test_pred, test_prob),
            "task_family": "vision", "task_variant": data.task_variant,
            "split_protocol": data.split_protocol,
            "primary_metric": task.primary_metric,
            "metric_directions": task.metric_directions,
            "artifact_path": artifact,
            "encoder_sha256": estimator.encoder_sha256,
            "growth_site": "head",
            "architecture_id": "v01-tiny-conv-frozen-random-control",
        }
