"""Explicit opt-in coordinated source-faithful V04 ConvNeXt runner.

Unlike the existing tiny random vision_v04, this adapter uses the *complete*
locally supplied torchvision ConvNeXt-Tiny state_dict and the real first
CNBlock channel FFN. Enabling it requires a positive source gate and a
source checkpoint + expected checksum. The checkpoint SHA256 becomes a
literal part of every route label and therefore the planner's job identity.
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from ..checkpoint import CheckpointStore
from ..config import SearchSettings
from ..registry import MODEL_REGISTRY
from ..visual_search import VisualResidualSearch
from ..visual_source_convnext import SourceConvNeXtSpec, file_sha256
from .artifacts import estimator_model_path
from .datasets import PreparedTaskData
from .plan import SplitSpec
from .route import ModelRoute


class VisionSourceConvNeXtModelAdapter:
    name = "vision_v04_source"
    family = "V04-source"
    SOURCE_WIDTH = 384

    @staticmethod
    def _source() -> tuple[str, str]:
        if os.environ.get("SIMPLE_ADP_ENABLE_V04_SOURCE") != "1":
            raise ValueError("source V04 requires explicit SIMPLE_ADP_ENABLE_V04_SOURCE=1")
        path_text = os.environ.get("SIMPLE_ADP_V04_SOURCE_WEIGHTS", "").strip()
        expected = os.environ.get("SIMPLE_ADP_V04_SOURCE_SHA256", "").strip().lower()
        if not path_text or len(expected) != 64 or any(
                c not in "0123456789abcdef" for c in expected):
            raise ValueError("source V04 requires a local source weights file and a full SHA256")
        file = Path(path_text).resolve()
        if not file.is_file():
            raise ValueError("source V04 pretrained checkpoint file does not exist")
        if file_sha256(file).lower() != expected:
            raise ValueError("source V04 checkpoint SHA256 mismatch")
        return str(file), expected

    def enumerate_routes(self, generic_profiles=("faithful", "corrected"),
                         settings=None) -> list[ModelRoute]:
        if settings is None or "vision" not in settings.tasks:
            return []
        if not generic_profiles or set(generic_profiles) - {"faithful", "corrected"}:
            raise ValueError("source V04 generic profile must be faithful or corrected")
        _, source_sha = self._source()
        routes: list[ModelRoute] = []
        for model_id, config in sorted(MODEL_REGISTRY.items()):
            profiles = (None,) if not config.generic else generic_profiles
            for profile in profiles:
                label = (f"V04-source:sha{source_sha[:20]}:{model_id}"
                         + (f"@{profile}" if profile else ""))
                routes.append(ModelRoute(
                    model_id,
                    label=label,
                    fixed_grid=((self.SOURCE_WIDTH,),) if model_id == "MLP-Fixed" else None,
                    suite_group="vision",
                    generic_profile=profile,
                    init_width=self.SOURCE_WIDTH,
                    init_depth=1,
                ))
        return routes

    def fit_predict(
        self, route: ModelRoute, *, data: PreparedTaskData, split: SplitSpec,
        seed: int, settings: SearchSettings,
        checkpoint_path: str | None = None,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> dict[str, Any]:
        if data.task_family != "vision" or route.suite_group != "vision":
            raise ValueError("source V04 adapter only accepts official/synthetic vision tasks")
        weights_path, source_sha = self._source()
        expected_label = f"V04-source:sha{source_sha[:20]}:{route.model_id}"
        if route.model_id not in MODEL_REGISTRY or route.route_label != (
                expected_label + (f"@{route.generic_profile}" if route.generic_profile else "")):
            raise ValueError("source V04 route/source checksum is inconsistent with planned identity")
        if route.init_width != self.SOURCE_WIDTH or route.init_depth != 1:
            raise ValueError("source V04 must start at original 384-wide single FFN")
        if route.model_id == "MLP-Fixed" and route.fixed_grid != ((self.SOURCE_WIDTH,),):
            raise ValueError("source V04 fixed baseline must use original width 384")
        if settings.max_width < self.SOURCE_WIDTH or settings.max_neurons < self.SOURCE_WIDTH:
            raise ValueError("source V04 run capacity must accommodate original 384 neurons")
        if settings.max_epochs == 0 or settings.max_candidates == 0:
            # The pressure runner may otherwise launch an unbounded 28M-parameter
            # search by default. Requiring explicit bounds prevents wasting GPU.
            raise ValueError("source V04 coordinated jobs require finite max_epochs and max_candidates")

        def take(rows: tuple[int, ...]) -> np.ndarray:
            return np.asarray(data.X[np.asarray(rows, dtype=np.int64)], dtype=np.float32) / 255.0

        fit_x, valid_x = take(split.train_indices), take(split.validation_indices)
        fit_y = np.asarray(data.y[list(split.train_indices)])
        valid_y = np.asarray(data.y[list(split.validation_indices)])
        # Load the official test only after the native search has completed.
        real_settings = SearchSettings(**{
            **settings.as_dict(), "seed": int(seed),
            "init_width": self.SOURCE_WIDTH, "init_depth": 1,
        })
        spec = SourceConvNeXtSpec(
            input_channels=int(fit_x.shape[1]),
            image_height=int(fit_x.shape[2]),
            image_width=int(fit_x.shape[3]),
            classes=int(np.unique(fit_y).size),
            weights_path=weights_path,
            weights_sha256=source_sha,
            fixed_seed=int(seed),
        )
        search = VisualResidualSearch(
            route.model_id, spec, settings=real_settings,
            generic_profile=route.generic_profile or "faithful",
            fixed_grid=route.fixed_grid,
        )
        started = time.perf_counter()
        search.fit(
            fit_x, fit_y, validation_data=(valid_x, valid_y),
            checkpoint_path=checkpoint_path,
            progress_callback=progress_callback,
        )
        fit_elapsed = time.perf_counter() - started
        search.model_.verify_frozen()
        task = search.task
        valid_prediction = search.predict(valid_x)
        valid_probability = search.predict_proba(valid_x)
        model_out = search.model_(
            torch.as_tensor(valid_x, dtype=torch.float32, device=real_settings.device))
        validation_loss = task.validation_score(
            model_out, task.encode_targets(valid_y, device=model_out.device))
        test_x, test_y = take(split.test_indices), np.asarray(
            data.y[list(split.test_indices)])
        test_prediction = search.predict(test_x)
        test_probability = search.predict_proba(test_x)

        artifact = None
        if checkpoint_path is not None:
            artifact = str(estimator_model_path(checkpoint_path))
            CheckpointStore(artifact).save({
                "format": "visual-v04-source-convnext-inference-v1",
                "model_id": route.model_id,
                "source_weights_sha256": source_sha,
                "source_route_label": route.route_label,
                "architecture_id": "V04:torchvision-convnext-tiny-source-ffn",
                "growth_site": spec.site_id,
                "split_sha256": split.split_sha256,
                "task": task.state_dict(),
                "best_snapshot": search.model_.snapshot(
                    cursor=search.model_._width_cursor),
                "history": search.history_,
                "best_score": search.result_.score,
            }, signature=(
                str(data.content_sha256) + ":" + str(split.split_sha256)
                + ":" + str(seed) + ":" + source_sha
            ))
        return {
            "fit_seconds": fit_elapsed,
            "history_entries": len(search.history_),
            "selected_architecture": list(search.model_.hidden_widths),
            "validation_selection_score": float(validation_loss),
            "validation_metrics": task.evaluation_metrics(
                valid_y, valid_prediction, valid_probability),
            "test_metrics": task.evaluation_metrics(
                test_y, test_prediction, test_probability),
            "task_family": "vision", "task_variant": data.task_variant,
            "split_protocol": data.split_protocol,
            "primary_metric": task.primary_metric,
            "metric_directions": task.metric_directions,
            "artifact_path": artifact,
            "growth_site": spec.site_id,
            "architecture_id": "V04:torchvision-convnext-tiny-source-ffn",
            "source_weights_sha256": source_sha,
            "frozen_sha256": search.model_.frozen_state_sha256(),
            "expansion_criterion": real_settings.expansion_criterion,
            "source_ffn_exact_at_initialization": (
                search.model_.source_ffn_exact_at_initialization),
        }
