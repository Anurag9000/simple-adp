"""V02 dedicated composite-image architecture search with exact stage restart.

This is intentionally limited to native dedicated controllers and generic phase-search on full-image
composites, with native profile semantics and signed per-batch recovery.
"""
from __future__ import annotations

import copy
import hashlib
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from .checkpoint import CheckpointStore, capture_rng_state, restore_rng_state, stable_signature
from .config import SearchSettings
from .growth import (
    can_add_depth, can_grow_width_all_layers, can_grow_width_round_robin,
    total_neurons,
)
from .models import rebuild_mlp
from .registry import MODEL_REGISTRY, get_model_config
from .results import SearchResult
from .tasks import ClassificationTask
from .training import StageTrainer
from .visual_accounting import visual_parameter_accounting
from .visual_memory import (
    visual_model_tensor_payload,
    prospective_tabular_growth_widths,
    forecast_visual_growth_payload_bytes,
)
from .visual_inner import ResidualAdapterSpec, VisualResidualAdapter
from .visual_native_ffn import NativeFFNSpec, VisualNativeFFN, build_visual_composite, restore_visual_composite
from .visual_mixer import MixerSpec
from .visual_multistage import MultistageSpec
from .visual_adapter_bank import BankSpec
from .visual_multistage_schedule import ScheduledMultistageSpec, VisualMultistageScheduled, SITES
from .visual_source_convnext import SourceConvNeXtSpec, VisualSourceConvNeXtFFN
from .visual_source_vit import SourceViTSpec, VisualSourceViTFFN


SUPPORTED_V02_METHODS = frozenset(MODEL_REGISTRY)


def _shadow_best_stage_evidence(stage: Any) -> dict[str, object]:
    """Serializable best-epoch gate evidence, never a replacement for posterior state.

    A legacy stage without the observer is visibly INCOMPLETE, not silently
    represented as a zero-contraction event. Actual growth still uses the
    untouched StageResult.variance_stagnated decision.
    """
    state = getattr(stage, "checkpoint_state", None)
    observer = state.get("shadow_bgd_variance") if isinstance(state, dict) else None
    if not isinstance(observer, dict):
        return {"status": "UNAVAILABLE", "gate_ready": bool(stage.variance_stagnated)}
    config = observer.get("config")
    if not isinstance(config, (tuple, list)) or len(config) != 5:
        raise ValueError("visual shadow diagnostic lacks observer hyperparameters")
    _, mc_samples, tolerance, patience, seed = config
    observations = observer.get("observations")
    streak = observer.get("streak")
    contraction = observer.get("contraction")
    if (type(observations) is not int or type(streak) is not int
            or not 0 <= streak <= observations
            or type(mc_samples) is not int or mc_samples < 1
            or type(patience) is not int or patience < 1):
        raise ValueError("visual shadow diagnostic has invalid counters")
    if (contraction is not None and (
        type(contraction) not in {int, float}
        or not np.isfinite(contraction)
    )):
        raise ValueError("visual shadow diagnostic has nonfinite contraction")
    gate = bool(observations >= patience and streak >= patience)
    if gate != bool(stage.variance_stagnated):
        raise ValueError("visual shadow diagnostic disagrees with actual stage growth gate")
    return {
        "status": "BEST_STAGE_OBSERVED",
        "observation_count": observations,
        "mc_samples_per_observation": mc_samples,
        "lower_bound_shadow_forward_passes": observations * mc_samples,
        "contraction": contraction,
        "tolerance": float(tolerance),
        "streak": streak,
        "required_streak": patience,
        "observer_seed": seed,
        "gate_ready": gate,
        "scope": "selected best-epoch stage only, not a full minibatch trace",
    }


def _hash_array(digest: Any, item: Any) -> None:
    array = np.ascontiguousarray(item.detach().cpu().numpy() if torch.is_tensor(item) else np.asarray(item))
    if array.dtype.hasobject:
        raise TypeError("V02 requires numeric input arrays and labels")
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())


def _restore(snapshot: dict[str, object], device: str) -> VisualResidualAdapter:
    model = restore_visual_composite(snapshot).to(device)
    model.verify_frozen()
    return model


class VisualResidualSearch:
    """Run native dedicated method policies over whole-image V02 candidates.

    Saves signed, atomic checkpoints *before the stage* and after every batch
    through the existing StageTrainer state. The checkpoint carries current,
    selected-best, architecture counters and exact RNG/batch/optimizer state.
    Only complete matching training data, configuration and site can resume.
    """

    def __init__(
        self,
        model_id: str,
        spec: ResidualAdapterSpec | NativeFFNSpec | MixerSpec | MultistageSpec | BankSpec | ScheduledMultistageSpec | SourceConvNeXtSpec | SourceViTSpec,
        *,
        settings: SearchSettings | None = None,
        fixed_grid: Sequence[Sequence[int]] | None = None,
        generic_profile: str = "faithful",
        max_model_tensor_payload_bytes: int | None = None,
        track_training_forward_macs: bool = False,
    ) -> None:
        config = get_model_config(model_id)
        if config.id not in SUPPORTED_V02_METHODS:
            raise NotImplementedError(
                f"V02 {model_id} needs a composite-controller implementation; "
                "do not report generic methods as supported"
            )
        if generic_profile not in {"faithful", "corrected"}:
            raise ValueError("generic_profile must be faithful or corrected")
        self.generic_profile = generic_profile
        if max_model_tensor_payload_bytes is not None and (
            isinstance(max_model_tensor_payload_bytes, bool)
            or not isinstance(max_model_tensor_payload_bytes, int)
            or max_model_tensor_payload_bytes <= 0
        ):
            raise ValueError("max_model_tensor_payload_bytes must be a positive integer or None")
        self.max_model_tensor_payload_bytes = max_model_tensor_payload_bytes
        if type(track_training_forward_macs) is not bool:
            raise ValueError("track_training_forward_macs must be a boolean")
        self.track_training_forward_macs = track_training_forward_macs
        self.config = config
        self.spec = spec
        self.settings = settings or SearchSettings()
        self.fixed_grid = (tuple(tuple(int(x) for x in row) for row in fixed_grid)
                           if fixed_grid is not None else None)
        if config.controller == "fixed":
            if not self.fixed_grid or any(not widths or min(widths) <= 0 for widths in self.fixed_grid):
                raise ValueError("V02 MLP-Fixed requires positive nonempty fixed_grid")
        elif self.fixed_grid is not None:
            raise ValueError("fixed_grid is allowed only for MLP-Fixed")
        self.model_: VisualResidualAdapter | None = None
        self.history_: list[dict[str, object]] = []
        self.search_state_: dict[str, object] = {}
        self.task = ClassificationTask()
        self.result_: SearchResult | None = None

    def _signature(self, train_x: Any, train_y: Any, valid_x: Any, valid_y: Any) -> str:
        digest = hashlib.sha256()
        for item in (train_x, train_y, valid_x, valid_y):
            _hash_array(digest, item)
        return stable_signature({
            "version": ("v02-dedicated-1" if isinstance(self.spec, ResidualAdapterSpec)
                        else "visual-native-ffn-search-1" if isinstance(self.spec, NativeFFNSpec)
                        else self._search_kind()),
            "model": self.config.signature_dict(),
            "spec": asdict(self.spec),
            "settings": self.settings.as_dict(),
            "fixed_grid": self.fixed_grid,
            **({"generic_profile": self.generic_profile} if self.config.generic else {}),
            "task": self.task.state_dict(),
            "raw_data_sha256": digest.hexdigest(),
            # Keep pre-existing no-limit signatures unchanged. A configured
            # admission policy must invalidate a different-cap checkpoint.
            **({"visual_tensor_payload_cap_bytes": self.max_model_tensor_payload_bytes}
               if self.max_model_tensor_payload_bytes is not None else {}),
            **({"track_training_forward_macs": True} if self.track_training_forward_macs else {}),
        })

    def _search_kind(self) -> str:
        if isinstance(self.spec, ResidualAdapterSpec):
            return "visual-residual-search-v02"
        if isinstance(self.spec, MultistageSpec):
            return "visual-multistage-search-v1"
        if isinstance(self.spec, SourceConvNeXtSpec):
            return "visual-source-convnext-v04-search-v1"
        if isinstance(self.spec, SourceViTSpec):
            return "visual-source-vit-v03-search-v1"
        if isinstance(self.spec, ScheduledMultistageSpec):
            return "visual-multistage-rr-sites-search-v1"
        if isinstance(self.spec, BankSpec):
            return "visual-adapter-bank-search-v1"
        if isinstance(self.spec, MixerSpec):
            return ("visual-mixer-both-search-v1" if self.spec.site_id == "both:block0"
                    else "visual-mixer-search-v1")
        return "visual-native-ffn-search-v1"

    def _new_state(self) -> dict[str, Any]:
        random.seed(self.settings.seed)
        np.random.seed(self.settings.seed)
        torch.manual_seed(self.settings.seed)
        first = (self.fixed_grid[0] if self.fixed_grid else
                 (self.settings.init_width,) * self.settings.init_depth)
        if isinstance(self.spec, SourceConvNeXtSpec) and tuple(first) != (384,):
            raise ValueError("pretrained V04 inner FFN requires initial hidden width exactly 384")
        if isinstance(self.spec, SourceViTSpec) and tuple(first) != (3072,):
            raise ValueError("source V03 ViT inner FFN requires initial hidden width exactly 3072")
        model = build_visual_composite(self.spec, tuple(first)).to(self.settings.device)
        self._check_model_payload(model, initial=True)
        return {
            "kind": self._search_kind(),
            "status": "running", "phase": "initial" if self.config.controller != "fixed" else "fixed",
            "current": model.snapshot(cursor=model._width_cursor),
            "best": None, "best_score": float("inf"),
            "history": [], "failed": 0, "proposals": 0, "fixed_index": 0,
            "gate_ready": True, "in_stage": True,
            "current_score": float("inf"),
            "previous_candidate_score": float("inf"),
            "phase_index": 0, "failure_counts": {},
            "stage_state": None, "rng_state": capture_rng_state(),
            "memory_admission_rejections": [],
            "trained_model_memory_trace": [],
            "trained_model_memory_trace_complete": True,
            **({
                "trained_forward_mac_trace": [],
                "trained_forward_mac_trace_complete": True,
            } if self.track_training_forward_macs else {}),
            "training_budget": {
                "optimizer_steps": 0, "trained_epochs": 0,
                "trained_stages": 0, "warmup_stages": 0,
                "complete": True,
            },
        }

    def _check_model_payload(
        self, model: VisualResidualAdapter, *, initial: bool = False,
    ) -> int:
        used = int(visual_model_tensor_payload(model)["total_model_tensor_payload_bytes"])
        cap = self.max_model_tensor_payload_bytes
        if initial and cap is not None and used > cap:
            raise ValueError(
                f"initial visual model logical tensor payload {used} bytes "
                f"exceeds max_model_tensor_payload_bytes={cap}; "
                "the limit does not include activations, optimizer state or peak memory"
            )
        return used

    def _preflight_growth_payload(
        self, current: VisualResidualAdapter, state: dict[str, Any], *,
        proposed_widths: tuple[int, ...] | None,
        other_widths: tuple[int, ...] | None = None,
    ) -> bool:
        """Reject predictable over-cap MLP growth *before deep-copying* it.

        The local-MLP shape calculation is exact for native TabularMLPs
        with supported dtype and buffer layouts. Unsupported layouts
        defer to the existing postconstruction guard. Frozen source
        models must still be materialized to know their tensor payload.
        """
        if self.max_model_tensor_payload_bytes is None or proposed_widths is None:
            return True
        predicted = forecast_visual_growth_payload_bytes(
            current, proposed_widths, other_hidden_widths=other_widths,
        )
        if predicted is None or predicted <= self.max_model_tensor_payload_bytes:
            return True
        state.setdefault("memory_admission_rejections", []).append({
            "phase": str(state["phase"]),
            "architecture": list(proposed_widths),
            "site": (
                "adapter:" + current._active
                if isinstance(self.spec, ScheduledMultistageSpec)
                else self.spec.site_id
            ),
            "logical_model_tensor_payload_bytes": int(predicted),
            "cap_bytes": self.max_model_tensor_payload_bytes,
            "reason": "model-parameter-buffer-payload-limit",
            "before_candidate_construction": True,
            "prediction_kind": "native-TabularMLP-geometry-v1",
        })
        return False

    def _preflight_operator(
        self, current: VisualResidualAdapter, state: dict[str, Any], *,
        operator: str, increment: int = 1, seed_width: int = 1,
    ) -> bool:
        if self.max_model_tensor_payload_bytes is None:
            return True
        kwargs = {
            "operator": operator, "increment": increment,
            "max_width": self.settings.max_width,
            "max_depth": self.settings.max_depth,
            "seed_width": seed_width,
        }
        proposed = prospective_tabular_growth_widths(current.local_mlp, **kwargs)
        other = None
        if isinstance(self.spec, MixerSpec) and self.spec.site_id == "both:block0":
            other = prospective_tabular_growth_widths(current.other_mlp, **kwargs)
            if proposed is None or other is None:
                return True  # Source operation will reject; do not create a fake memory denial.
        return self._preflight_growth_payload(
            current, state, proposed_widths=proposed, other_widths=other,
        )

    def _persist(self, store: CheckpointStore | None, signature: str, state: dict[str, Any],
                 progress_callback: Callable[[dict[str, object]], None] | None) -> None:
        if store is not None:
            store.save(copy.deepcopy(state), signature=signature)
        if progress_callback is not None:
            progress_callback({
                "configuration_id": self.config.id,
                "growth_site": (
                    "adapter:" + state["current"]["active_site"]
                    if isinstance(self.spec, ScheduledMultistageSpec)
                    else self.spec.site_id
                ),
                "phase": state["phase"],
                "stage_status": (state["stage_state"] or {}).get("stage_status", "idle"),
                "architecture": list(state["current"]["hidden_widths"]),
                "proposals": state["proposals"],
                "best_score": state["best_score"],
                "epoch": (state["stage_state"] or {}).get("epoch", 0),
                "batch_cursor": (state["stage_state"] or {}).get("batch_cursor", 0),
                "batches_done": (state["stage_state"] or {}).get("batches_done", 0),
            })

    def _propose(self, state: dict[str, Any]) -> VisualResidualAdapter | None:
        current = _restore(state["current"], self.settings.device)
        if isinstance(self.spec, ScheduledMultistageSpec):
            current.activate(SITES[int(state["proposals"]) % len(SITES)])
        local = current.local_mlp
        s = self.settings
        site = self.spec.site_id
        if self.config.generic:
            if state["phase"] == "width":
                operator = ("all_layers" if self.config.width_strategy == "all_layers"
                            else "round_robin")
                if not self._preflight_operator(
                    current, state, operator=operator, increment=s.ex_k_width,
                ):
                    return None
                candidate = current.grow_width(
                    site_id=site, operator=operator,
                    increment=s.ex_k_width, max_width=s.max_width,
                )
            elif state["phase"] == "depth":
                depth_operator = (
                    "Dcur" if self.config.depth_seed == "current" else "D1"
                )
                if not self._preflight_operator(
                    current, state, operator=depth_operator,
                    seed_width=s.min_new_layer_width,
                ):
                    return None
                candidate = current.grow_depth(
                    site_id=site,
                    operator=depth_operator,
                    max_depth=s.max_depth, seed_width=s.min_new_layer_width,
                )
            else:
                raise ValueError("unsupported generic phase")
            if (candidate is not None and self.generic_profile == "corrected"
                    and total_neurons(candidate.local_mlp) > s.max_neurons):
                return None
            return candidate
        if state["phase"] == "width":
            if self.config.controller == "lgnd" or self.config.width_strategy == "all_layers":
                eligible = can_grow_width_all_layers(local, s.ex_k_width, s.max_width, s.max_neurons)
                operation = "all_layers"
            else:
                eligible = can_grow_width_round_robin(local, s.ex_k_width, s.max_width, s.max_neurons)
                operation = "den_round_robin"
            if not eligible:
                return None
            if not self._preflight_operator(
                current, state, operator=operation, increment=s.ex_k_width,
            ):
                return None
            return current.grow_width(site_id=site, operator=operation,
                                      increment=s.ex_k_width, max_width=s.max_width)
        if state["phase"] == "depth":
            width = current.hidden_widths[-1]
            seed = 1 if self.config.progressive_depth else width
            if not can_add_depth(local, s.max_depth, s.max_neurons, seed):
                return None
            if isinstance(current.spec, MixerSpec) and current.spec.site_id == "both:block0":
                if total_neurons(local) + total_neurons(current.other_mlp) + 2 * seed > s.max_neurons:
                    return None
            if not self._preflight_operator(
                current, state,
                operator="D1" if self.config.progressive_depth else "Dcur",
                seed_width=1 if self.config.progressive_depth else width,
            ):
                return None
            if self.config.progressive_depth:
                candidate = current.grow_depth(site_id=site, operator="D1",
                                               max_depth=s.max_depth, seed_width=1)
                if candidate is not None and width > 1:
                    # One *logical* depth proposal has optimization-only stages
                    # at widths 1..(current width - 1), exactly as dedicated DEN.
                    state["depth_target"] = width
                    state["phase"] = "depth_warmup"
                return candidate
            return current.grow_depth(site_id=site, operator="Dcur",
                                      max_depth=s.max_depth)
        if state["phase"] == "depth_warmup":
            target = int(state["depth_target"])
            new_width = current.hidden_widths[-1] + 1
            if new_width > target or new_width > s.max_width or total_neurons(local) + 1 > s.max_neurons:
                return None
            projected_widths = (*current.hidden_widths[:-1], new_width)
            if not self._preflight_growth_payload(
                current, state, proposed_widths=projected_widths,
                other_widths=(
                    (*current.other_mlp.hidden_widths[:-1], new_width)
                    if isinstance(self.spec, MixerSpec)
                    and self.spec.site_id == "both:block0" else None
                ),
            ):
                return None
            candidate = copy.deepcopy(current)
            replacement = rebuild_mlp(
                current.local_mlp, [*current.hidden_widths[:-1], new_width],
                cursor=current._width_cursor,
            )
            if isinstance(candidate, VisualMultistageScheduled):
                candidate.site_mlps[candidate._active] = replacement
            elif isinstance(candidate, VisualSourceConvNeXtFFN):
                candidate.source.features[1][0].block[3] = replacement
            elif isinstance(candidate, VisualSourceViTFFN):
                candidate.source.encoder.layers[0].mlp[0] = replacement
            else:
                candidate.local_mlp = replacement
            if isinstance(current.spec, MixerSpec) and current.spec.site_id == "both:block0":
                candidate.other_mlp = rebuild_mlp(
                    current.other_mlp,
                    [*current.other_mlp.hidden_widths[:-1], new_width],
                    cursor=int(getattr(current.other_mlp, "_width_cursor", 0)),
                )
            if hasattr(candidate, "_gelu_hidden"):
                candidate._gelu_hidden()
            candidate.train(current.training)
            candidate.verify_frozen()
            if new_width == target:
                state["phase"] = "depth"
                state.pop("depth_target", None)
            return candidate
        if state["phase"] == "fixed":
            index = state["fixed_index"]
            if index >= len(self.fixed_grid):
                return None
            proposed = tuple(self.fixed_grid[index])
            if not self._preflight_growth_payload(
                current, state, proposed_widths=proposed,
                other_widths=(
                    proposed if isinstance(self.spec, MixerSpec)
                    and self.spec.site_id == "both:block0" else None
                ),
            ):
                return None
            return build_visual_composite(self.spec, self.fixed_grid[index]).to(s.device)
        raise ValueError(f"unknown V02 dedicated phase {state['phase']!r}")

    def _accept(self, state: dict[str, Any], score: float) -> bool:
        margin = (0.0 if self.config.controller == "lgnd" else
                  self.settings.delta_depth if state["phase"] == "depth" else
                  self.settings.delta_width if state["phase"] == "width" else 0.0)
        return score < float(state["best_score"]) - margin

    def _finish_stage(self, state: dict[str, Any], stage: Any) -> None:
        kind = state["phase"]
        current = _restore(stage.best_snapshot, self.settings.device)
        score = float(stage.best_score)
        model_bytes = self._check_model_payload(current, initial=True)
        memory_trace = state.setdefault("trained_model_memory_trace", [])
        if "trained_model_memory_trace_complete" not in state:
            # Resume from a legacy checkpoint: prior stage memory is unknown.
            state["trained_model_memory_trace_complete"] = False
        memory_trace.append({
            "stage": kind,
            "architecture": list(current.hidden_widths),
            "site": (
                "adapter:" + current._active
                if isinstance(self.spec, ScheduledMultistageSpec)
                else self.spec.site_id
            ),
            "logical_model_tensor_payload_bytes": model_bytes,
            "optimizer_steps": int(stage.batches),
        })
        budget = state.setdefault("training_budget", {
            # Old exact-resume payloads have no prior-stage budget fields.
            # Never incorrectly call their partially recovered totals complete.
            "optimizer_steps": 0, "trained_epochs": 0,
            "trained_stages": 0, "warmup_stages": 0,
            "complete": False,
        })
        budget["optimizer_steps"] += int(stage.batches)
        budget["trained_epochs"] += int(stage.epochs)
        budget["trained_stages"] += 1
        if self.settings.expansion_criterion == "variance_stagnation":
            trace = state.setdefault("posterior_gate_trace", [])
            entry = _shadow_best_stage_evidence(stage)
            trace.append({
                "stage": kind,
                "architecture": list(current.hidden_widths),
                **entry,
            })
            state["posterior_gate_trace_complete"] = bool(
                state.get("posterior_gate_trace_complete", True)
                and entry["status"] == "BEST_STAGE_OBSERVED"
                and len(trace) == budget["trained_stages"]
            )
        if kind == "depth_warmup":
            budget["warmup_stages"] += 1
        if self.track_training_forward_macs:
            from .visual_train_compute import summarize_visual_training_forward_trace
            measured = stage.training_forward_macs
            if not isinstance(measured, dict):
                raise RuntimeError("trained visual stage has no observed forward MAC evidence")
            if bool(measured.get("complete")) and (
                measured.get("recorded_training_forward_batches") != int(stage.batches)
            ):
                raise RuntimeError("visual observed forward batch count differs from signed stage optimizer steps")
            trace = state.setdefault("trained_forward_mac_trace", [])
            if "trained_forward_mac_trace_complete" not in state:
                state["trained_forward_mac_trace_complete"] = False
            trace.append({
                "stage": kind,
                "architecture": list(current.hidden_widths),
                "site": (
                    "adapter:" + current._active
                    if isinstance(self.spec, ScheduledMultistageSpec)
                    else self.spec.site_id
                ),
                **measured,
            })
            state["trained_forward_mac_trace_complete"] = bool(
                state["trained_forward_mac_trace_complete"] and measured.get("complete")
            )
            state["training_forward_compute"] = summarize_visual_training_forward_trace(
                trace,
                training_stages=int(budget["trained_stages"]),
                complete=bool(state["trained_forward_mac_trace_complete"]),
            )
        if kind == "depth_warmup":
            # Dedicated D1 intermediate stages optimize only. They must not
            # compete in architecture selection or reset failure patience.
            state["current"] = current.snapshot(cursor=current._width_cursor)
            state["stage_state"] = None
            state["in_stage"] = False
            state["rng_state"] = capture_rng_state()
            current.verify_frozen()
            return
        if self.config.generic and kind != "initial":
            margin = self.settings.delta_width if kind == "width" else self.settings.delta_depth
            if self.generic_profile == "faithful":
                # Source faithful profile first updates stage-best=min(old,new),
                # then checks new < stage-best - margin, which cannot hold.
                accepted = score < float(state["previous_candidate_score"])
                if accepted:
                    state["best"] = current.snapshot(cursor=current._width_cursor)
                state["best_score"] = score  # Source faithfully overwrites score.
                state["current_score"] = score
                state["previous_candidate_score"] = score
                state["failed"] = int(state["failed"]) + 1
            else:
                improved_phase = score < float(state["current_score"]) - margin
                accepted = score < float(state["best_score"]) - margin
                if accepted:
                    state["best_score"] = score
                    state["best"] = current.snapshot(cursor=current._width_cursor)
                if improved_phase:
                    state["current_score"] = score
                    state["failed"] = 0
                else:
                    state["failed"] = int(state["failed"]) + 1
                state["previous_candidate_score"] = score
            state["failure_counts"][kind] = state["failed"]
        else:
            accepted = self._accept(state, score)
            if accepted:
                state["best_score"] = score
                state["best"] = current.snapshot(cursor=current._width_cursor)
                state["failed"] = 0
            else:
                state["failed"] = int(state["failed"]) + 1
        parameter_counts = visual_parameter_accounting(
            current, current.growth_targets,
        )
        state["history"].append({
            "stage": kind,
            "architecture": list(current.hidden_widths),
            "score": score, "accepted": bool(accepted),
            "trained_epochs": int(stage.epochs),
            "optimizer_steps": int(stage.batches),
            "candidate_total_parameters": parameter_counts["total_parameters"],
            "candidate_eligible_mlp_parameters": parameter_counts["eligible_mlp_parameters"],
            "candidate_model_tensor_payload_bytes": model_bytes,
            "site": ("adapter:" + current._active
                     if isinstance(self.spec, ScheduledMultistageSpec)
                     else self.spec.site_id),
            "criterion": self.settings.expansion_criterion,
            **({"profile": self.generic_profile} if self.config.generic else {}),
        })
        state["current"] = current.snapshot(cursor=current._width_cursor)
        state["gate_ready"] = bool(stage.variance_stagnated)
        state["stage_state"] = None
        state["in_stage"] = False
        state["rng_state"] = capture_rng_state()
        if kind == "initial":
            state["failed"] = 0
            state["current_score"] = score
            state["previous_candidate_score"] = score
            state["phase"] = ("width" if self.config.order in {"width", "width_then_depth"}
                              else "depth")
        elif kind == "fixed":
            state["fixed_index"] = int(state["fixed_index"]) + 1
        current.verify_frozen()

    def _advance_phase(self, state: dict[str, Any]) -> bool:
        if not self.config.generic or int(state["phase_index"]) != 0:
            return False
        state["phase_index"] = 1
        state["phase"] = ("depth" if self.config.order == "width_then_depth"
                          else "width")
        state["failed"] = 0
        state["rng_state"] = capture_rng_state()
        return True

    def fit(self, train_images: Any, train_labels: Any, *, validation_data: tuple[Any, Any],
            checkpoint_path: str | Path | None = None,
            progress_callback: Callable[[dict[str, object]], None] | None = None) -> "VisualResidualSearch":
        if not isinstance(validation_data, tuple) or len(validation_data) != 2:
            raise ValueError("V02 requires explicit (images, labels) validation_data")
        valid_images, valid_labels = validation_data
        self.task.fit_targets(train_labels)
        if self.task.output_dim != self.spec.classes:
            raise ValueError("V02 classifier output must equal the number of training classes")
        signature = self._signature(train_images, train_labels, valid_images, valid_labels)
        dev = torch.device(self.settings.device)
        x_train = torch.as_tensor(train_images, dtype=torch.float32, device=dev)
        x_valid = torch.as_tensor(valid_images, dtype=torch.float32, device=dev)
        y_train = self.task.encode_targets(train_labels, device=dev)
        y_valid = self.task.encode_targets(valid_labels, device=dev)
        if x_train.ndim != 4 or x_valid.ndim != 4 or len(x_train) == 0 or len(x_valid) == 0:
            raise ValueError("V02 requires nonempty NCHW train/validation images")
        if not torch.isfinite(x_train).all() or not torch.isfinite(x_valid).all():
            raise ValueError("V02 images must be finite")
        store = CheckpointStore(checkpoint_path) if checkpoint_path is not None else None
        state = (store.load(signature=signature, map_location="cpu")
                 if store and store.path.exists() else self._new_state())
        if state.get("kind") != self._search_kind():
            raise ValueError("exact-resume failure: unrecognized composite controller state")
        if state["status"] == "completed":
            selected = _restore(state["best"], self.settings.device)
            self.model_, self.history_, self.search_state_ = selected, list(state["history"]), dict(state)
            self.result_ = SearchResult(selected, float(state["best_score"]), self.history_, dict(state))
            return self

        while True:
            if state["in_stage"]:
                model = _restore(state["current"], self.settings.device)
                self._check_model_payload(model, initial=True)
                if state["stage_state"] is None:
                    restore_rng_state(state["rng_state"])
                def persist_stage(stage_state: dict[str, object]) -> None:
                    state["stage_state"] = copy.deepcopy(stage_state)
                    self._persist(store, signature, state, progress_callback)
                trainer = StageTrainer(
                    self.task, self.settings,
                    optimizer_kind="adamw" if self.config.generic else "adam",
                    use_penalties=self.config.generic,
                    checkpoint_callback=persist_stage,
                    track_visual_training_forward_macs=self.track_training_forward_macs,
                )
                stage = trainer.train_stage(
                    model, x_train, y_train, x_valid, y_valid,
                    resume_state=state["stage_state"],
                )
                self._finish_stage(state, stage)
                self._persist(store, signature, state, progress_callback)
                continue

            if state["phase"] == "fixed":
                exhausted = int(state["fixed_index"]) >= len(self.fixed_grid)
            elif state["phase"] == "width":
                exhausted = (self.settings.expansion_criterion == "loss"
                             and (int(state["failed"]) >=
                                  (1 if self.config.controller == "lgnd" else self.settings.patience_width_exp)))
            else:
                exhausted = (self.settings.expansion_criterion == "loss"
                             and int(state["failed"]) >= self.settings.patience_depth_exp)
            warmup = state["phase"] == "depth_warmup"
            capped = (not warmup and self.settings.max_candidates > 0
                      and int(state["proposals"]) >= self.settings.max_candidates)
            gated = (not warmup and self.settings.expansion_criterion == "variance_stagnation"
                     and not state["gate_ready"])
            if (not warmup and exhausted) or capped or gated:
                if exhausted and not capped and not gated and self._advance_phase(state):
                    self._persist(store, signature, state, progress_callback)
                    continue
                break
            candidate = self._propose(state)
            if (candidate is not None and isinstance(self.spec, MixerSpec)
                    and self.spec.site_id == "both:block0"
                    and (total_neurons(candidate.local_mlp)
                         + total_neurons(candidate.other_mlp)) > self.settings.max_neurons):
                candidate = None
            if (candidate is not None and isinstance(self.spec, ScheduledMultistageSpec)
                    and candidate.total_local_neurons() > self.settings.max_neurons):
                candidate = None
            if candidate is not None and self.max_model_tensor_payload_bytes is not None:
                used = self._check_model_payload(candidate)
                if used > self.max_model_tensor_payload_bytes:
                    state.setdefault("memory_admission_rejections", []).append({
                        "phase": str(state["phase"]),
                        "architecture": list(candidate.hidden_widths),
                        "site": (
                            "adapter:" + candidate._active
                            if isinstance(self.spec, ScheduledMultistageSpec)
                            else self.spec.site_id
                        ),
                        "logical_model_tensor_payload_bytes": used,
                        "cap_bytes": self.max_model_tensor_payload_bytes,
                        "reason": "model-parameter-buffer-payload-limit",
                    })
                    candidate = None
            if candidate is None:
                if self._advance_phase(state):
                    self._persist(store, signature, state, progress_callback)
                    continue
                break
            if not warmup:
                state["proposals"] = int(state["proposals"]) + 1
            state["current"] = candidate.snapshot(cursor=candidate._width_cursor)
            state["in_stage"] = True
            state["stage_state"] = None
            state["rng_state"] = capture_rng_state()
            self._persist(store, signature, state, progress_callback)

        if state["best"] is None:
            raise ValueError("V02 search produced no trained candidate")
        state["status"] = "completed"
        state["phase"] = "completed"
        self._persist(store, signature, state, progress_callback)
        best = _restore(state["best"], self.settings.device)
        self.model_ = best
        self.history_ = list(state["history"])
        self.search_state_ = dict(state)
        self.result_ = SearchResult(best, float(state["best_score"]), self.history_, dict(state))
        return self

    def predict(self, images: Any) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("fit before predict")
        x = torch.as_tensor(images, dtype=torch.float32, device=self.settings.device)
        self.model_.eval()
        with torch.no_grad():
            return self.task.predictions(self.model_(x))

    def predict_proba(self, images: Any) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("fit before predict")
        x = torch.as_tensor(images, dtype=torch.float32, device=self.settings.device)
        self.model_.eval()
        with torch.no_grad():
            return self.task.predict_proba(self.model_(x))
