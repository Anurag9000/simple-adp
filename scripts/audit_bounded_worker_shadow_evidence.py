"""Audit persisted best-stage shadow-BGD posterior growth evidence.

Read actual production estimator checkpoints using PyTorch weights_only=True,
validate stage geometry and observer sigma, and classify the recorded
posterior-stagnation gate. This is *best-epoch stage* evidence, not the
full per-minibatch history, uncertainty calibration or compute-matched FLOPs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from simple_adp.runner.artifacts import estimator_checkpoint_path


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_shadow_checkpoint(path, *, sha256: str) -> dict:
    path = Path(path)
    if not path.is_file() or _sha(path) != sha256:
        raise ValueError("shadow posterior checkpoint is absent or fails SHA-256 verification")
    envelope = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(envelope, dict) or not isinstance(envelope.get("payload"), dict):
        raise ValueError("missing signed estimator checkpoint payload")
    payload = envelope["payload"]
    stage = payload.get("stage_state")
    if not isinstance(stage, dict) or stage.get("stage_status") != "completed":
        raise ValueError("missing completed stage in posterior checkpoint")
    observer = stage.get("shadow_bgd_variance")
    best = stage.get("best_shadow_bgd_variance")
    if not isinstance(observer, dict) or not isinstance(best, dict):
        raise ValueError("variance-triggered worker has no persisted shadow-BGD evidence")
    if observer.get("schema") != 1 or best.get("schema") != 1:
        raise ValueError("unsupported posterior evidence schema")
    config = observer.get("config")
    if not isinstance(config, (tuple, list)) or len(config) != 5:
        raise ValueError("incompatible posterior observer configuration receipt")
    initial_std, samples, tolerance, patience, observer_seed = config
    if (not isinstance(samples, int) or samples < 1
            or not isinstance(patience, int) or patience < 1
            or not isinstance(observer_seed, int)
            or not math.isfinite(float(initial_std)) or initial_std <= 0
            or not math.isfinite(float(tolerance)) or tolerance < 0):
        raise ValueError("invalid posterior observer hyperparameter receipt")
    observations = observer.get("observations")
    streak = observer.get("streak")
    contraction = observer.get("contraction")
    if (type(observations) is not int or type(streak) is not int
            or observations < 0 or not 0 <= streak <= observations):
        raise ValueError("posterior observer evidence counters are invalid")
    if contraction is not None and (type(contraction) not in {int, float}
                                    or not math.isfinite(contraction)):
        raise ValueError("posterior contraction is not finite")
    if observations and contraction is None:
        raise ValueError("posterior evidence omitted a recorded contraction statistic")
    sigma = observer.get("sigma")
    if not isinstance(sigma, dict) or (observations > 0 and not sigma):
        raise ValueError("posterior checkpoint lacks sigma coordinates")
    total = 0
    minimum = float("inf")
    maximum = float("-inf")
    sigma_sum = 0.0
    for name, tensor in sigma.items():
        if (not isinstance(name, str) or not name
                or not isinstance(tensor, torch.Tensor)
                or not tensor.is_floating_point() or not tensor.numel()
                or not bool(torch.isfinite(tensor).all())
                or not bool((tensor > 0).all())):
            raise ValueError("invalid sigma coordinate in posterior checkpoint")
        total += tensor.numel()
        minimum = min(minimum, float(tensor.min()))
        maximum = max(maximum, float(tensor.max()))
        sigma_sum += float(tensor.to(torch.float64).sum())
    history = payload.get("search_history")
    if not isinstance(history, list) or not history:
        raise ValueError("posterior growth checkpoint has no source search history")
    gate_ready = observations >= patience and streak >= patience
    return {
        "observer_schema": 1,
        "observer_seed": observer_seed,
        "initial_sigma_std": float(initial_std),
        "mc_samples_per_successful_observation": samples,
        "minimum_sampled_posterior_forwards_from_successful_observations": (
            observations * samples
        ),
        "stagnation_tolerance": float(tolerance),
        "required_streak": patience,
        "posterior_observations_at_best_stage": observations,
        "stagnation_streak_at_best_stage": streak,
        "contraction_at_best_stage": contraction,
        "gate_ready_at_best_stage": gate_ready,
        "sigma_parameter_tensors": len(sigma),
        "sigma_parameter_coordinates": total,
        "sigma_min": None if total == 0 else minimum,
        "sigma_max": None if total == 0 else maximum,
        "sigma_mean": None if total == 0 else sigma_sum / total,
        "source_search_history_length": len(history),
        "source_search_history": [
            {key: value for key, value in row.items()
             if key in {"architecture", "score", "accepted", "stage", "profile"}}
            for row in history
        ],
        "no_growth_explanation": (
            "best-stage posterior stagnation gate not met"
            if not gate_ready and len(history) == 1 else
            "at least one candidate was evaluated or gate was ready; examine proposal trace"
        ),
        "evidence_scope": "restored best-epoch stage checkpoint, not complete posterior time series",
    }


def audit_worker_posterior(root: Path) -> dict:
    root = Path(root)
    manifest = root / "worker_pair_receipt.json"
    if not manifest.is_file():
        raise ValueError("real worker pair receipt missing")
    source = json.loads(manifest.read_text(encoding="utf-8"))
    if (source.get("schema") != "real-sklearn-planner-to-worker-pairs-v1"
            or source.get("evidence_grade") != "BOUNDED_REAL_DATA_WORKER_EXECUTED"):
        raise ValueError("source receipt is not an executed original/variance real worker proof")
    records = []
    pairs = source.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("source real worker receipt has no pairs")
    for pair in pairs:
        original = pair.get("original")
        bayesian = pair.get("bayesian")
        if not isinstance(original, dict) or not isinstance(bayesian, dict):
            raise ValueError("source receipt omitted an original/Bayesian worker")
        if (original.get("arm") != "original" or bayesian.get("arm") != "bayesian"
                or original.get("seed") != bayesian.get("seed")
                or original.get("split_sha256") != bayesian.get("split_sha256")
                or original.get("task") != bayesian.get("task")):
            raise ValueError("source real worker pair attribution is invalid")
        original_checkpoint = estimator_checkpoint_path(root / "results" / original["run_id"])
        bayesian_checkpoint = estimator_checkpoint_path(root / "results" / bayesian["run_id"])
        for arm, location in ((original, original_checkpoint), (bayesian, bayesian_checkpoint)):
            if not isinstance(arm.get("checkpoint_sha256"), str) or _sha(location) != arm["checkpoint_sha256"]:
                raise ValueError("source real worker stage checkpoint mismatches recorded SHA-256")
        original_payload = torch.load(
            original_checkpoint, map_location="cpu", weights_only=True
        ).get("payload", {})
        if not isinstance(original_payload, dict) or not isinstance(original_payload.get("stage_state"), dict):
            raise ValueError("original source checkpoint lacks a completed stage")
        if original_payload["stage_state"].get("shadow_bgd_variance") is not None:
            raise ValueError("original loss-triggered arm unexpectedly owns a shadow posterior")
        observer = inspect_shadow_checkpoint(
            bayesian_checkpoint, sha256=bayesian["checkpoint_sha256"]
        )
        if observer["observer_seed"] != bayesian["seed"]:
            raise ValueError("posterior observer seed differs from Bayesian job seed")
        records.append({
            "pair_id": pair["pair_id"],
            "real_task": original["task"],
            "seed": original["seed"],
            "original_model_architecture": original.get("selected_architecture"),
            "bayesian_model_architecture": bayesian.get("selected_architecture"),
            "original_checkpoint_sha256": original["checkpoint_sha256"],
            "bayesian_checkpoint_sha256": bayesian["checkpoint_sha256"],
            "posterior": observer,
        })
    return {
        "schema": "real-worker-shadow-bgd-best-stage-evidence-v1",
        "source_revision": source.get("source_commit"),
        "source_manifest_sha256": source["manifest_sha256"],
        "pairs_checked": len(records),
        "evidence_grade": "BEST_STAGE_POSTERIOR_RECEIPTS_VERIFIED",
        "pairs": sorted(records, key=lambda x: x["real_task"]),
        "scope_and_limits": [
            "Best-epoch stage snapshot, not every minibatch or earlier transient trigger",
            "Monte Carlo forward-call lower bound, not whole-model training FLOPs or energy",
            "No calibration or sample-efficiency/superiority claims",
        ],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = audit_worker_posterior(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
