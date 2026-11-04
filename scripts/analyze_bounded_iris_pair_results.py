"""Validate bounded real Iris paired receipts and analyze seed-level differences.

Only successfully hash-verified original and Bayesian inference artifacts
enter the analysis. Report paired uncertainty across model seeds conditional
on one fixed dataset/split; this is NOT broad cross-dataset inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

METRIC_DIRECTIONS = {
    "accuracy": "higher",
    "balanced_accuracy": "higher",
    "macro_f1": "higher",
    "log_loss": "lower",
}
EXPECTED_SCHEMA = "bounded-iris-original-vs-shadow-bgd-v1"
EXPECTED_DIFF = {"expansion_criterion": ["loss", "variance_stagnation"]}


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_receipt(path):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != EXPECTED_SCHEMA or payload.get("scientific_settings_diff") != EXPECTED_DIFF:
        raise ValueError(f"not a matched original/variance Iris pair: {path}")
    if payload.get("evidence_grade") != "BOUNDED_REAL_DATA_PASSED":
        raise ValueError(f"unverified real-data execution status: {path}")
    if payload.get("dataset_source") != "sklearn.datasets.load_iris":
        raise ValueError(f"dataset source differs: {path}")
    if payload.get("split_sizes") != {"train": 90, "validation": 30, "test": 30}:
        raise ValueError(f"unexpected train/validation/test split sizes: {path}")
    if not isinstance(payload.get("seed"), int):
        raise ValueError(f"missing integer model seed: {path}")
    if payload.get("source_commit") in (None, "UNAVAILABLE"):
        raise ValueError(f"run has no source-commit identity: {path}")
    if (not isinstance(payload.get("arms"), dict)
            or set(payload["arms"]) != {"original", "bayesian"}):
        raise ValueError(f"missing original/Bayesian output arms: {path}")
    split = path.parent / "iris_split.npz"
    if not split.is_file() or _hash(split) != payload.get("split_archive_sha256"):
        raise ValueError(f"source split artifact missing or tampered: {path}")
    for arm_name, trigger in (("original", "loss"), ("bayesian", "variance_stagnation")):
        arm = payload["arms"][arm_name]
        if arm.get("growth_trigger") != trigger:
            raise ValueError(f"swapped or missing growth criterion: {path}")
        filename = path.parent / arm_name / "estimator.model.pt"
        if not filename.is_file() or _hash(filename) != arm.get("model_artifact_sha256"):
            raise ValueError(f"{arm_name} model artifact missing or tampered: {path}")
        if filename.stat().st_size != arm.get("model_artifact_size_bytes"):
            raise ValueError(f"{arm_name} artifact-size mismatch: {path}")
        for metric in METRIC_DIRECTIONS:
            value = arm.get("test_metrics", {}).get(metric)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"{arm_name} has invalid {metric}: {path}")
        widths = arm.get("selected_architecture")
        if not isinstance(widths, list) or not widths or any(
            type(w) is not int or w < 1 for w in widths
        ):
            raise ValueError(f"{arm_name} has invalid final architecture: {path}")
    return payload


def analyze(paths, *, bootstrap_samples=10000, bootstrap_seed=1927):
    if bootstrap_samples < 1000:
        raise ValueError("at least 1000 seeded bootstrap samples required")
    data = [validate_receipt(path) for path in paths]
    if len(data) < 2:
        raise ValueError("paired variability requires at least two distinct model seeds")
    seed_values = [record["seed"] for record in data]
    if len(set(seed_values)) != len(seed_values):
        raise ValueError("duplicate model seeds cannot be independent observations")
    ground = ("source_commit", "dataset_sha256", "split_legacy_sha256",
              "split_partition_sha256_v2", "split_archive_sha256", "route", "sklearn_version")
    anchor = data[0]
    for record in data[1:]:
        for key in ground:
            if record.get(key) != anchor.get(key):
                raise ValueError(f"model seeds do not share {key}")
    paired = []
    for record in sorted(data, key=lambda item: item["seed"]):
        arm = record["arms"]
        paired.append({
            "seed": record["seed"],
            "original_architecture": arm["original"]["selected_architecture"],
            "bayesian_architecture": arm["bayesian"]["selected_architecture"],
            "metrics": {
                key: {
                    "original": float(arm["original"]["test_metrics"][key]),
                    "bayesian": float(arm["bayesian"]["test_metrics"][key]),
                    "bayesian_minus_original": (
                        float(arm["bayesian"]["test_metrics"][key])
                        - float(arm["original"]["test_metrics"][key])
                    ),
                } for key in METRIC_DIRECTIONS
            },
        })
    rng = np.random.default_rng(bootstrap_seed)
    stats = {}
    for metric, direction in METRIC_DIRECTIONS.items():
        delta = np.asarray([item["metrics"][metric]["bayesian_minus_original"] for item in paired])
        draws = rng.choice(delta, size=(bootstrap_samples, len(delta)), replace=True).mean(axis=1)
        lower, upper = np.quantile(draws, [0.025, 0.975])
        stats[metric] = {
            "direction": direction,
            "paired_mean_bayesian_minus_original": float(delta.mean()),
            "sample_std_bayesian_minus_original": float(delta.std(ddof=1)),
            "paired_bootstrap_95_percentile_interval": [float(lower), float(upper)],
        }
    return {
        "schema": "bounded-iris-multiseed-analysis-v1",
        "evidence_grade": "BOUNDED_REAL_DATA_MULTI_SEED",
        "source_commit": anchor["source_commit"],
        "dataset_sha256": anchor["dataset_sha256"],
        "split_partition_sha256_v2": anchor["split_partition_sha256_v2"],
        "conditioned_on_one_fixed_iris_split": True,
        "distinct_seed_count": len(paired),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "interpretation_limits": [
            "The bootstrap resamples only model seeds, not independent datasets or data splits",
            "Small n intervals are exploratory; no claim of population-level superiority",
            "Shadow posterior compute, calibration, parameter-matched and compute-matched controls remain unmeasured",
        ],
        "seed_pairs": paired,
        "paired_summary": stats,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receipts", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap-samples", type=int, default=10000)
    a = p.parse_args()
    report = analyze(a.receipts, bootstrap_samples=a.bootstrap_samples)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
