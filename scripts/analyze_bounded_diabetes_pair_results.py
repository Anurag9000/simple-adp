"""Receipt-validated exploratory paired-seed regression analysis on real Diabetes.

Compare ONLY genuinely completed same-source, same-split loss/variance pairs.
The bootstrap is conditional on one fixed dataset/split and is not general
scientific evidence of criterion superiority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


SCHEMA = "bounded-diabetes-original-vs-shadow-bgd-v1"
PAIRED_CHANGE = {"expansion_criterion": ["loss", "variance_stagnation"]}
METRICS = {"rmse": "lower", "mae": "lower", "r2": "higher"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_receipt(path: Path) -> dict:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"incorrect real Diabetes receipt schema: {path}")
    if payload.get("evidence_grade") != "BOUNDED_REAL_DATA_PASSED":
        raise ValueError(f"unverified regression execution: {path}")
    if payload.get("scientific_settings_diff") != PAIRED_CHANGE:
        raise ValueError(f"original and variance settings are not matched: {path}")
    if payload.get("dataset_source") != "sklearn.datasets.load_diabetes":
        raise ValueError(f"wrong regression dataset origin: {path}")
    if payload.get("split_sizes") != {"train": 264, "validation": 89, "test": 89}:
        raise ValueError(f"wrong regression split sizes: {path}")
    if not isinstance(payload.get("source_commit"), str) or len(payload["source_commit"]) != 40:
        raise ValueError(f"unbound source revision: {path}")
    if type(payload.get("seed")) is not int:
        raise ValueError(f"invalid model seed: {path}")
    split_file = path.parent / "diabetes_split.npz"
    if not split_file.is_file() or _sha256(split_file) != payload.get("split_archive_sha256"):
        raise ValueError(f"split archive tampering or absence: {path}")
    arms = payload.get("arms")
    if not isinstance(arms, dict) or set(arms) != {"original", "bayesian"}:
        raise ValueError(f"paired regression arms missing: {path}")
    for name, trigger in (("original", "loss"), ("bayesian", "variance_stagnation")):
        arm = arms[name]
        if arm.get("growth_trigger") != trigger:
            raise ValueError(f"wrong growth trigger in {name}: {path}")
        for artifact_name, file_name in (
            ("model_artifact", "estimator.model.pt"),
            ("checkpoint", "checkpoint.pt"),
        ):
            file = path.parent / name / file_name
            if not file.is_file() or _sha256(file) != arm.get(artifact_name + "_sha256"):
                raise ValueError(f"{name} {artifact_name} failed SHA256 validation: {path}")
            if file.stat().st_size != arm.get(artifact_name + "_size_bytes"):
                raise ValueError(f"{name} {artifact_name} wrong size: {path}")
        widths = arm.get("selected_architecture")
        if not isinstance(widths, list) or not widths or any(type(w) is not int or w <= 0 for w in widths):
            raise ValueError(f"invalid resulting architecture: {path}")
        for metric in METRICS:
            value = arm.get("test_metrics", {}).get(metric)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"invalid regression {metric} in {name}: {path}")
    return payload


def analyze(receipts, *, bootstrap_samples: int = 10000, seed: int = 1931) -> dict:
    if bootstrap_samples < 1000:
        raise ValueError("bootstrap needs >=1000 replicates")
    records = [verify_receipt(Path(item)) for item in receipts]
    if len(records) < 2:
        raise ValueError("two independent model seeds required")
    seeds = [record["seed"] for record in records]
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate model seed")
    reference = records[0]
    same = ("source_commit", "dataset_source", "dataset_sha256", "sklearn_version",
            "split_sha256", "split_partition_sha256_v2", "split_archive_sha256",
            "split_sizes", "route", "scientific_settings_diff")
    for record in records[1:]:
        for field in same:
            if record.get(field) != reference.get(field):
                raise ValueError(f"unpaired dataset/split/source configuration: {field}")
    pairs = []
    for item in sorted(records, key=lambda r: r["seed"]):
        original, alternative = item["arms"]["original"], item["arms"]["bayesian"]
        pairs.append({
            "seed": item["seed"],
            "original_architecture": original["selected_architecture"],
            "bayesian_architecture": alternative["selected_architecture"],
            "metrics": {
                metric: {
                    "original": float(original["test_metrics"][metric]),
                    "bayesian": float(alternative["test_metrics"][metric]),
                    "bayesian_minus_original": (
                        float(alternative["test_metrics"][metric])
                        - float(original["test_metrics"][metric])
                    ),
                }
                for metric in METRICS
            },
        })
    rng = np.random.default_rng(seed)
    summary = {}
    for metric, direction in METRICS.items():
        deltas = np.asarray([item["metrics"][metric]["bayesian_minus_original"] for item in pairs])
        draws = rng.choice(deltas, (bootstrap_samples, len(deltas)), replace=True).mean(axis=1)
        lo, hi = np.quantile(draws, [0.025, 0.975])
        summary[metric] = {
            "direction": direction,
            "mean_difference_bayesian_minus_original": float(deltas.mean()),
            "sample_std": float(deltas.std(ddof=1)),
            "paired_seed_bootstrap_95_percentile": [float(lo), float(hi)],
        }
    return {
        "schema": "bounded-diabetes-seed-paired-analysis-v1",
        "evidence_grade": "BOUNDED_REAL_DATA_MULTI_SEED",
        "source_commit": reference["source_commit"],
        "dataset_sha256": reference["dataset_sha256"],
        "split_partition_sha256_v2": reference["split_partition_sha256_v2"],
        "seed_count": len(pairs),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_rng_seed": seed,
        "conditional_on_one_diabetes_split": True,
        "limitations": [
            "Bootstrap resamples only model initialization seeds",
            "The four-epoch bounded jobs do not establish useful predictive regression performance",
            "No independent datasets, data split repetitions, calibration or matched resource analyses",
        ],
        "paired_results": pairs,
        "paired_summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    report = analyze(args.receipts, bootstrap_samples=args.bootstrap_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
