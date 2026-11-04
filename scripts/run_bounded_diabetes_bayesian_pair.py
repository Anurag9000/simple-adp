"""Bounded real diabetes regression experiment: original ADP vs shadow-BGD.

Uses the real scikit-learn diabetes progression dataset, with one shared
train/validation/test partition. A proof of executable regression parity,
not a full scientific campaign or a compute-matched comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from simple_adp.config import SearchSettings
from simple_adp.runner.backend import seed_everything
from simple_adp.runner.datasets import PreparedTaskData
from simple_adp.runner.models import SimpleADPModelAdapter
from simple_adp.runner.plan import SplitSpec, _sha_indices, _sha_indices_v2
from simple_adp.runner.route import ModelRoute


def _default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _dataset():
    raw = load_diabetes()
    X = np.asarray(raw.data, dtype=np.float32)
    y = np.asarray(raw.target, dtype=np.float32)
    h = hashlib.sha256(b"sklearn-diabetes-packed-real-data-v1")
    h.update(X.tobytes(order="C"))
    h.update(y.tobytes(order="C"))
    all_rows = np.arange(len(y), dtype=np.int64)
    train_validation, test_rows = train_test_split(
        all_rows, test_size=0.2, random_state=731,
    )
    train_rows, validation_rows = train_test_split(
        train_validation, test_size=0.25, random_state=732,
    )
    train = tuple(map(int, train_rows))
    validation = tuple(map(int, validation_rows))
    test = tuple(map(int, test_rows))
    if len(set(train) | set(validation) | set(test)) != len(y):
        raise RuntimeError("split does not cover all original rows exactly once")
    split = SplitSpec(
        repeat=0, fold=0, sample=0,
        train_indices=train, validation_indices=validation, test_indices=test,
        split_sha256=_sha_indices(train, validation, test),
    )
    data = PreparedTaskData(
        dataset_id="sklearn-diabetes-442",
        name="Scikit-learn Diabetes progression",
        target_name="disease progression",
        X=X, y=y, feature_schema=None,
        content_sha256=h.hexdigest(),
        dataset_metadata={
            "origin": "sklearn.datasets.load_diabetes",
            "sklearn_version": sklearn_version,
            "sample_count": len(y), "feature_count": X.shape[1],
            "train_test_split_seed": 731, "train_validation_split_seed": 732,
            "target_units": "sklearn diabetes progression target",
        },
        task_kind="regression",
        task_adapter=None,
        task_family="tabular",
        task_variant="diabetes_real_regression",
        split_protocol="random_fixed_60_20_20",
    )
    return data, split


def _settings(seed, criterion):
    return SearchSettings(
        init_width=3, init_depth=1, max_width=6, max_depth=2,
        max_neurons=50, max_epochs=4, batch_size=32, patience_es=3,
        patience_width_exp=1, patience_depth_exp=1,
        ex_k_width=1, ex_k_depth=1,
        max_candidates=2, seed=seed, device="cpu",
        expansion_criterion=criterion,
    )


def run(output, *, seed=73):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("a new empty results directory is required")
    output.mkdir(parents=True, exist_ok=True)
    data, split = _dataset()
    split_path = output / "diabetes_split.npz"
    split.save(split_path)
    if SplitSpec.load(split_path, repeat=0, fold=0, sample=0) != split:
        raise RuntimeError("split integrity check failed")
    original = _settings(seed, "loss").as_dict()
    alternative = _settings(seed, "variance_stagnation").as_dict()
    changed = {k: [original.get(k), alternative.get(k)]
               for k in sorted(set(original) | set(alternative))
               if original.get(k) != alternative.get(k)}
    if changed != {"expansion_criterion": ["loss", "variance_stagnation"]}:
        raise RuntimeError(f"unpermitted paired-science difference: {changed}")
    route = ModelRoute("ADP-W-All", suite_group="width_only",
                       init_width=3, init_depth=1)
    arms = {}
    for arm, criterion in (("original", "loss"), ("bayesian", "variance_stagnation")):
        seed_everything(seed)
        arm_dir = output / arm
        arm_dir.mkdir()
        result = SimpleADPModelAdapter().fit_predict(
            route, data=data, split=split, seed=seed,
            settings=_settings(seed, criterion),
            checkpoint_path=str(arm_dir / "checkpoint.pt"),
        )
        artifact = result.get("artifact_path")
        if not isinstance(artifact, str) or not Path(artifact).is_file():
            raise RuntimeError(f"missing {arm} trained inference bundle")
        metrics = result.get("test_metrics", {})
        for key in ("rmse", "mae", "r2"):
            if not np.isfinite(metrics.get(key, float("nan"))):
                raise RuntimeError(f"{arm} has invalid held-out {key}")
        result.update({
            "growth_trigger": criterion,
            "model_artifact_sha256": _hash(artifact),
            "model_artifact_size_bytes": Path(artifact).stat().st_size,
            "checkpoint_sha256": _hash(arm_dir / "checkpoint.pt"),
            "checkpoint_size_bytes": (arm_dir / "checkpoint.pt").stat().st_size,
        })
        arms[arm] = result
    science = {
        "schema": "bounded-diabetes-original-vs-shadow-bgd-v1",
        "evidence_grade": "BOUNDED_REAL_DATA_PASSED",
        "not_full_science": True,
        "source_commit": os.environ.get("GITHUB_SHA", "UNAVAILABLE"),
        "dataset_source": "sklearn.datasets.load_diabetes",
        "sklearn_version": sklearn_version,
        "dataset_sha256": data.content_sha256,
        "split_sha256": split.split_sha256,
        "split_partition_sha256_v2": _sha_indices_v2(
            split.train_indices, split.validation_indices, split.test_indices,
        ),
        "split_archive_sha256": _hash(split_path),
        "split_sizes": {
            "train": len(split.train_indices),
            "validation": len(split.validation_indices),
            "test": len(split.test_indices),
        },
        "route": route.as_dict(),
        "seed": seed,
        "scientific_settings_diff": changed,
        "arms": arms,
        "paired_rmse_delta_bayesian_minus_original": float(
            arms["bayesian"]["test_metrics"]["rmse"] -
            arms["original"]["test_metrics"]["rmse"]
        ),
        "unmeasured": [
            "multi-seed regression confidence intervals",
            "posterior calibration and uncertainty quality",
            "MC observer gradient-pass overhead, energy and peak memory",
            "full forecasting/recommendation, visual and cross-dataset studies",
        ],
    }
    receipt = output / "pair_receipt.json"
    receipt.write_text(json.dumps(science, indent=2, sort_keys=True,
                                  default=_default) + "\n", encoding="utf-8")
    reread = json.loads(receipt.read_text(encoding="utf-8"))
    if reread["schema"] != science["schema"] or set(reread["arms"]) != {"original", "bayesian"}:
        raise RuntimeError("regression pair receipt corrupted during publication")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=73)
    options = parser.parse_args()
    receipt = run(options.output, seed=options.seed)
    print(f"real regression paired receipt: {receipt}")
    print(receipt.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
