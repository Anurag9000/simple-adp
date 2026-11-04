"""Bounded *real Iris data* loss-vs-shadow-BGD supervised ADP paired proof.

This is not the full scientific campaign. It runs one matched CPU seed through
the actual SimpleADPModelAdapter and emits source-hashed result/checkpoint
receipts. No network download or synthetic data is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from simple_adp.config import SearchSettings
from simple_adp.registry import MODEL_REGISTRY
from simple_adp.runner.backend import seed_everything
from simple_adp.runner.datasets import PreparedTaskData
from simple_adp.runner.models import SimpleADPModelAdapter
from simple_adp.runner.plan import SplitSpec, _sha_indices, _sha_indices_v2
from simple_adp.runner.route import ModelRoute


def _canonical(data):
    return json.dumps(data, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"), default=_json_default)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"nonserializable object: {type(value).__name__}")


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_and_split():
    iris = load_iris()
    X = np.asarray(iris.data, dtype=np.float32)
    y = np.asarray(iris.target, dtype=np.int64)
    source_hasher = hashlib.sha256()
    source_hasher.update(b"sklearn-iris-packed-real-data-v1")
    source_hasher.update(X.tobytes(order="C"))
    source_hasher.update(y.tobytes(order="C"))
    dataset_sha256 = source_hasher.hexdigest()
    all_ids = np.arange(len(y), dtype=np.int64)
    development, test = train_test_split(
        all_ids, test_size=0.2, stratify=y, random_state=731,
    )
    train, validation = train_test_split(
        development, test_size=0.25, stratify=y[development],
        random_state=732,
    )
    train = tuple(int(v) for v in train)
    validation = tuple(int(v) for v in validation)
    test = tuple(int(v) for v in test)
    split = SplitSpec(
        repeat=0, fold=0, sample=0, train_indices=train,
        validation_indices=validation, test_indices=test,
        split_sha256=_sha_indices(train, validation, test),
    )
    if set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test):
        raise RuntimeError("real Iris partitions overlap")
    data = PreparedTaskData(
        dataset_id="sklearn-iris-150",
        name="Iris flowers (Fisher)",
        target_name="species",
        X=X,
        y=y,
        feature_schema=None,
        content_sha256=dataset_sha256,
        dataset_metadata={
            "origin": "sklearn.datasets.load_iris",
            "sklearn_version": sklearn_version,
            "samples": len(y),
            "features": X.shape[1],
            "split_seed_train_test": 731,
            "split_seed_train_validation": 732,
            "train_rows": len(train), "validation_rows": len(validation),
            "test_rows": len(test),
        },
        task_kind="classification",
        task_adapter=None,
        task_family="tabular",
        task_variant="iris_real_multiclass",
        split_protocol="stratified_60_20_20",
    )
    return data, split


def _settings(criterion, seed):
    return SearchSettings(
        init_width=3, init_depth=1,
        max_epochs=4, batch_size=16, patience_es=3,
        max_width=6, max_depth=2, max_neurons=30,
        patience_width_exp=1, patience_depth_exp=1,
        ex_k_width=1, ex_k_depth=1,
        max_candidates=2, seed=seed,
        expansion_criterion=criterion, device="cpu",
    )


GROWTH_FAMILY_PROOFS = {
    "ADP-W-All": "width_only",
    "ADP-D-Dcur": "depth_only",
    "ADP-WD-All-Dcur": "width_to_depth",
    "ADP-DW-Dcur-All": "depth_to_width",
}
# Every canonical growing method, not just the four original family proofs.
# Derive the route family from the authoritative source taxonomy. The
# MLP-Fixed baseline remains a deliberately unpaired no-growth control.
_ORDER_TO_FAMILY = {
    "width": "width_only",
    "depth": "depth_only",
    "width_then_depth": "width_to_depth",
    "depth_then_width": "depth_to_width",
}
ALL_NATIVE_METHOD_FAMILIES = {
    model_id: _ORDER_TO_FAMILY[config.order]
    for model_id, config in MODEL_REGISTRY.items()
    if config.controller != "fixed"
}
if len(ALL_NATIVE_METHOD_FAMILIES) != 13 or any(
    ALL_NATIVE_METHOD_FAMILIES.get(model_id) != family
    for model_id, family in GROWTH_FAMILY_PROOFS.items()
):
    raise RuntimeError("canonical ADP taxonomy changed; re-audit bounded method eligibility")


def run(output, *, seed=73, model_id="ADP-W-All"):
    root = Path(output).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("bounded proof requires a fresh empty output directory")
    root.mkdir(parents=True, exist_ok=True)
    data, split = _dataset_and_split()
    split_path = root / "iris_split.npz"
    split.save(split_path)
    if SplitSpec.load(split_path, repeat=0, fold=0, sample=0) != split:
        raise RuntimeError("split reread differs from original")
    if model_id not in ALL_NATIVE_METHOD_FAMILIES:
        raise ValueError(f"unsupported bounded Iris growing method: {model_id!r}")
    route = ModelRoute(
        model_id, suite_group=ALL_NATIVE_METHOD_FAMILIES[model_id],
        generic_profile="corrected" if MODEL_REGISTRY[model_id].generic else None,
        init_width=3, init_depth=1,
    )
    common = _settings("loss", seed).as_dict()
    alternative = _settings("variance_stagnation", seed).as_dict()
    diff = {key: (common.get(key), alternative.get(key))
            for key in set(common) | set(alternative)
            if common.get(key) != alternative.get(key)}
    if set(diff) != {"expansion_criterion"}:
        raise RuntimeError(f"paired scientific configuration changed: {diff}")
    arms = {}
    for name, criterion in (("original", "loss"), ("bayesian", "variance_stagnation")):
        seed_everything(seed)
        out = root / name
        out.mkdir()
        result = SimpleADPModelAdapter().fit_predict(
            route, data=data, split=split, seed=seed,
            settings=_settings(criterion, seed),
            checkpoint_path=str(out / "checkpoint.pt"),
        )
        artifact = result.get("artifact_path")
        if not isinstance(artifact, str) or not Path(artifact).is_file():
            raise RuntimeError(f"{name} has no verified inference model artifact")
        metrics = result.get("test_metrics")
        if not isinstance(metrics, dict) or not np.isfinite(metrics.get("accuracy", float("nan"))):
            raise RuntimeError(f"{name} has no finite held-out classification accuracy")
        result["model_artifact_sha256"] = _sha256_file(artifact)
        result["model_artifact_size_bytes"] = Path(artifact).stat().st_size
        checkpoint = out / "checkpoint.pt"
        if not checkpoint.is_file():
            raise RuntimeError(f"{name} has no persisted search checkpoint")
        result["checkpoint_sha256"] = _sha256_file(checkpoint)
        result["checkpoint_size_bytes"] = checkpoint.stat().st_size
        result["growth_trigger"] = criterion
        arms[name] = result
    compared = {
        "schema": "bounded-iris-original-vs-shadow-bgd-v1",
        "evidence_grade": "BOUNDED_REAL_DATA_PASSED",
        "not_full_science": True,
        "source_commit": os.environ.get("GITHUB_SHA", "UNAVAILABLE"),
        "dataset_source": "sklearn.datasets.load_iris",
        "dataset_sha256": data.content_sha256,
        "sklearn_version": sklearn_version,
        "split_legacy_sha256": split.split_sha256,
        "split_partition_sha256_v2": _sha_indices_v2(
            split.train_indices, split.validation_indices, split.test_indices,
        ),
        "split_archive_sha256": _sha256_file(split_path),
        "split_sizes": {
            "train": len(split.train_indices),
            "validation": len(split.validation_indices),
            "test": len(split.test_indices),
        },
        "route": route.as_dict(),
        "growth_family": ALL_NATIVE_METHOD_FAMILIES[model_id],
        "seed": int(seed),
        "shared_settings_digest": hashlib.sha256(_canonical({
            key: value for key, value in common.items()
            if key != "expansion_criterion"
        }).encode()).hexdigest(),
        "scientific_settings_diff": diff,
        "arms": arms,
        "accuracy_delta_bayesian_minus_original": (
            float(arms["bayesian"]["test_metrics"]["accuracy"])
            - float(arms["original"]["test_metrics"]["accuracy"])
        ),
        "science_gaps": [
            "one real dataset/one split; further growth controllers and task families not necessarily covered",
            "uncertainty calibration, multi-seed confidence intervals and compute matched controls absent",
            "separate shadow Monte Carlo gradient-pass costs not measured in adapter result",
            "full visual and nonclassification provider matrix not run",
        ],
    }
    receipt = root / "pair_receipt.json"
    receipt.write_text(json.dumps(compared, indent=2, sort_keys=True,
                                  default=_json_default) + "\n", encoding="utf-8")
    # Read receipt back before reporting success.
    saved = json.loads(receipt.read_text(encoding="utf-8"))
    if saved["schema"] != compared["schema"] or set(saved["arms"]) != {"original", "bayesian"}:
        raise RuntimeError("bounded Iris pair receipt failed read-back")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--model-id", choices=tuple(ALL_NATIVE_METHOD_FAMILIES),
                        default="ADP-W-All")
    arguments = parser.parse_args()
    path = run(arguments.output, seed=arguments.seed, model_id=arguments.model_id)
    print(f"real-data paired receipt: {path}")
    print(path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
