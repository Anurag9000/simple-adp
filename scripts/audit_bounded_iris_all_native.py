"""Strict 13-method native real-Iris original/variance execution auditor.

Verifies actual model+checkpoint artifact hashes, shared source/split/seed
and original-vs-shadow-BGD treatment parity. This is one real dataset,
one fixed split and one seed, not a claim of full scientific coverage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .analyze_bounded_iris_pair_results import validate_receipt


EXPECTED_NATIVE_METHOD_FAMILIES = {
    "LGND-W-Greedy": "width_only",
    "ADP-W-RR": "width_only",
    "ADP-W-All": "width_only",
    "ADP-WD-RR-D1": "width_to_depth",
    "ADP-WD-All-Dcur": "width_to_depth",
    "ADP-WD-RR-Dcur": "width_to_depth",
    "ADP-WD-All-D1": "width_to_depth",
    "ADP-DW-D1-RR": "depth_to_width",
    "ADP-DW-Dcur-All": "depth_to_width",
    "ADP-DW-Dcur-RR": "depth_to_width",
    "ADP-DW-D1-All": "depth_to_width",
    "ADP-D-D1": "depth_only",
    "ADP-D-Dcur": "depth_only",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_native_methods(paths) -> dict:
    records = [validate_receipt(path) for path in paths]
    if len(records) != len(EXPECTED_NATIVE_METHOD_FAMILIES):
        raise ValueError("requires exactly 13 real native growth method pairs")
    shared = (
        "source_commit", "dataset_sha256", "dataset_source",
        "split_legacy_sha256", "split_partition_sha256_v2",
        "split_archive_sha256", "shared_settings_digest",
        "seed", "sklearn_version",
    )
    anchor = records[0]
    seen = {}
    for path, record in zip(paths, records):
        for key in shared:
            if record.get(key) != anchor.get(key):
                raise ValueError(f"native method matrix changed shared source field {key}")
        route = record.get("route") or {}
        method = route.get("model_id")
        expected = EXPECTED_NATIVE_METHOD_FAMILIES.get(method)
        if expected is None or record.get("growth_family") != expected:
            raise ValueError(f"unexpected or incompatible native growing method {method!r}")
        if route.get("suite_group") != expected:
            raise ValueError(f"native route suite group mismatch for {method}")
        profile = "corrected" if expected in ("width_to_depth", "depth_to_width") else None
        if route.get("generic_profile") != profile:
            raise ValueError(f"native method generic profile mismatch for {method}")
        if method in seen:
            raise ValueError(f"duplicate native growth method: {method}")
        for arm in ("original", "bayesian"):
            info = record["arms"][arm]
            artifact = Path(path).parent / arm / "checkpoint.pt"
            if (not artifact.is_file() or
                    _sha(artifact) != info.get("checkpoint_sha256") or
                    artifact.stat().st_size != info.get("checkpoint_size_bytes")):
                raise ValueError(f"missing or tampered native checkpoint: {method}/{arm}")
        seen[method] = {
            "growth_family": expected,
            "original_architecture": record["arms"]["original"]["selected_architecture"],
            "bayesian_architecture": record["arms"]["bayesian"]["selected_architecture"],
            "original_accuracy": float(record["arms"]["original"]["test_metrics"]["accuracy"]),
            "bayesian_accuracy": float(record["arms"]["bayesian"]["test_metrics"]["accuracy"]),
            "accuracy_delta_bayesian_minus_original": float(
                record["arms"]["bayesian"]["test_metrics"]["accuracy"]
                - record["arms"]["original"]["test_metrics"]["accuracy"]
            ),
        }
    if set(seen) != set(EXPECTED_NATIVE_METHOD_FAMILIES):
        raise ValueError("native paired matrix is missing canonical growing methods")
    counts = dict(sorted(Counter(EXPECTED_NATIVE_METHOD_FAMILIES.values()).items()))
    return {
        "schema": "bounded-iris-13-native-growing-methods-audit-v1",
        "evidence_grade": "BOUNDED_REAL_DATA_NATIVE_METHODS_PASSED",
        "source_commit": anchor["source_commit"],
        "dataset_sha256": anchor["dataset_sha256"],
        "split_partition_sha256_v2": anchor["split_partition_sha256_v2"],
        "seed": anchor["seed"],
        "method_count": len(seen),
        "pair_count": len(seen),
        "execution_records": 2 * len(seen),
        "family_counts": counts,
        "fixed_no_growth": ["MLP-Fixed"],
        "methods": dict(sorted(seen.items())),
        "limitations": [
            "One genuine Iris dataset and one fixed train/validation/test split",
            "One model seed, short four-epoch bounded training, no broad statistics",
            "Successful job execution does not imply accepted structural growth",
            "No per-method total FLOP/MC overhead, calibrated posterior or matched compute",
            "No source-weighted real visual model matrix, unavailable data, or GPU proof",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    report = audit_native_methods(options.receipts)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
