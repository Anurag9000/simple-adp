"""Audit four real Iris original/variance architecture-mechanism pair receipts.

One dataset and one seed are intentionally not treated as multi-dataset science.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyze_bounded_iris_pair_results import validate_receipt

EXPECTED_FAMILIES = {
    "ADP-W-All": "width_only",
    "ADP-D-Dcur": "depth_only",
    "ADP-WD-All-Dcur": "width_to_depth",
    "ADP-DW-Dcur-All": "depth_to_width",
}


def audit_family_receipts(paths) -> dict:
    records = [validate_receipt(path) for path in paths]
    if len(records) != len(EXPECTED_FAMILIES):
        raise ValueError("requires precisely four real-growth family pairs")
    source = records[0]
    shared = ("source_commit", "dataset_sha256", "split_legacy_sha256",
              "split_partition_sha256_v2", "split_archive_sha256",
              "shared_settings_digest", "seed", "sklearn_version")
    observed = {}
    for record in records:
        for field in shared:
            if record.get(field) != source.get(field):
                raise ValueError(f"architecture-family comparison changed {field}")
        route = record.get("route") or {}
        model_id = route.get("model_id")
        required_family = EXPECTED_FAMILIES.get(model_id)
        if required_family is None or record.get("growth_family") != required_family:
            raise ValueError(f"unrecognized growth mechanism family {model_id!r}")
        if model_id in observed:
            raise ValueError(f"duplicate growth mechanism job {model_id}")
        observed[model_id] = {
            "growth_family": required_family,
            "original_widths": record["arms"]["original"]["selected_architecture"],
            "bayesian_widths": record["arms"]["bayesian"]["selected_architecture"],
            "original_accuracy": float(record["arms"]["original"]["test_metrics"]["accuracy"]),
            "bayesian_accuracy": float(record["arms"]["bayesian"]["test_metrics"]["accuracy"]),
            "source_revision": record["source_commit"],
            "evidence": "BOUNDED_REAL_DATA_PASSED",
        }
    if set(observed) != set(EXPECTED_FAMILIES):
        raise ValueError("missing one or more native growth mechanism families")
    return {
        "schema": "bounded-iris-four-growth-families-audit-v1",
        "evidence_grade": "BOUNDED_REAL_DATA_FAMILY_PASSED",
        "source_commit": source["source_commit"],
        "dataset_sha256": source["dataset_sha256"],
        "split_partition_sha256_v2": source["split_partition_sha256_v2"],
        "seed": source["seed"],
        "distinct_native_growth_family_count": len(observed),
        "model_ids": sorted(observed),
        "families": observed,
        "limits": [
            "Only one genuine Iris dataset, one fixed split, and one model seed",
            "One representative method from each of four native policy families; not all 13 adaptive source methods",
            "Only bounded four-epoch CPU training; no actual GPU, visual or cross-task inference",
            "No matched compute/parameter/FLOPs, no observer calibration and no model-seed statistical comparison",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_family_receipts(args.receipts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
