"""Analyze original signed visual worker summaries without fitting new models.

Use only descriptive paired differences. Exact held-out split and derived
runner-seed identity must agree across every controlled adapter/site.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from simple_adp.visual_paired_metrics import summarize_visual_paired_metrics
from visual_official_fashion_matrix import CONTROL_ADAPTERS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True,
                        help="completed official visual fixed/RR matrix JSON")
    parser.add_argument("--output", type=Path,
                        help="output path; defaults to _paired_metrics suffix")
    args = parser.parse_args(argv)
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    if not isinstance(matrix, dict) or matrix.get("status") != "completed":
        raise ValueError("paired descriptive analysis requires a completed matrix")
    if matrix.get("failed_jobs") != 0:
        raise ValueError("paired descriptive analysis rejects failed jobs")
    if not isinstance(matrix.get("scientific_scope"), str) or not matrix["scientific_scope"]:
        raise ValueError("matrix must explicitly disclose its scientific scope")
    rows = matrix.get("rows")
    if not isinstance(rows, list):
        raise ValueError("matrix rows must be a list of signed worker summaries")
    seed_count = matrix.get("model_seed_count", 1)
    if type(seed_count) is not int or seed_count < 1:
        raise ValueError("invalid matrix model seed count")
    if seed_count == 1:
        run_seeds = sorted({r.get("seed") for r in rows if isinstance(r, dict)})
    else:
        run_seeds = matrix.get("derived_run_seeds")
    if (
        not isinstance(run_seeds, list)
        or len(run_seeds) != seed_count
        or len(set(run_seeds)) != seed_count
        or any(type(seed) is not int for seed in run_seeds)
    ):
        raise ValueError("invalid or missing actual derived runner seed provenance")
    expected = 2 * len(CONTROL_ADAPTERS) * seed_count
    if len(rows) != expected or matrix.get("selected_worker_jobs") != expected:
        raise ValueError("matrix is missing one or more expected controlled worker jobs")
    if matrix.get("completed_jobs") != expected:
        raise ValueError("matrix worker completion total does not match signed rows")

    analysis = summarize_visual_paired_metrics(
        rows, expected_adapters=CONTROL_ADAPTERS, expected_seeds=run_seeds,
    )
    analysis["dataset"] = matrix.get("dataset")
    analysis["source_matrix_format"] = matrix.get("format")
    analysis["source_scientific_scope"] = matrix["scientific_scope"]
    analysis["source_matrix_file"] = args.matrix.name
    target = (
        args.output if args.output is not None else
        args.matrix.with_name(args.matrix.stem + "_paired_metrics.json")
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    try:
        tmp.write_text(json.dumps(analysis, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()
    print(json.dumps({
        "output": str(target),
        "adapter_count": analysis["adapter_count"],
        "model_seed_count": len(analysis["actual_run_seeds"]),
        "scientific_performance_conclusion": None,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
