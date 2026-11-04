"""Execute real packaged Iris and Diabetes through the persisted suite worker.

Unlike adapter-only smoke studies, this builds a complete unsliced
SuitePlan and launches independent original/Bayesian *child worker* commands.
Outputs have SHA-256 receipts but do not establish broad science.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.models import SimpleADPModelAdapter
from simple_adp.runner.pair_audit import audit_manifest
from simple_adp.runner.plan import SuitePlan
from simple_adp.runner.plugins import load_registries


class _OnlyWidthAll(SimpleADPModelAdapter):
    """Planning filter only. Workers resolve the unchanged built-in adapter."""
    def enumerate_routes(self, generic_profiles=("corrected",), settings=None):
        routes = super().enumerate_routes(generic_profiles, settings)
        selected = [route for route in routes if route.model_id == "ADP-W-All"]
        if len(selected) != 1:
            raise RuntimeError("exactly one ADP-W-All concrete source route required")
        return selected


def _hash(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(root: Path):
    root = Path(root).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("real worker proof requires a fresh empty output directory")
    root.mkdir(parents=True, exist_ok=True)
    os.environ["SIMPLE_ADP_ENABLE_SKLEARN_REAL"] = "1"
    os.environ["SIMPLE_ADP_DISABLE_GPU_ACCELERATORS"] = "1"
    os.environ["TRAINING_CONTROL_CPU_ONLY"] = "1"
    os.environ["TABULAR_SKIP_SYSTEMD_SCOPE"] = "1"
    datasets, models = load_registries()
    settings = RunnerSettings(
        concurrency=1, cpu_only=True, scheduler="fixed",
        repeat_count=1, missing_present_task_repeats=1,
        model_seeds=(73,), generic_profiles=("corrected",),
        tasks=("width_only",),
        include_vanilla=False, include_width_to_depth=False,
        include_depth_to_width=False, include_depth_only=False,
        run_hardness_screen=False,
        width_only_depths=(1,), width_only_width=3,
        width_to_depth_depth=1, width_to_depth_width=3,
        max_width=6, max_depth=2, max_neurons=30,
        max_epochs=4, batch_size=16, patience_es=3,
        patience_width_exp=1, patience_depth_exp=1,
        max_candidates=2,
        pressure_poll_seconds=0.01, post_launch_sample_delay_seconds=0,
    )
    plan = SuitePlan.build(
        settings=settings,
        results_dir=root / "results",
        run_root=root / "runtime",
        dataset_providers=(datasets.get("sklearn_real"),),
        dataset_suites=(("sklearn_real", "iris"), ("sklearn_real", "diabetes")),
        model_adapters=(_OnlyWidthAll(),),
    )
    audit = audit_manifest(
        plan.manifest, source_commit=os.environ.get("GITHUB_SHA", "UNAVAILABLE")
    )
    if (len(plan.runs) != 4 or audit["summary"]["matched_pairs_selected"] != 2
            or audit["summary"]["primary_execution_records_selected"] != 4):
        raise RuntimeError("real worker suite did not materialize two exact pairs")
    plan.write_manifest(root / "runtime")
    (root / "plan_audit.json").write_text(
        json.dumps(audit, sort_keys=True, indent=2) + "\n", encoding="utf-8",
    )
    receipts = {}
    for spec in plan.runs:
        completed = subprocess.run(
            spec.command, cwd=Path(__file__).resolve().parents[1],
            env=os.environ.copy(), text=True, capture_output=True,
            check=False, timeout=180,
        )
        log_dir = root / "logs" / spec.identity.run_id
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (log_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"worker failed for {spec.identity.run_id}: "
                + completed.stderr[-1600:]
            )
        result_path = Path(spec.result_dir) / "result.json"
        if not result_path.is_file():
            raise RuntimeError("worker returned without a persisted result")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (result.get("status") != "completed"
                or result.get("job_id") != spec.identity.run_id
                or result.get("run_signature") != plan.run_signature
                or result.get("split_sha256") != spec.identity.split_sha256
                or result.get("expansion_criterion") != spec.worker_settings.get(
                    "expansion_criterion", "loss"
                )):
            raise RuntimeError("worker result does not match its planned job identity")
        metrics = result.get("test_metrics") or {}
        required = "accuracy" if spec.task_ref.suite == "iris" else "rmse"
        if not isinstance(metrics.get(required), (float, int)) or not math.isfinite(metrics[required]):
            raise RuntimeError(f"worker has no finite real-data {required} metric")
        artifact = result.get("artifact_path")
        if not isinstance(artifact, str) or not Path(artifact).is_file():
            raise RuntimeError("worker has no trained inference bundle")
        receipts[spec.identity.run_id] = {
            "task": spec.task_ref.suite,
            "arm": "bayesian" if "counterpart_of" in spec.worker_settings else "original",
            "run_id": spec.identity.run_id,
            "split_sha256": spec.identity.split_sha256,
            "seed": spec.identity.seed,
            "result_json_sha256": _hash(result_path),
            "model_sha256": _hash(Path(artifact)),
            "metric": required,
            "metric_value": float(metrics[required]),
            "selected_architecture": result.get("selected_architecture"),
        }
    pair_records = []
    for row in audit["pairs"]:
        original = receipts[row["original_job_id"]]
        bayesian = receipts[row["bayesian_job_id"]]
        if (original["task"] != bayesian["task"]
                or original["seed"] != bayesian["seed"]
                or original["split_sha256"] != bayesian["split_sha256"]
                or original["arm"] != "original" or bayesian["arm"] != "bayesian"):
            raise RuntimeError("real worker pair changed dataset, split or initialization seed")
        pair_records.append({
            "pair_id": row["pair_id"],
            "task": original["task"],
            "original": original,
            "bayesian": bayesian,
        })
    evidence = {
        "schema": "real-sklearn-planner-to-worker-pairs-v1",
        "evidence_grade": "BOUNDED_REAL_DATA_WORKER_EXECUTED",
        "source_commit": os.environ.get("GITHUB_SHA", "UNAVAILABLE"),
        "pairs_count": len(pair_records),
        "execution_records": len(plan.runs),
        "manifest_sha256": _hash(root / "runtime" / "manifest.json"),
        "pairs": sorted(pair_records, key=lambda x: x["task"]),
        "limitations": [
            "one model seed per real task",
            "one width-only method, other 12 adaptive native methods not executed here",
            "bounded four-epoch CPU training, no source-weighted visual model",
            "no MC observer overhead, GPU proof or full OpenML / forecast / recommend matrix",
        ],
    }
    file = root / "worker_pair_receipt.json"
    file.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    path = run(args.output)
    print(path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
