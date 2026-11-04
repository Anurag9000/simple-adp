"""Bounded real Fashion-MNIST four-family original-vs-shadow-BGD worker proof.

Production SuitePlan and runner.worker are used unchanged. The dataset is
the official torchvision FashionMNIST source. Encoders in these four
test families are tiny frozen-*random* modules, not published pretrained
weights. One derived seed, short training and tiny splits are not science.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

from simple_adp.runner.artifacts import estimator_checkpoint_path
from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.pair_audit import audit_manifest
from simple_adp.runner.plan import SuitePlan

VISUAL_ADAPTERS = (
    "vision_inner", "vision_v03", "vision_v04", "vision_v07_token",
)
METHOD_ID = "ADP-W-RR"
TRIGGERS = ("loss", "variance_stagnation")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_pairs(runs):
    """Choose exactly one matched policy route per tested native adapter."""
    indexed = {run.identity.run_id: run for run in runs}
    selected = {}
    for job in runs:
        if job.model_adapter not in VISUAL_ADAPTERS or job.route.model_id != METHOD_ID:
            continue
        key = job.model_adapter
        if key not in selected:
            selected[key] = {}
        kind = "bayesian" if "counterpart_of" in job.worker_settings else "original"
        if kind in selected[key]:
            raise ValueError(f"duplicate visual {kind} arm for {key}")
        selected[key][kind] = job
    if set(selected) != set(VISUAL_ADAPTERS):
        raise ValueError("native visual proof is missing a required architecture family")
    for name, arms in selected.items():
        if set(arms) != {"original", "bayesian"}:
            raise ValueError(f"incomplete original/variance visual pair for {name}")
        original, bayesian = arms["original"], arms["bayesian"]
        if bayesian.worker_settings["counterpart_of"] != original.identity.run_id:
            raise ValueError(f"incorrect source attribution for {name}")
        if (original.identity.seed != bayesian.identity.seed
                or original.identity.split_sha256 != bayesian.identity.split_sha256
                or original.task_ref != bayesian.task_ref
                or original.worker_settings.get("expansion_criterion", "loss") != TRIGGERS[0]
                or bayesian.worker_settings.get("expansion_criterion") != TRIGGERS[1]):
            raise ValueError(f"visual pair changes data, seed or original trigger: {name}")
        source_settings = dict(original.worker_settings)
        alt_settings = dict(bayesian.worker_settings)
        for key in ("counterpart_of", "expansion_criterion", "variance_posterior"):
            alt_settings.pop(key, None)
        if alt_settings != source_settings:
            raise ValueError(f"visual paired treatment changes non-trigger settings: {name}")
        if original.identity.run_id not in indexed or bayesian.identity.run_id not in indexed:
            raise ValueError("visual pair includes a job outside its full source plan")
    return selected


def validate_completed_job(job, *, run_signature):
    """Load result and require real completed model/checkpoint artifacts."""
    path = Path(job.result_dir) / "result.json"
    if not path.is_file():
        raise ValueError(f"missing persisted real visual worker result: {job.identity.run_id}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    alt = "counterpart_of" in job.worker_settings
    trigger = TRIGGERS[1] if alt else TRIGGERS[0]
    if (raw.get("status") != "completed" or raw.get("job_id") != job.identity.run_id
            or raw.get("run_signature") != run_signature
            or raw.get("split_sha256") != job.identity.split_sha256
            or raw.get("expansion_criterion") != trigger
            or raw.get("task_family") != "vision"
            or raw.get("model_id") != METHOD_ID):
        raise ValueError("real visual worker result differs from signed source plan")
    metrics = raw.get("test_metrics")
    if not isinstance(metrics, dict):
        raise ValueError("visual worker has no true held-out test metrics")
    for key in ("accuracy", "macro_f1", "top_label_ece_10", "multiclass_brier"):
        value = metrics.get(key)
        if type(value) not in {int, float} or not math.isfinite(value):
            raise ValueError(f"visual worker has nonfinite or missing {key}")
    bundle = raw.get("artifact_path")
    if not isinstance(bundle, str):
        raise ValueError("visual result lacks a trained inference bundle")
    bundle = Path(bundle)
    checkpoint = estimator_checkpoint_path(job.result_dir)
    if not bundle.is_file() or not checkpoint.is_file():
        raise ValueError("visual result has no complete model/checkpoint artifacts")
    if not raw.get("frozen_sha256"):
        raise ValueError("visual result does not record its frozen source weights")
    import torch
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload = saved.get("payload", {}) if isinstance(saved, dict) else {}
    posterior_trace = payload.get("posterior_gate_trace")
    posterior_complete = payload.get("posterior_gate_trace_complete", False)
    if alt:
        if (posterior_complete is not True or not isinstance(posterior_trace, list)
                or not posterior_trace or any(
                    item.get("status") != "BEST_STAGE_OBSERVED"
                    for item in posterior_trace
                )):
            raise ValueError("visual Bayesian result lacks complete stage posterior gate evidence")
    elif posterior_trace is not None:
        raise ValueError("original visual source unexpectedly recorded a shadow posterior")
    return {
        "model_adapter": job.model_adapter,
        "arm": "bayesian" if alt else "original",
        "job_id": job.identity.run_id,
        "derived_seed": job.identity.seed,
        "content_sha256": (job.task_ref.metadata or {}).get("content_sha256"),
        "split_sha256": job.identity.split_sha256,
        "result_sha256": _sha(path),
        "model_sha256": _sha(bundle),
        "model_size_bytes": bundle.stat().st_size,
        "checkpoint_sha256": _sha(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "frozen_sha256": raw["frozen_sha256"],
        "selected_architecture": raw.get("selected_architecture"),
        "history_entries": raw.get("history_entries"),
        **({
            "posterior_gate_trace_complete": bool(posterior_complete),
            "posterior_gate_trace": posterior_trace,
        } if alt else {}),
        "heldout_metrics": {key: float(metrics[key]) for key in (
            "accuracy", "macro_f1", "top_label_ece_10", "multiclass_brier"
        )},
    }


def run(root: Path) -> Path:
    root = Path(root).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("real Fashion-MNIST paired proof requires fresh empty directory")
    root.mkdir(parents=True, exist_ok=True)
    data_root = root / "dataset"
    from torchvision.datasets import FashionMNIST

    FashionMNIST(root=str(data_root), train=True, download=True)
    FashionMNIST(root=str(data_root), train=False, download=True)
    os.environ.update({
        "SIMPLE_ADP_ENABLE_VISION": "1",
        "SIMPLE_ADP_ENABLE_V03_SOURCE": "0",
        "SIMPLE_ADP_ENABLE_V04_SOURCE": "0",
        "SIMPLE_ADP_VISION_ROOT": str(data_root),
        "SIMPLE_ADP_VISION_TRAIN_LIMIT": "40",
        "SIMPLE_ADP_VISION_VALID_LIMIT": "20",
        "SIMPLE_ADP_VISION_TEST_LIMIT": "20",
        "SIMPLE_ADP_DISABLE_GPU_ACCELERATORS": "1",
        "TRAINING_CONTROL_CPU_ONLY": "1",
        "TABULAR_BACKEND": "cpu",
        "TABULAR_SKIP_SYSTEMD_SCOPE": "1",
    })
    from simple_adp.runner.plugins import load_registries
    datasets, models = load_registries()
    settings = RunnerSettings(
        tasks=("vision",), run_hardness_screen=False,
        repeat_count=1, missing_present_task_repeats=1,
        model_seeds=(73,), generic_profiles=("faithful",),
        include_vanilla=False,
        width_to_depth_width=2, width_to_depth_depth=1,
        max_width=512, max_depth=5, max_neurons=128,
        max_candidates=1, max_epochs=1,
        batch_size=16, smoke_test=True, cpu_only=True,
        scheduler="fixed",
        post_launch_sample_delay_seconds=0,
    )
    plan = SuitePlan.build(
        settings=settings,
        results_dir=root / "results", run_root=root / "runtime",
        dataset_providers=(datasets.get("vision"),),
        dataset_suites=(("vision", "fashion_mnist"),),
        model_adapters=tuple(models.get(name) for name in VISUAL_ADAPTERS),
    )
    static = audit_manifest(plan.manifest, source_commit=os.environ.get("GITHUB_SHA"))
    if static["inventory_status"] != "STATIC_PAIR_AUDITED":
        raise RuntimeError("full visual paired plan did not pass static audit")
    chosen = choose_pairs(plan.runs)
    if len(chosen) != 4 or sum(map(len, chosen.values())) != 8:
        raise RuntimeError("bounded visual worker proof requires exactly four complete pairs")
    plan.write_manifest(root / "runtime")
    log_root = root / "logs"
    executed = {}
    for name in VISUAL_ADAPTERS:
        for arm in ("original", "bayesian"):
            job = chosen[name][arm]
            log_path = log_root / name / f"{arm}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as output:
                result = subprocess.run(
                    job.command, cwd=Path(__file__).resolve().parents[1],
                    env=os.environ.copy(), stdout=output,
                    stderr=subprocess.STDOUT, check=False, timeout=300,
                )
            if result.returncode:
                tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
                raise RuntimeError(f"visual worker {name}/{arm} failed: {tail}")
            executed[job.identity.run_id] = validate_completed_job(
                job, run_signature=plan.run_signature,
            )
    records = {}
    for name, arms in chosen.items():
        original = executed[arms["original"].identity.run_id]
        bayesian = executed[arms["bayesian"].identity.run_id]
        if (original["derived_seed"] != bayesian["derived_seed"]
                or original["content_sha256"] != bayesian["content_sha256"]
                or original["split_sha256"] != bayesian["split_sha256"]
                or original["frozen_sha256"] != bayesian["frozen_sha256"]):
            raise RuntimeError(f"visual original/Bayesian data or frozen weights differ for {name}")
        records[name] = {
            "source": original,
            "bayesian": bayesian,
            "accuracy_delta_bayesian_minus_original": (
                bayesian["heldout_metrics"]["accuracy"] - original["heldout_metrics"]["accuracy"]
            ),
        }
    science = {
        "schema": "bounded-official-fashion-visual-original-vs-variance-v1",
        "evidence_grade": "BOUNDED_REAL_VISUAL_WORKER_PAIRS_PASSED",
        "source_revision": os.environ.get("GITHUB_SHA", "UNAVAILABLE"),
        "official_source": "torchvision.datasets.FashionMNIST",
        "official_train_validation_test_rows": [40, 20, 20],
        "model_seed_inputs": [73],
        "source_weight_provenance": (
            "tiny frozen random visual source encoders, not published pretrained models"
        ),
        "pair_count": len(records),
        "worker_execution_count": len(executed),
        "manifest_sha256": _sha(root / "runtime" / "manifest.json"),
        "model_adapters": records,
        "limitations": [
            "Single model seed, one real official image dataset, tiny split and one epoch per stage",
            "V02/V03/V04/Mixer-token only; other native vision sites remain unexecuted here",
            "No published pretrained source checkpoints or full GPU execution",
            "No MC shadow overhead, matched compute or posterior calibration",
        ],
    }
    receipt = root / "four_visual_pair_receipt.json"
    receipt.write_text(json.dumps(science, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(science, sort_keys=True, indent=2), flush=True)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
