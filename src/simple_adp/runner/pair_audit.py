"""Audit complete, materialized original-versus-BGD-variance ADP plans.

This is a *static* provenance certificate. It does not claim any scientific
training happened, and it rejects sliced manifests as complete inventories.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import fields
from typing import Any

from .plan import RunIdentity


VARIANCE_SUFFIX = "__bgd_variance_stagnation"
ALLOWED_SHADOW_SETTINGS = frozenset({
    "expansion_criterion", "counterpart_of", "variance_posterior",
})
# The built-in plan.py pairing capability set at the audited source commit.
# Maintain this in lockstep until plan.py adopts a shared capability registry.
from .adapter_capabilities import SHADOW_VARIANCE_ADAPTERS as KNOWN_GROWTH_ADAPTERS


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _manifest_digest(manifest: dict[str, Any]) -> str:
    payload = json.loads(_canonical(manifest))
    payload.pop("manifest_digest", None)
    for job in payload.get("jobs", []):
        command = job.get("command") if isinstance(job, dict) else None
        if isinstance(command, list) and command and isinstance(command[0], str):
            if os.path.isabs(command[0]):
                command[0] = os.path.realpath(command[0])
    return _digest(payload)


def _identity(job: dict[str, Any]) -> dict[str, Any]:
    """Verify the persisted label against the authoritative RunIdentity codec."""
    value = job.get("identity")
    if not isinstance(value, dict) or not isinstance(value.get("run_id"), str) or not value["run_id"]:
        raise ValueError("job lacks a complete run identity")
    required = {field.name for field in fields(RunIdentity)}
    if set(value) != required | {"run_id"}:
        raise ValueError("job run identity has missing or unexpected fields")
    try:
        regenerated = RunIdentity(**{key: value[key] for key in required}).run_id
    except (TypeError, ValueError) as exc:
        raise ValueError("job run identity cannot be reconstructed") from exc
    if regenerated != value["run_id"]:
        raise ValueError("job run_id does not match canonical RunIdentity fields")
    route = job.get("route")
    if not isinstance(route, dict) or not isinstance(route.get("route_label"), str):
        raise ValueError("job route lacks its original route_label")
    criterion = (job.get("worker_settings") or {}).get("counterpart_of")
    expected_route = route["route_label"] + (VARIANCE_SUFFIX if criterion is not None else "")
    if value["model_route"] != expected_route:
        raise ValueError("job identity model_route differs from executed route label")
    return value


def _settings(job: dict[str, Any]) -> dict[str, Any]:
    value = job.get("worker_settings")
    if not isinstance(value, dict):
        raise ValueError("job has no worker_settings object")
    return value


def _fixed(job: dict[str, Any]) -> bool:
    route = job.get("route")
    if not isinstance(route, dict):
        raise ValueError("job has no model route")
    return bool(
        route.get("model_id") == "MLP-Fixed" or route.get("screening_only")
        or route.get("suite_group") in {"vanilla", "screening"}
    )


def _native_loss_growth(job: dict[str, Any]) -> bool:
    if _fixed(job):
        return False
    if job.get("model_adapter") not in KNOWN_GROWTH_ADAPTERS:
        return False
    return _settings(job).get("expansion_criterion", "loss") == "loss"


def _scientific_payload(job: dict[str, Any]) -> dict[str, Any]:
    identity = dict(_identity(job))
    identity.pop("run_id", None)
    identity.pop("model_route", None)
    settings = dict(_settings(job))
    for key in ALLOWED_SHADOW_SETTINGS:
        settings.pop(key, None)
    return {
        "identity": identity,
        "task_ref": job.get("task_ref"),
        "model_adapter": job.get("model_adapter"),
        "route": job.get("route"),
        "split_file": job.get("split_file"),
        "worker_settings": settings,
        "device_capable": job.get("device_capable", True),
    }


def _pair_check(source: dict[str, Any], alternative: dict[str, Any]) -> dict[str, Any]:
    a = _identity(source)
    b = _identity(alternative)
    original_id = a["run_id"]
    other_id = b["run_id"]
    if other_id == original_id:
        raise ValueError(f"Bayesian counterpart reused original job ID: {original_id}")
    if b.get("model_route") != str(a.get("model_route")) + VARIANCE_SUFFIX:
        raise ValueError(f"Bayesian counterpart route identity differs from source: {original_id}")
    common_keys = (set(a) | set(b)) - {"model_route", "run_id"}
    changes = {key: [a.get(key), b.get(key)] for key in common_keys if _canonical(a.get(key)) != _canonical(b.get(key))}
    if changes:
        raise ValueError(f"Bayesian counterpart changed original identity fields: {original_id}: {changes}")
    aw = _settings(source)
    bw = _settings(alternative)
    if bw.get("counterpart_of") != original_id or bw.get("expansion_criterion") != "variance_stagnation":
        raise ValueError(f"invalid variance counterpart attribution or criterion: {original_id}")
    if bw.get("variance_posterior") != "shadow_bgd_sigma_not_exact_bayes":
        raise ValueError(f"missing exact shadow-observer provenance: {original_id}")
    if any(key in aw for key in ALLOWED_SHADOW_SETTINGS):
        raise ValueError(f"original unexpectedly carries Bayesian-only worker settings: {original_id}")
    if set(bw) != set(aw) | ALLOWED_SHADOW_SETTINGS:
        raise ValueError(f"Bayesian counterpart changed worker argument keys: {original_id}")
    if _canonical(_scientific_payload(source)) != _canonical(_scientific_payload(alternative)):
        raise ValueError(f"Bayesian counterpart changed scientific inputs: {original_id}")
    if source.get("result_dir") == alternative.get("result_dir"):
        raise ValueError(f"original and alternative share a result directory: {original_id}")
    acmd = source.get("command")
    bcmd = alternative.get("command")
    if not (isinstance(acmd, list) and isinstance(bcmd, list)
            and len(acmd) >= 5 and len(bcmd) == len(acmd)
            and acmd[:-1] == bcmd[:-1] and acmd[-2] == "--spec"
            and acmd[-1] != bcmd[-1]):
        raise ValueError(f"original/counterpart execution command differs unexpectedly: {original_id}")
    source_fingerprint = _digest(_scientific_payload(source))
    pair_id = _digest({"original_job_id": original_id, "bayesian_job_id": other_id})
    return {
        "pair_id": pair_id,
        "original_job_id": original_id,
        "bayesian_job_id": other_id,
        "same_science_digest": source_fingerprint,
        "source_trigger_kind": "loss",
        "alternative_trigger_kind": "posterior_variance_stagnation",
        "comparison_cohort": "LOSS_VERSUS_SHADOW_BGD_VARIANCE",
        "allowed_differences": [
            "identity.run_id", "identity.model_route", "result_dir", "command[-1]",
            "worker_settings.expansion_criterion", "worker_settings.counterpart_of",
            "worker_settings.variance_posterior",
        ],
        "provenance": {
            "provider": a.get("provider"),
            "dataset_suite": a.get("suite"),
            "task_id": a.get("task_id"),
            "split_sha256": a.get("split_sha256"),
            "seed": a.get("seed"),
            "model_adapter": source.get("model_adapter"),
            "model_route": a.get("model_route"),
            "data_content_sha256": (source.get("task_ref") or {}).get("metadata", {}).get("content_sha256"),
        },
        "audit_status": "STATIC_PAIR_AUDITED",
        "original_execution_status": "NOT_EXECUTED",
        "bayesian_execution_status": "NOT_EXECUTED",
    }


def audit_manifest(manifest: dict[str, Any], *, require_full: bool = True,
                   source_commit: str | None = None) -> dict[str, Any]:
    """Verify original/variance pairing without inventing experimental runs."""
    if not isinstance(manifest, dict) or not isinstance(manifest.get("jobs"), list):
        raise ValueError("expected SuitePlan manifest with concrete jobs")
    jobs = manifest["jobs"]
    if manifest.get("manifest_digest") is None:
        raise ValueError("manifest digest is missing; cannot audit unbound job specifications")
    if manifest["manifest_digest"] != _manifest_digest(manifest):
        raise ValueError("manifest digest mismatch")
    if manifest.get("job_count") != len(jobs):
        raise ValueError("manifest job count does not match concrete jobs")
    by_id = {}
    seen_result_dirs = set()
    seen_spec_paths = set()
    for job in jobs:
        job_id = _identity(job)["run_id"]
        if job_id in by_id:
            raise ValueError(f"duplicate job identity: {job_id}")
        by_id[job_id] = job
        if _identity(job).get("run_signature") != manifest.get("run_signature"):
            raise ValueError(f"job has incompatible plan run signature: {job_id}")
        result_dir = job.get("result_dir")
        cmd = job.get("command")
        if not isinstance(result_dir, str) or not result_dir or result_dir in seen_result_dirs:
            raise ValueError(f"duplicate or absent result directory: {job_id}")
        if not isinstance(cmd, list) or not cmd or not isinstance(cmd[-1], str) or cmd[-1] in seen_spec_paths:
            raise ValueError(f"duplicate or absent worker spec command path: {job_id}")
        seen_result_dirs.add(result_dir)
        seen_spec_paths.add(cmd[-1])
    originals = {key: job for key, job in by_id.items() if "counterpart_of" not in _settings(job)}
    alternatives = {key: job for key, job in by_id.items() if "counterpart_of" in _settings(job)}
    pairs = []
    parent_counts = Counter()
    for alt in alternatives.values():
        parent_id = _settings(alt)["counterpart_of"]
        if not isinstance(parent_id, str) or parent_id not in originals:
            raise ValueError(f"Bayesian job has missing/orphan original: {parent_id}")
        if not _native_loss_growth(originals[parent_id]):
            raise ValueError(f"Bayesian job targets ineligible original: {parent_id}")
        parent_counts[parent_id] += 1
        if parent_counts[parent_id] > 1:
            raise ValueError(f"multiple Bayesian counterparts for one source: {parent_id}")
        pairs.append(_pair_check(originals[parent_id], alt))
    eligible = {name for name, job in originals.items() if _native_loss_growth(job)}
    unpaired = sorted(eligible - set(parent_counts))
    settings = manifest.get("runner_settings") or {}
    sliced = bool(settings.get("job_start_index", 0) or settings.get("job_limit", 0))
    full_audit = manifest.get("variance_pair_audit") or {}
    promised = full_audit.get("eligible_original_growth_runs")
    if promised is not None and full_audit.get("matched_variance_counterparts") != promised:
        raise ValueError("full-plan source pair-count claims disagree")
    if promised is not None and len(eligible) != promised:
        sliced = True
    if promised is not None and len(pairs) != promised and not unpaired:
        sliced = True
    if require_full and sliced:
        raise ValueError("selected manifest does not represent the full unsliced source plan")
    if require_full and unpaired:
        raise ValueError(f"{len(unpaired)} eligible original growth jobs lack Bayesian counterparts")
    controls = {"fixed": 0, "nonloss": 0, "unregistered_adapters": 0}
    excluded = []
    for job_id, job in originals.items():
        if _fixed(job):
            kind = "FIXED_NO_GROWTH"
            controls["fixed"] += 1
        elif _settings(job).get("expansion_criterion", "loss") != "loss":
            kind = "NONLOSS_GROWTH_CONTROL"
            controls["nonloss"] += 1
        elif not _native_loss_growth(job):
            kind = "UNSUPPORTED_OR_INCOMPATIBLE"
            controls["unregistered_adapters"] += 1
        else:
            continue
        excluded.append({"job_id": job_id, "classification": kind,
                         "model_adapter": job.get("model_adapter"),
                         "exclusion_reason": (
                             "source route is fixed/screening" if kind == "FIXED_NO_GROWTH" else
                             "source route uses non-loss proposal criterion" if kind == "NONLOSS_GROWTH_CONTROL" else
                             "adapter is not registered for built-in shadow-growth pairing"
                         )})
    pairs.sort(key=lambda record: record["original_job_id"])
    excluded.sort(key=lambda record: record["job_id"])
    return {
        "schema_version": 1,
        "source_commit": source_commit or "NOT_PROVIDED",
        "source_run_signature": manifest.get("run_signature"),
        "inventory_status": ("STATIC_PAIR_AUDITED" if not sliced and not unpaired
                             else "PARTIAL_SELECTED_SLICE" if sliced else "FAILED_UNPAIRED"),
        "summary": {
            "original_jobs_selected": len(originals),
            "eligible_original_loss_growth_jobs_selected": len(eligible),
            "bayesian_alternative_jobs_selected": len(alternatives),
            "matched_pairs_selected": len(pairs),
            "primary_execution_records_selected": len(eligible) + len(alternatives),
            "unpaired_eligible_original_ids": unpaired,
            "fixed_controls_selected": controls["fixed"],
            "other_trigger_controls_selected": controls["nonloss"],
            "unregistered_adapter_jobs_selected": controls["unregistered_adapters"],
            "is_sliced": sliced,
            "full_source_eligible_claim": promised,
            "real_training_pairs_proved": 0,
        },
        "pairs": pairs,
        "excluded_sources": excluded,
    }
