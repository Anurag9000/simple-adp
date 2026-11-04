import copy

import pytest

from simple_adp.runner.pair_audit import audit_manifest, _manifest_digest


def _jobs():
    base = {
        "identity": {
            "run_id": "original", "provider": "openml", "suite": "tabzilla",
            "task_id": "7", "seed": 12, "split_sha256": "sha-split",
            "run_signature": "runsig", "model_route": "ADP-W-All",
        },
        "task_ref": {"provider": "openml", "suite": "tabzilla", "task_id": "7",
                     "metadata": {"content_sha256": "sha-data"}},
        "model_adapter": "simple_adp",
        "route": {"model_id": "ADP-W-All", "suite_group": "width_only"},
        "split_file": "/splits/id.npz",
        "result_dir": "/results/original",
        "worker_settings": {"seed": 12, "learning_rate": 0.001, "max_width": 10},
        "command": ["python", "-m", "simple_adp.runner.worker", "--spec", "/jobs/original/run_spec.json"],
        "device_capable": True,
    }
    alt = copy.deepcopy(base)
    alt["identity"]["run_id"] = "bayesian"
    alt["identity"]["model_route"] += "__bgd_variance_stagnation"
    alt["result_dir"] = "/results/bayesian"
    alt["worker_settings"].update({
        "counterpart_of": "original", "expansion_criterion": "variance_stagnation",
        "variance_posterior": "shadow_bgd_sigma_not_exact_bayes",
    })
    alt["command"][-1] = "/jobs/bayesian/run_spec.json"
    fixed = copy.deepcopy(base)
    fixed["identity"]["run_id"] = "fixed"
    fixed["identity"]["model_route"] = "MLP-Fixed"
    fixed["route"] = {"model_id": "MLP-Fixed", "suite_group": "vanilla"}
    fixed["result_dir"] = "/results/fixed"
    fixed["command"][-1] = "/jobs/fixed/run_spec.json"
    return base, alt, fixed


def _manifest(jobs, claimed=1):
    value = {
        "jobs": jobs, "job_count": len(jobs), "run_signature": "runsig",
        "runner_settings": {"job_start_index": 0, "job_limit": 0},
        "variance_pair_audit": {"eligible_original_growth_runs": claimed,
                                "matched_variance_counterparts": claimed},
    }
    value["manifest_digest"] = _manifest_digest(value)
    return value


def test_exact_one_to_one_with_fixed_controls():
    original, alternative, fixed = _jobs()
    report = audit_manifest(_manifest([original, alternative, fixed]), source_commit="a" * 40)
    assert report["inventory_status"] == "STATIC_PAIR_AUDITED"
    assert report["summary"]["matched_pairs_selected"] == 1
    assert report["summary"]["primary_execution_records_selected"] == 2
    assert report["summary"]["fixed_controls_selected"] == 1
    assert report["summary"]["real_training_pairs_proved"] == 0
    assert report["pairs"][0]["provenance"]["data_content_sha256"] == "sha-data"


@pytest.mark.parametrize("change", [
    lambda alt: alt["identity"].__setitem__("split_sha256", "wrong"),
    lambda alt: alt["worker_settings"].__setitem__("max_width", 99),
    lambda alt: alt["worker_settings"].__setitem__("surprise_flag", True),
    lambda alt: alt.__setitem__("split_file", "/not-the-same-split.npz"),
    lambda alt: alt.__setitem__("result_dir", "/results/original"),
    lambda alt: alt["command"].__setitem__(2, "another.worker"),
    lambda alt: alt["identity"].__setitem__("model_route", "wrong-alt-label"),
])
def test_scientific_or_execution_mismatch_is_rejected(change):
    base, alt, _ = _jobs()
    change(alt)
    with pytest.raises(ValueError):
        audit_manifest(_manifest([base, alt]))


def test_duplicate_child_and_orphan_rejected():
    original, alt, _ = _jobs()
    second = copy.deepcopy(alt)
    second["identity"]["run_id"] = "other-bayesian"
    second["result_dir"] = "/results/other-bayesian"
    second["command"][-1] = "/jobs/other-bayesian/run_spec.json"
    with pytest.raises(ValueError, match="multiple Bayesian counterparts"):
        audit_manifest(_manifest([original, alt, second]))
    orphan = copy.deepcopy(alt)
    orphan["worker_settings"]["counterpart_of"] = "unknown"
    with pytest.raises(ValueError, match="orphan"):
        audit_manifest(_manifest([original, orphan]))


def test_unpaired_primary_job_fails_but_unregistered_is_not_falsely_paired():
    original, _, _ = _jobs()
    with pytest.raises(ValueError, match="lack Bayesian counterparts"):
        audit_manifest(_manifest([original]))
    external = copy.deepcopy(original)
    external["model_adapter"] = "my_unregistered_worker"
    report = audit_manifest(_manifest([external], claimed=0))
    assert report["summary"]["unregistered_adapter_jobs_selected"] == 1
    assert report["summary"]["matched_pairs_selected"] == 0


def test_sliced_manifest_is_not_mistaken_for_full_science_inventory():
    original, alt, _ = _jobs()
    value = _manifest([original], claimed=1)
    value["runner_settings"]["job_limit"] = 1
    value["manifest_digest"] = _manifest_digest(value)
    with pytest.raises(ValueError, match="unsliced"):
        audit_manifest(value)
    report = audit_manifest(value, require_full=False)
    assert report["inventory_status"] == "PARTIAL_SELECTED_SLICE"


def test_tampered_manifest_and_duplicate_job_id_rejected():
    original, alt, _ = _jobs()
    value = _manifest([original, alt])
    value["jobs"][1]["worker_settings"]["seed"] = 13
    with pytest.raises(ValueError, match="digest"):
        audit_manifest(value)
    second = copy.deepcopy(original)
    with pytest.raises(ValueError, match="duplicate job identity"):
        audit_manifest(_manifest([original, second]))


def test_missing_manifest_digest_rejected():
    source, alt, _ = _jobs()
    manifest = _manifest([source, alt])
    del manifest["manifest_digest"]
    with pytest.raises(ValueError, match="digest is missing"):
        audit_manifest(manifest)


def test_result_collision_and_changed_plan_signature_rejected():
    source, alt, _ = _jobs()
    alt["result_dir"] = source["result_dir"]
    with pytest.raises(ValueError, match="result directory"):
        audit_manifest(_manifest([source, alt]))
    source, alt, _ = _jobs()
    alt["identity"]["run_signature"] = "other"
    with pytest.raises(ValueError, match="run signature"):
        audit_manifest(_manifest([source, alt]))


def test_auditor_cli_outputs_inspectable_receipt(tmp_path, monkeypatch):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path
    source, alt, _ = _jobs()
    saved_manifest = tmp_path / "manifest.json"
    saved_manifest.write_text(json.dumps(_manifest([source, alt])))
    receipt = tmp_path / "pair_inventory.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_bayesian_pairs.py"
    env = dict(os.environ)
    repo_src = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = str(repo_src)
    completed = subprocess.run(
        [sys.executable, str(script), str(saved_manifest), "--output", str(receipt),
         "--source-commit", "a" * 40],
        capture_output=True, text=True, env=env, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    data = json.loads(receipt.read_text())
    assert data["summary"]["matched_pairs_selected"] == 1
    assert data["source_commit"] == "a" * 40
