import copy

import pytest

from simple_adp.runner.pair_audit import audit_manifest, _manifest_digest
from simple_adp.runner.plan import RunIdentity


def _canonical_id(identity):
    return RunIdentity(**{key: value for key, value in identity.items()
                          if key != "run_id"}).run_id


def _jobs():
    base_identity = {
        "provider": "openml", "suite": "tabzilla", "task_id": "7",
        "split_repeat": 0, "fold": 0, "sample": 0, "suite_repeat": 1,
        "model_route": "ADP-W-All", "seed": 12,
        "split_sha256": "sha-split", "run_signature": "runsig",
    }
    base_identity["run_id"] = _canonical_id(base_identity)
    source_id = base_identity["run_id"]
    base = {
        "identity": base_identity,
        "task_ref": {"provider": "openml", "suite": "tabzilla", "task_id": "7",
                     "metadata": {"content_sha256": "sha-data"}},
        "model_adapter": "simple_adp",
        "route": {"model_id": "ADP-W-All", "suite_group": "width_only",
                  "route_label": "ADP-W-All"},
        "split_file": "/splits/id.npz",
        "result_dir": f"/results/{source_id}",
        "worker_settings": {"seed": 12, "learning_rate": 0.001, "max_width": 10},
        "command": ["python", "-m", "simple_adp.runner.worker", "--spec",
                    f"/jobs/{source_id}/run_spec.json"],
        "device_capable": True,
    }
    alt = copy.deepcopy(base)
    alt["identity"]["model_route"] += "__bgd_variance_stagnation"
    alt["identity"]["run_id"] = _canonical_id(alt["identity"])
    alt_id = alt["identity"]["run_id"]
    alt["result_dir"] = f"/results/{alt_id}"
    alt["worker_settings"].update({
        "counterpart_of": source_id, "expansion_criterion": "variance_stagnation",
        "variance_posterior": "shadow_bgd_sigma_not_exact_bayes",
    })
    alt["command"][-1] = f"/jobs/{alt_id}/run_spec.json"
    fixed = copy.deepcopy(base)
    fixed["identity"]["model_route"] = "MLP-Fixed"
    fixed["identity"]["run_id"] = _canonical_id(fixed["identity"])
    fixed_id = fixed["identity"]["run_id"]
    fixed["route"] = {"model_id": "MLP-Fixed", "suite_group": "vanilla",
                      "route_label": "MLP-Fixed"}
    fixed["result_dir"] = f"/results/{fixed_id}"
    fixed["command"][-1] = f"/jobs/{fixed_id}/run_spec.json"
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
    second["identity"]["seed"] += 1
    second["identity"]["run_id"] = _canonical_id(second["identity"])
    second["result_dir"] = "/results/" + second["identity"]["run_id"]
    second["command"][-1] = "/jobs/" + second["identity"]["run_id"] + "/run_spec.json"
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
    # Preserve internal ID/path integrity so this tests the manifest-level
    # signature check rather than the earlier canonical-ID defense.
    alt["identity"]["run_id"] = _canonical_id(alt["identity"])
    alt["result_dir"] = "/results/" + alt["identity"]["run_id"]
    alt["command"][-1] = "/jobs/" + alt["identity"]["run_id"] + "/run_spec.json"
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


@pytest.mark.parametrize("field", ["run_id", "model_route", "suite_repeat", "unexpected"])
def test_forged_or_incomplete_run_identity_fails_even_if_manifest_digest_is_recomputed(field):
    source, alt, _ = _jobs()
    if field == "run_id":
        source["identity"]["run_id"] = "forged"
    elif field == "model_route":
        source["identity"]["model_route"] = "ADP-W-RR"
        source["identity"]["run_id"] = _canonical_id(source["identity"])
    elif field == "suite_repeat":
        del source["identity"]["suite_repeat"]
    else:
        source["identity"]["unexpected"] = "hidden"
    with pytest.raises(ValueError, match="run_id|route label|identity"):
        audit_manifest(_manifest([source, alt]))


def test_forged_result_and_spec_namespaces_are_rejected_independently():
    original, alternative, _ = _jobs()
    alternative["result_dir"] = "/results/unrelated/collision"
    with pytest.raises(ValueError, match="result directory"):
        audit_manifest(_manifest([original, alternative]))

    original, alternative, _ = _jobs()
    alternative["command"][-1] = "/jobs/unrelated/run_spec.json"
    with pytest.raises(ValueError, match="worker specification path"):
        audit_manifest(_manifest([original, alternative]))
