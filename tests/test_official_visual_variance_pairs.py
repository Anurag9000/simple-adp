"""Original/variance worker selection must stay one-to-one across four visual families."""
from types import SimpleNamespace

import pytest

from scripts.run_official_fashion_visual_variance_pairs import (
    VISUAL_ADAPTERS, choose_pairs,
)


def _job(name, kind, seed=73, split="shared"):
    job_id = name + "-" + kind
    source_id = name + "-original"
    worker = {"learning_rate": 0.001, "seed": seed}
    if kind == "bayesian":
        worker.update({
            "counterpart_of": source_id,
            "expansion_criterion": "variance_stagnation",
            "variance_posterior": "shadow_bgd_sigma_not_exact_bayes",
        })
    return SimpleNamespace(
        model_adapter=name,
        route=SimpleNamespace(model_id="ADP-W-RR"),
        identity=SimpleNamespace(run_id=job_id, seed=seed,
                                 split_sha256=split),
        task_ref=("vision", "fashion_mnist", "fashion_mnist-v01"),
        worker_settings=worker,
    )


def _all():
    return [_job(name, kind) for name in VISUAL_ADAPTERS
            for kind in ("original", "bayesian")]


def test_four_distinct_real_visual_families_are_paired_by_job_identity():
    selected = choose_pairs(_all())
    assert set(selected) == set(VISUAL_ADAPTERS)
    assert sum(len(pair) for pair in selected.values()) == 8
    assert all(arms["original"].identity.seed == arms["bayesian"].identity.seed
               for arms in selected.values())


def test_missing_or_duplicate_visual_arm_rejected():
    jobs = _all()
    with pytest.raises(ValueError, match="incomplete"):
        choose_pairs(jobs[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        choose_pairs(jobs + [jobs[0]])


def test_wrong_source_attribution_rejected():
    jobs = _all()
    jobs[1].worker_settings["counterpart_of"] = "wrong"
    with pytest.raises(ValueError, match="source attribution"):
        choose_pairs(jobs)


def test_different_seed_or_split_rejected():
    jobs = _all()
    jobs[1].identity.seed = 74
    with pytest.raises(ValueError, match="data, seed"):
        choose_pairs(jobs)
    jobs = _all()
    jobs[1].identity.split_sha256 = "wrong"
    with pytest.raises(ValueError, match="data, seed"):
        choose_pairs(jobs)


def test_nontrigger_worker_setting_change_rejected():
    jobs = _all()
    jobs[1].worker_settings["learning_rate"] = 0.2
    with pytest.raises(ValueError, match="non-trigger"):
        choose_pairs(jobs)
