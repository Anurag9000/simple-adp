"""Budget-pair audit must never turn a tiny smoke into a science claim."""
import copy

import pytest

from simple_adp.visual_budget_audit import audit_visual_worker_pairs


def pair(adapter="vision_inner", *, fixed_steps=4, grown_steps=8):
    common = {
        "adapter": adapter, "status": "completed",
        "dataset_id": "the-same-dataset", "seed": 0,
        "split_sha256": "the-same-split", "architecture_id": "V02",
        "growth_site": "adapter:block1",
    }
    return [
        {
            **common, "method": "MLP-Fixed", "candidate_stages_executed": 1,
            "training_budget": {"complete": True, "optimizer_steps": fixed_steps},
        },
        {
            **common, "method": "ADP-W-RR", "candidate_stages_executed": 2,
            "training_budget": {"complete": True, "optimizer_steps": grown_steps},
        },
    ]


def test_mismatched_candidate_work_is_flagged_without_science_claim():
    audit = audit_visual_worker_pairs(pair(), expected_adapters=["vision_inner"])
    assert audit["expected_pairs"] == 1
    assert audit["mismatched_optimizer_step_pairs"] == 1
    assert audit["same_optimizer_step_pairs"] == 0
    assert audit["unknown_optimizer_step_pairs"] == 0
    record = audit["paired_budget_evidence"][0]
    assert record["identity_status"] == "paired-input-identity"
    assert record["same_optimizer_step_count"] is False
    assert record["growth_to_fixed_step_ratio"] == 2.0
    assert record["full_training_flops_matched"] is None
    assert audit["scientific_performance_conclusion"] is None


def test_equal_optimizer_steps_are_not_assumed_equal_training_flops():
    audit = audit_visual_worker_pairs(pair(fixed_steps=8, grown_steps=8), expected_adapters=["vision_inner"])
    assert audit["same_optimizer_step_pairs"] == 1
    assert audit["paired_budget_evidence"][0]["same_optimizer_step_count"] is True
    assert audit["full_training_flops_matched"] is None


def test_original_v01_without_native_budget_is_unknown_not_zero_cost():
    records = pair("vision_adp")
    for row in records:
        row["training_budget"] = None
    audit = audit_visual_worker_pairs(records, expected_adapters=["vision_adp"])
    assert audit["unknown_optimizer_step_pairs"] == 1
    row = audit["paired_budget_evidence"][0]
    assert row["fixed_optimizer_steps"] is None
    assert row["growth_to_fixed_step_ratio"] is None
    assert row["same_optimizer_step_count"] is None


def test_partial_checkpoint_budget_is_undetermined():
    records = pair()
    records[1]["training_budget"]["complete"] = False
    audit = audit_visual_worker_pairs(records, expected_adapters=["vision_inner"])
    assert audit["unknown_optimizer_step_pairs"] == 1


def test_failed_or_cross_split_worker_never_qualifies_for_step_match():
    records = pair(fixed_steps=4, grown_steps=4)
    records[0]["split_sha256"] = "wrong"
    audit = audit_visual_worker_pairs(records, expected_adapters=["vision_inner"])
    assert audit["unknown_optimizer_step_pairs"] == 1
    assert audit["paired_budget_evidence"][0]["identity_status"] == "split-mismatch"

    records = pair(fixed_steps=4, grown_steps=4)
    records[1]["status"] = "failed"
    audit = audit_visual_worker_pairs(records, expected_adapters=["vision_inner"])
    assert audit["unknown_optimizer_step_pairs"] == 1


@pytest.mark.parametrize("modifier", [
    lambda rows: rows.pop(),
    lambda rows: rows.append(copy.deepcopy(rows[0])),
    lambda rows: rows[0].update({"method": "not-an-adp-policy"}),
    lambda rows: rows[0].update({"adapter": "unexpected"}),
])
def test_invalid_pair_sets_rejected(modifier):
    rows = pair()
    modifier(rows)
    with pytest.raises(ValueError):
        audit_visual_worker_pairs(rows, expected_adapters=["vision_inner"])


def test_distinct_adapters_keep_deterministic_report_order():
    first = pair("a")
    second = pair("b")
    result = audit_visual_worker_pairs(second + first, expected_adapters=["a", "b"])
    assert [x["adapter"] for x in result["paired_budget_evidence"]] == ["a", "b"]
    assert result["expected_pairs"] == 2


def test_cross_dataset_or_seed_work_must_not_be_compared():
    records = pair(fixed_steps=8, grown_steps=8)
    records[1]["dataset_id"] = "different-source"
    out = audit_visual_worker_pairs(records, expected_adapters=["vision_inner"])
    assert out["unknown_optimizer_step_pairs"] == 1
    assert out["paired_budget_evidence"][0]["identity_status"] == "dataset-mismatch"
    records = pair(fixed_steps=8, grown_steps=8)
    records[1]["seed"] = 1
    out = audit_visual_worker_pairs(records, expected_adapters=["vision_inner"])
    assert out["unknown_optimizer_step_pairs"] == 1
    assert out["paired_budget_evidence"][0]["identity_status"] == "seed-mismatch"
    records = pair()
    records[0].pop("dataset_id")
    out = audit_visual_worker_pairs(records, expected_adapters=["vision_inner"])
    assert out["unknown_optimizer_step_pairs"] == 1
    assert out["paired_budget_evidence"][0]["identity_status"] == "dataset-mismatch"


def test_multiseed_audits_are_independent_and_refuse_cross_seed_pairing():
    from simple_adp.visual_budget_audit import audit_visual_multiseed_worker_pairs

    first = pair("a", fixed_steps=4, grown_steps=8)
    second = pair("a", fixed_steps=4, grown_steps=4)
    for row in first:
        row["seed"] = 401
    for row in second:
        row["seed"] = 991
    combined = second + first  # Deliberately reverse source order.
    audit = audit_visual_multiseed_worker_pairs(
        combined, expected_adapters=["a"], expected_seeds=[401, 991],
    )
    assert audit["format"] == "simple-adp-visual-fixed-rr-multiseed-budget-audit-v1"
    assert audit["seed_count"] == 2
    assert audit["total_expected_pairs"] == 2
    assert audit["same_optimizer_step_pairs"] == 1
    assert audit["mismatched_optimizer_step_pairs"] == 1
    assert audit["unknown_optimizer_step_pairs"] == 0
    assert [x["run_seed"] for x in audit["per_seed_budget_audits"]] == [401, 991]
    assert audit["full_training_flops_matched"] is None
    assert audit["scientific_performance_conclusion"] is None


@pytest.mark.parametrize("case", ["duplicate", "missing", "stray", "cross_seed"])
def test_multiseed_incomplete_or_misidentified_pairs_are_rejected(case):
    from simple_adp.visual_budget_audit import audit_visual_multiseed_worker_pairs

    one, two = pair("a"), pair("a")
    for row in one:
        row["seed"] = 7
    for row in two:
        row["seed"] = 8
    records = one + two
    if case == "duplicate":
        records.append(copy.deepcopy(records[0]))
    elif case == "missing":
        records.pop()
    elif case == "stray":
        records[-1]["seed"] = 9
    else:
        records[-1]["seed"] = 7
    with pytest.raises(ValueError):
        audit_visual_multiseed_worker_pairs(
            records, expected_adapters=["a"], expected_seeds=[7, 8],
        )


@pytest.mark.parametrize("bad_seeds", [[1, 1], [], [True, 2]])
def test_multiseed_nonunique_or_nonnumeric_seed_contract_rejected(bad_seeds):
    from simple_adp.visual_budget_audit import audit_visual_multiseed_worker_pairs

    with pytest.raises(ValueError):
        audit_visual_multiseed_worker_pairs(
            [], expected_adapters=["a"], expected_seeds=bad_seeds,
        )
