"""Complete visual shadow posterior evidence is stage-local and decision-neutral."""
from types import SimpleNamespace

import pytest

from simple_adp.visual_search import _shadow_best_stage_evidence


def _stage(*, observations=7, streak=0, contraction=0.05, ready=False):
    return SimpleNamespace(
        checkpoint_state={
            "shadow_bgd_variance": {
                "schema": 1,
                "config": (0.5, 2, 0.0001, 3, 12),
                "observations": observations,
                "streak": streak,
                "contraction": contraction,
            },
        },
        variance_stagnated=ready,
    )


def test_real_stage_without_streak_is_not_growth_ready():
    evidence = _shadow_best_stage_evidence(_stage())
    assert evidence["status"] == "BEST_STAGE_OBSERVED"
    assert not evidence["gate_ready"]
    assert evidence["observation_count"] == 7
    assert evidence["lower_bound_shadow_forward_passes"] == 14


def test_three_consecutive_low_contractions_unlock_growth():
    evidence = _shadow_best_stage_evidence(
        _stage(observations=3, streak=3, contraction=0.0, ready=True)
    )
    assert evidence["gate_ready"]
    assert evidence["required_streak"] == 3


def test_rebuilt_evidence_cannot_disagree_with_actual_growth_decision():
    with pytest.raises(ValueError, match="disagrees"):
        _shadow_best_stage_evidence(_stage(ready=True))


def test_missing_legacy_stage_is_marked_incomplete():
    evidence = _shadow_best_stage_evidence(
        SimpleNamespace(checkpoint_state={}, variance_stagnated=False)
    )
    assert evidence == {"status": "UNAVAILABLE", "gate_ready": False}


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_invalid_posterior_contraction_is_rejected(bad):
    with pytest.raises(ValueError, match="nonfinite"):
        _shadow_best_stage_evidence(_stage(contraction=bad))


def test_negative_finite_contraction_is_legitimate_evidence():
    # Shadow sigma can expand rather than contract on an observation;
    # rejecting all negative log contractions would corrupt actual runs.
    evidence = _shadow_best_stage_evidence(_stage(contraction=-2.0))
    assert evidence["contraction"] == -2.0
    assert not evidence["gate_ready"]
