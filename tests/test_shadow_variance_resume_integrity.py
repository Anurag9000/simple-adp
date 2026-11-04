"""Fail-closed exact-resume contracts for auxiliary posterior state."""
import copy

import pytest
import torch

from simple_adp.shadow_bgd_variance import ShadowBGDVariance


def _valid():
    model = torch.nn.Linear(3, 2)
    observer = ShadowBGDVariance(seed=4)
    x = torch.randn(8, 3)
    y = torch.randint(0, 2, (8,))
    assert observer.observe(model, x, y) is False
    return model, observer, observer.state_dict()


@pytest.mark.parametrize("invalid", ["nan", "zero", "negative", "inf", "wrong_shape", "missing_name", "bad_rng", "bad_counts", "bad_contraction", "missing_rng", "wrong_device_rng", "bool_contraction"])
def test_corrupt_shadow_resume_fails_closed_and_atomically(invalid):
    model, observer, original = _valid()
    state = copy.deepcopy(original)
    if invalid == "nan":
        state["sigma"]["weight"].view(-1)[0] = float("nan")
    elif invalid == "zero":
        state["sigma"]["weight"].view(-1)[0] = 0
    elif invalid == "negative":
        state["sigma"]["weight"].view(-1)[0] = -1
    elif invalid == "inf":
        state["sigma"]["weight"].view(-1)[0] = float("inf")
    elif invalid == "wrong_shape":
        state["sigma"]["weight"] = state["sigma"]["weight"][:1]
    elif invalid == "missing_name":
        del state["sigma"]["bias"]
    elif invalid == "bad_rng":
        state["rng"]["cpu"] = torch.zeros(3, dtype=torch.float32)
    elif invalid == "missing_rng":
        state["rng"].clear()
    elif invalid == "wrong_device_rng":
        state["rng"]["cuda:0"] = state["rng"].pop("cpu")
    elif invalid == "bool_contraction":
        state["contraction"] = False
    elif invalid == "bad_counts":
        state["streak"] = state["observations"] + 1
    elif invalid == "bad_contraction":
        state["contraction"] = float("nan")
    before = observer.state_dict()
    with pytest.raises(ValueError):
        observer.load_state_dict(state, model=model)
    after = observer.state_dict()
    assert after["config"] == before["config"]
    assert after["streak"] == before["streak"]
    assert after["observations"] == before["observations"]
    for name in before["sigma"]:
        assert torch.equal(after["sigma"][name], before["sigma"][name])


def test_resuming_posterior_for_another_model_is_not_growth_migration():
    model, observer, state = _valid()
    larger = torch.nn.Linear(3, 3)
    with pytest.raises(ValueError, match="geometry"):
        observer.load_state_dict(state, model=larger)
    restored = ShadowBGDVariance(seed=4)
    restored.load_state_dict(state, model=model)
    for name in state["sigma"]:
        assert torch.equal(restored.sigma[name], state["sigma"][name])


def test_before_first_batch_empty_posterior_checkpoint_is_restorable():
    model = torch.nn.Linear(3, 2)
    observer = ShadowBGDVariance(seed=4)
    restored = ShadowBGDVariance(seed=4)
    restored.load_state_dict(observer.state_dict(), model=model)
    assert restored.observations == 0
    assert restored.sigma == {}
