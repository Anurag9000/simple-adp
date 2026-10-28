import torch
import pytest

from simple_adp.config import SearchSettings
from simple_adp.shadow_bgd_variance import ShadowBGDVariance
from simple_adp.dedicated import DedicatedController
from simple_adp.generic_search import GenericCandidateSearch


def test_shadow_bgd_preserves_original_weights_and_gradients():
    model = torch.nn.Linear(3, 2)
    x = torch.randn(6, 3)
    y = torch.tensor([0, 1, 0, 1, 0, 1])
    old = {n: p.detach().clone() for n, p in model.named_parameters()}
    model.weight.grad = torch.ones_like(model.weight)
    old_grad = model.weight.grad.clone()
    observer = ShadowBGDVariance(seed=42, mc_samples=6)
    observer.observe(model, x, y)
    assert observer.contraction is not None
    assert torch.equal(model.weight.grad, old_grad)
    for n, p in model.named_parameters():
        assert torch.equal(p, old[n])


def test_shadow_resume_is_deterministic():
    model = torch.nn.Linear(3, 2)
    x = torch.randn(8, 3)
    y = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1])
    a = ShadowBGDVariance(seed=11, mc_samples=4)
    a.observe(model, x, y)
    b = ShadowBGDVariance(seed=11, mc_samples=4)
    b.load_state_dict(a.state_dict())
    assert a.observe(model, x, y) == b.observe(model, x, y)
    assert a.contraction == b.contraction


def test_variance_gates_are_opt_in_for_both_search_families():
    native = SearchSettings()
    alternate = SearchSettings(expansion_criterion="variance_stagnation")
    dedicated = DedicatedController(alternate, lambda *a, **kw: None)
    generic = GenericCandidateSearch(alternate, "corrected", lambda *a, **kw: None)
    assert not dedicated._growth_ready()
    assert not generic._growth_ready()
    dedicated._variance_stagnated = generic._variance_stagnated = True
    assert dedicated._growth_ready()
    assert generic._growth_ready()
    assert DedicatedController(native, lambda *a, **kw: None)._growth_ready()


def test_invalid_expansion_criterion_rejected():
    with pytest.raises(ValueError):
        SearchSettings(expansion_criterion="random_noise")
