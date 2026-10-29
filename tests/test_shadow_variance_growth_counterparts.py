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


def test_shadow_preserves_mixed_child_train_eval_modes():
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 6),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.25),
        torch.nn.Linear(6, 2),
    )
    model.train()
    model[2].eval()
    before = [child.training for child in model.modules()]
    x = torch.randn(12, 3)
    target = torch.randint(0, 2, (12,))
    monitor = ShadowBGDVariance(mc_samples=2, seed=7)
    monitor.observe(model, x, target)
    assert [child.training for child in model.modules()] == before


def test_variance_criterion_not_cut_short_by_original_loss_patience():
    from types import SimpleNamespace
    from simple_adp.config import ModelConfig
    from simple_adp.models import TabularMLP

    def simulate(controller_cls, model_config, criterion):
        visits = []
        def stage(model, arrays, **kwargs):
            del arrays, kwargs
            visits.append(tuple(model.hidden_widths))
            # No validation improvement: loss-driven greed/patience stops.
            # Posterior non-contraction continues until the third stage.
            return SimpleNamespace(
                best_score=1.0,
                best_snapshot=model.snapshot(),
                variance_stagnated=len(visits) < 3,
            )
        settings = SearchSettings(
            init_width=1, init_depth=1, max_width=8, max_depth=3,
            max_neurons=20, patience_width_exp=1, max_candidates=5,
            expansion_criterion=criterion,
        )
        controller = controller_cls(settings, stage) if controller_cls is DedicatedController else (
            controller_cls(settings, "corrected", stage)
        )
        result = controller.run(
            TabularMLP(3, [1], 2), None, model_config
        )
        return result, visits

    greedy = ModelConfig(id="LGND", controller="lgnd", order="width")
    generic = ModelConfig(id="GENERIC", controller="phase_search",
                          order="width", generic=True)
    for cls, config in (
        (DedicatedController, greedy),
        (GenericCandidateSearch, generic),
    ):
        base, original_visits = simulate(cls, config, "loss")
        alt, variance_visits = simulate(cls, config, "variance_stagnation")
        assert len(original_visits) == 2
        assert len(variance_visits) == 3
        # The original loss-selection objective still retains the initial
        # architecture, even when the variance criterion trials more growth.
        assert base.model.hidden_widths == [1]
        assert alt.model.hidden_widths == [1]
