"""Stage-level variance growth evidence must match loss-selected best weights."""
import copy
import pytest
import torch

from simple_adp.config import SearchSettings
from simple_adp.models import TabularMLP
from simple_adp.tasks import RegressionTask
from simple_adp.training import StageTrainer


def _trainer():
    settings = SearchSettings(
        init_width=2, init_depth=1,
        max_width=4, max_depth=2, max_neurons=16,
        max_epochs=3, batch_size=8, patience_es=2,
        expansion_criterion="variance_stagnation",
        seed=17,
    )
    task = RegressionTask()
    task.fit_targets([0.0, 1.0])
    return StageTrainer(task, settings)


def test_variance_state_is_restored_with_best_validation_model(monkeypatch):
    scores = iter([0.1, 0.5, 0.8])
    monkeypatch.setattr(
        StageTrainer, "_score",
        staticmethod(lambda *args, **kwargs: next(scores)),
    )
    trainer = _trainer()
    model = TabularMLP(3, [2], 1)
    x = torch.randn(8, 3)
    y = torch.randn(8, 1)
    result = trainer.train_stage(model, x, y, x, y)
    assert result.best_score == 0.1
    assert result.epochs == 3
    assert result.batches == 3
    # The selected candidate was from epoch 1, so the 3rd epoch's
    # posterior cannot decide whether the retained model may expand.
    state = result.checkpoint_state
    assert state["shadow_bgd_variance"]["observations"] == 1
    assert state["best_shadow_bgd_variance"]["observations"] == 1
    assert result.variance_stagnated is False
    for key, value in model.state_dict().items():
        assert torch.equal(value, state["best_model_state"][key])


def test_previous_variance_checkpoint_without_best_posterior_fails_closed():
    trainer = _trainer()
    model = TabularMLP(3, [2], 1)
    x = torch.randn(8, 3)
    y = torch.randn(8, 1)
    result = trainer.train_stage(model, x, y, x, y)
    invalid = copy.deepcopy(result.checkpoint_state)
    del invalid["best_shadow_bgd_variance"]
    with pytest.raises(ValueError, match="best-epoch posterior provenance"):
        _trainer().train_stage(
            TabularMLP(3, [2], 1), x, y, x, y,
            resume_state=invalid,
        )
