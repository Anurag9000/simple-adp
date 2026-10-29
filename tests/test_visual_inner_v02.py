"""V02 native inner-MLP growth: full frozen-downstream differentiable tests."""
import math

import numpy as np
import pytest
import torch

from simple_adp.config import SearchSettings
from simple_adp.tasks import ClassificationTask
from simple_adp.training import StageTrainer
from simple_adp.visual_inner import ResidualAdapterSpec, VisualResidualAdapter


def _spec():
    return ResidualAdapterSpec(
        input_channels=3, backbone_channels=6, local_dim=4,
        classes=2, image_height=8, image_width=9,
        fixed_seed=17, site_id="adapter:block1",
    )


def _images():
    torch.manual_seed(111)
    x = torch.randn(20, 3, 8, 9)
    y = np.asarray([0, 1] * 10, dtype=np.int64)
    x[y == 1] += 0.35
    return x, y


def test_v02_backward_flows_through_frozen_downstream_operations():
    torch.manual_seed(13)
    model = VisualResidualAdapter(_spec(), (3, 3))
    frozen = model.frozen_state_sha256()
    x, _ = _images()
    output = model(x[:4])
    assert output.shape == (4, 2)
    loss = output.square().mean()
    loss.backward()
    assert model.local_mlp.output_layer.weight.grad is not None
    assert model.local_mlp.output_layer.weight.grad.abs().sum().item() > 0
    assert all(parameter.grad is None for group in model._fixed_modules() for parameter in group.parameters())
    assert all(not module.training for module in model._fixed_modules())
    assert model.frozen_state_sha256() == frozen
    assert model(x[:1]).shape == (1, 2)
    with pytest.raises(ValueError, match="resolution"):
        model(torch.randn(2, 3, 9, 9))


def test_v02_local_rr_all_and_depth_are_real_mutations_with_fixed_interfaces():
    torch.manual_seed(22)
    original = VisualResidualAdapter(_spec(), (3, 3))
    old_state = original.local_mlp.state_dict()
    old_frozen = original.frozen_state_sha256()
    rr = original.grow_width(site_id="adapter:block1", operator="round_robin",
                             increment=1, max_width=5)
    assert rr is not None
    assert rr.hidden_widths == [4, 3]
    assert rr._width_cursor == 1
    assert rr.input_dim == original.input_dim == 4
    assert rr.output_dim == original.output_dim == 4
    assert torch.equal(rr.local_mlp.hidden_layers[0].weight[:3], old_state["hidden_layers.0.weight"])
    assert rr.frozen_state_sha256() == old_frozen
    assert sum(p.numel() for p in rr.parameters()) > sum(p.numel() for p in original.parameters())
    all_layers = original.grow_width(site_id="adapter:block1", operator="all_layers",
                                     increment=1, max_width=5)
    assert all_layers is not None and all_layers.hidden_widths == [4, 4]
    den = original.grow_width(site_id="adapter:block1", operator="den_round_robin",
                              increment=1, max_width=5)
    assert den is not None and den.hidden_widths == [4, 3]
    source_d1 = original.grow_depth(site_id="adapter:block1", operator="D1",
                                    max_depth=3, seed_width=1)
    source_dcur = original.grow_depth(site_id="adapter:block1", operator="Dcur",
                                      max_depth=3)
    assert source_d1 is not None and source_d1.hidden_widths == [3, 3, 1]
    assert source_dcur is not None and source_dcur.hidden_widths == [3, 3, 3]
    assert source_d1.frozen_state_sha256() == old_frozen
    with pytest.raises(ValueError, match="unknown"):
        original.grow_width(site_id="adapter:wrong", operator="round_robin",
                            increment=1, max_width=5)
    assert original.hidden_widths == [3, 3]


def test_v02_stage_trainer_with_original_task_and_exact_architecture_restore():
    x, labels = _images()
    task = ClassificationTask()
    task.fit_targets(labels[:12])
    device = torch.device("cpu")
    model = VisualResidualAdapter(_spec(), (3,))
    fixed_before = model.frozen_state_sha256()
    trainer = StageTrainer(
        task, SearchSettings(
            init_width=3, init_depth=1, max_width=5,
            max_epochs=1, max_candidates=1, batch_size=4,
            patience_es=1, device="cpu", seed=51,
        )
    )
    stage = trainer.train_stage(
        model,
        x[:12], task.encode_targets(labels[:12], device=device),
        x[12:], task.encode_targets(labels[12:], device=device),
    )
    assert math.isfinite(stage.best_score)
    assert stage.epochs == 1
    assert stage.batches >= 1
    assert model.frozen_state_sha256() == fixed_before
    state = torch.get_rng_state().clone()
    restored = VisualResidualAdapter.from_snapshot(stage.best_snapshot)
    assert torch.equal(torch.get_rng_state(), state)
    restored.verify_frozen()
    assert restored.hidden_widths == model.hidden_widths
    with torch.no_grad():
        assert torch.allclose(restored(x[12:]), model(x[12:]), atol=1e-7)
    grown = restored.grow_width(site_id="adapter:block1", operator="round_robin",
                                increment=1, max_width=5)
    assert grown is not None
    second = trainer.train_stage(
        grown, x[:12], task.encode_targets(labels[:12], device=device),
        x[12:], task.encode_targets(labels[12:], device=device),
    )
    assert math.isfinite(second.best_score)
    assert grown.frozen_state_sha256() == fixed_before
    assert grown.hidden_widths != model.hidden_widths
