"""Exact accounting of frozen backbone, eligible ADP sites and trainable heads."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from simple_adp.models import TabularMLP, rebuild_mlp
from simple_adp.visual_accounting import visual_parameter_accounting


def _model():
    encoder = nn.Linear(4, 4).requires_grad_(False)
    local = TabularMLP(4, (3,), 2)
    task_head = nn.Linear(2, 2)
    return nn.ModuleDict({"encoder": encoder, "local": local, "head": task_head}), local


def test_visual_counts_are_exact_and_do_not_consume_rng():
    model, local = _model()
    state = torch.get_rng_state().clone()
    result = visual_parameter_accounting(model, {"adapter:one": local})
    assert torch.equal(torch.get_rng_state(), state)
    assert result["total_parameters"] == 49
    assert result["frozen_parameters"] == 20
    assert result["trainable_parameters"] == 29
    assert result["eligible_mlp_parameters"] == 23
    assert result["currently_trainable_eligible_mlp_parameters"] == 23
    assert result["other_trainable_parameters"] == 6
    site = result["eligible_growth_sites"]["adapter:one"]
    assert site["input_dim"] == 4 and site["output_dim"] == 2
    assert site["hidden_widths"] == [3]
    assert site["linear_macs_per_mlp_invocation"] == 18
    assert site["currently_trainable"]
    assert "whole-model FLOPs" in result["linear_compute_note"]


def test_true_growth_increases_only_eligible_parameter_count():
    model, local = _model()
    before = visual_parameter_accounting(model, {"head": local})
    grown = rebuild_mlp(local, (4,))
    model["local"] = grown
    after = visual_parameter_accounting(model, {"head": grown})
    assert after["eligible_mlp_parameters"] - before["eligible_mlp_parameters"] == 7
    assert after["frozen_parameters"] == before["frozen_parameters"]
    assert after["other_trainable_parameters"] == before["other_trainable_parameters"]
    assert after["eligible_growth_sites"]["head"]["hidden_widths"] == [4]
    assert after["eligible_growth_sites"]["head"]["linear_macs_per_mlp_invocation"] == 24


def test_inactive_multisite_targets_are_distinguished_from_trainable_site():
    model, local = _model()
    inactive = TabularMLP(4, (2,), 2).requires_grad_(False)
    model["previous"] = inactive
    counts = visual_parameter_accounting(model, {
        "adapter:active": local, "adapter:previous": inactive,
    })
    assert counts["eligible_growth_sites"]["adapter:active"]["currently_trainable"]
    assert not counts["eligible_growth_sites"]["adapter:previous"]["currently_trainable"]
    assert counts["currently_trainable_eligible_mlp_parameters"] == 23
    assert counts["eligible_mlp_parameters"] > 23
    assert counts["other_trainable_parameters"] == 6


def test_reject_nonowned_aliased_partial_and_nonlinear_sites():
    model, local = _model()
    with pytest.raises(ValueError, match="owned"):
        visual_parameter_accounting(model, {"foreign": TabularMLP(4, (3,), 2)})
    with pytest.raises(ValueError, match="alias"):
        visual_parameter_accounting(model, {"a": local, "b": local})
    with pytest.raises(TypeError, match="TabularMLP"):
        visual_parameter_accounting(model, {"convolution": nn.Conv2d(3, 3, 1)})
    with pytest.raises(ValueError, match="partially frozen"):
        local.output_layer.weight.requires_grad_(False)
        visual_parameter_accounting(model, {"partial": local})
