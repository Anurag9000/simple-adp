"""Observed visual compute: exact dense/conv MACs, mode/RNG safety and growth."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from simple_adp.models import TabularMLP, rebuild_mlp
from simple_adp.visual_compute import visual_linear_conv_macs


def test_true_2d_convolution_and_linear_macs_preserve_modes_and_rng():
    model = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4),
        nn.ReLU(), nn.Flatten(), nn.Linear(4 * 5 * 7, 2),
    )
    model[0].requires_grad_(False)
    model[1].requires_grad_(False)
    model.eval()
    model[4].train(True)  # Mixed input modes must be restored exactly.
    flags = [module.training for module in model.modules()]
    states = {name: item.clone() for name, item in model.state_dict().items()}
    grad_flags = [p.requires_grad for p in model.parameters()]
    rng = torch.get_rng_state().clone()
    x = torch.ones((1, 3, 5, 7))
    output = visual_linear_conv_macs(model, x)
    assert output["observed_linear_conv_macs_per_image"] == 3780 + 280
    assert output["observed_modules"]["0"]["macs"] == 3780
    assert output["observed_modules"]["4"]["macs"] == 280
    assert output["whole_model_flops"] is None
    assert "partial" in output["coverage"]
    assert [module.training for module in model.modules()] == flags
    assert [p.requires_grad for p in model.parameters()] == grad_flags
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(states[key], tensor) for key, tensor in model.state_dict().items())


class SpatialFFN(nn.Module):
    def __init__(self, width=3):
        super().__init__()
        self.local_mlp = TabularMLP(4, (width,), 4)
        self.readout = nn.Linear(4, 2).requires_grad_(False)

    def forward(self, image):
        spatial = image.permute(0, 2, 3, 1)
        z = self.local_mlp(spatial)
        return self.readout(z.mean(dim=(1, 2)))


def test_spatial_mlp_macs_scale_with_positions_and_growth_width():
    before = SpatialFFN(width=3)
    after = SpatialFFN(width=4)
    after.local_mlp = rebuild_mlp(before.local_mlp, (4,))
    x = torch.ones((1, 4, 5, 7))
    p = visual_linear_conv_macs(before, x)
    q = visual_linear_conv_macs(after, x)
    # Input + output matmul per spatial pixel: (4 + 4) MACs/neuron.
    assert q["observed_linear_conv_macs_per_image"] - p["observed_linear_conv_macs_per_image"] == 5 * 7 * 8
    assert p["observed_modules"]["local_mlp.hidden_layers.0"]["macs"] == 5 * 7 * 4 * 3
    assert p["observed_modules"]["local_mlp.output_layer"]["macs"] == 5 * 7 * 3 * 4
    assert p["observed_modules"]["readout"]["macs"] == 4 * 2


def test_repeated_linear_call_is_counted_each_time():
    linear = nn.Linear(3, 5)
    class Reused(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = linear
        def forward(self, image):
            tokens = image.flatten(2).transpose(1, 2)
            return self.block(tokens) + self.block(tokens)
    item = Reused()
    estimate = visual_linear_conv_macs(item, torch.ones((1, 3, 2, 3)))
    assert estimate["observed_modules"]["block"]["forward_calls"] == 2
    assert estimate["observed_linear_conv_macs_per_image"] == 2 * 6 * 3 * 5


@pytest.mark.parametrize("invalid", [
    torch.ones((3, 5, 7)),
    torch.ones((2, 3, 5, 7)),
    torch.full((1, 3, 5, 7), float("nan")),
    torch.ones((1, 3, 5, 7), dtype=torch.int64),
])
def test_invalid_visual_probe_rejected(invalid):
    with pytest.raises(ValueError):
        visual_linear_conv_macs(nn.Identity(), invalid)
