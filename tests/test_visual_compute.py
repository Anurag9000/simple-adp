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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires physical CUDA GPU")
def test_cuda_inference_trace_matches_cpu_exact_mac_counts():
    cpu = SpatialFFN(width=3).eval()
    gpu = SpatialFFN(width=3).eval().to("cuda")
    gpu.load_state_dict(cpu.state_dict())
    image = torch.ones((1, 4, 5, 7))
    reference = visual_linear_conv_macs(cpu, image)
    before = torch.cuda.get_rng_state().clone()
    observed = visual_linear_conv_macs(gpu, image.cuda())
    assert observed["observed_linear_conv_macs_per_image"] == reference[
        "observed_linear_conv_macs_per_image"
    ]
    assert observed["observed_modules"] == reference["observed_modules"]
    assert torch.equal(torch.cuda.get_rng_state(), before)


class SelfAttentionImage(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(8, 2, dropout=0, batch_first=True)

    def forward(self, image):
        tokens = image.flatten(2).transpose(1, 2)
        output, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        return output.mean(dim=1)


def test_attention_dense_macs_separately_account_projections_and_score_value():
    model = SelfAttentionImage()
    model.train(True)
    x = torch.ones((1, 8, 2, 3))
    rng = torch.get_rng_state().clone()
    result = visual_linear_conv_macs(model, x)
    # 6 tokens, E=8, 2 heads: 4 linear projections + QK/AV products.
    projections = 4 * 6 * 8 * 8
    score_value = 2 * 6 * 6 * 8
    mha = result["analytical_multihead_attention_modules"]["attention"]
    assert mha["qkv_and_output_projection_macs"] == projections
    assert mha["score_and_value_product_macs"] == score_value
    assert mha["total_dense_attention_macs"] == projections + score_value
    assert mha["forward_calls"] == 1
    # The MHA.out_proj nn.Linear is not double-counted.
    assert result["observed_linear_conv_macs_per_image"] == 0
    assert result["analytical_multihead_attention_macs_per_image"] == projections + score_value
    assert result["partial_dense_attention_inclusive_macs_per_image"] == projections + score_value
    assert "attention.out_proj" not in result["observed_modules"]
    assert model.training and model.attention.training
    assert torch.equal(torch.get_rng_state(), rng)
    assert result["whole_model_flops"] is None


def test_cross_attention_with_distinct_key_value_dimensions():
    class Cross(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.MultiheadAttention(
                8, 2, dropout=0, kdim=4, vdim=6, batch_first=True
            )
            self.register_buffer("key", torch.ones((1, 3, 4)))
            self.register_buffer("value", torch.ones((1, 3, 6)))

        def forward(self, image):
            query = image.flatten(2).transpose(1, 2)
            output, _ = self.attention(query, self.key, self.value, need_weights=False)
            return output.mean(dim=1)

    result = visual_linear_conv_macs(Cross(), torch.ones((1, 8, 1, 2)))
    # Q(2x8->8) + K(3x4->8) + V(3x6->8) + O(2x8->8).
    projections = 2 * 8 * 8 + 3 * 4 * 8 + 3 * 6 * 8 + 2 * 8 * 8
    score_value = 2 * 2 * 3 * 8
    assert result["analytical_multihead_attention_macs_per_image"] == projections + score_value


def test_attention_sequence_first_layout_and_repeated_calls():
    class SequenceFirst(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.MultiheadAttention(8, 2, batch_first=False, dropout=0)

        def forward(self, image):
            tokens = image.flatten(2).permute(2, 0, 1)
            a, _ = self.attention(tokens, tokens, tokens, need_weights=False)
            b, _ = self.attention(tokens, tokens, tokens, need_weights=False)
            return (a + b).mean(dim=0)

    result = visual_linear_conv_macs(SequenceFirst(), torch.ones((1, 8, 2, 3)))
    expected_one = 4 * 6 * 8 * 8 + 2 * 6 * 6 * 8
    assert result["analytical_multihead_attention_modules"]["attention"]["forward_calls"] == 2
    assert result["analytical_multihead_attention_macs_per_image"] == 2 * expected_one


def test_probe_restores_parent_and_child_training_flags_on_forward_failure():
    class Fails(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Conv2d(3, 4, 1)
            self.last = nn.Linear(4, 2)

        def forward(self, image):
            self.first(image)
            raise RuntimeError("intentional probe failure")

    model = Fails()
    model.eval()
    model.last.train(True)
    modes = tuple(module.training for module in model.modules())
    before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="intentional probe failure"):
        visual_linear_conv_macs(model, torch.ones((1, 3, 4, 4)))
    assert modes == tuple(module.training for module in model.modules())
    assert torch.equal(torch.get_rng_state(), before)
    assert not model.first._forward_hooks
    assert not model.last._forward_hooks
