"""CPU exact logical byte accounting and signed visual growth admission tests."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from simple_adp.config import SearchSettings
from simple_adp.visual_inner import ResidualAdapterSpec
from simple_adp.visual_native_ffn import build_visual_composite
from simple_adp.visual_memory import (
    visual_model_tensor_payload, projected_tabular_mlp_payload_bytes,
    prospective_tabular_growth_widths, forecast_visual_growth_payload_bytes,
)
from simple_adp.models import TabularMLP, rebuild_mlp
from simple_adp.growth import (
    expand_width_all_layers, expand_width_round_robin,
    expand_width_den_round_robin, expand_depth, expand_depth_match_width,
)
from simple_adp.visual_inner import VisualResidualAdapter
from simple_adp.visual_mixer import MixerSpec
from simple_adp.visual_search import VisualResidualSearch


class SharedMixedDtype(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 3, dtype=torch.float64))
        self.again = self.weight
        self.frozen = nn.Parameter(torch.zeros(3, dtype=torch.complex64), requires_grad=False)
        self.register_buffer("buffer", torch.ones((5,), dtype=torch.float32))
        self.register_buffer("counter", torch.zeros(1, dtype=torch.int64))

    def forward(self, image):
        return image


def test_logical_model_payload_exact_dtypes_shared_parameter_and_buffers():
    m = SharedMixedDtype()
    before = torch.get_rng_state().clone()
    state = {name: tensor.detach().clone() for name, tensor in m.state_dict().items()}
    accounted = visual_model_tensor_payload(m)
    assert accounted["parameter_tensors"] == 2
    assert accounted["buffer_tensors"] == 2
    assert accounted["trainable_parameter_bytes"] == 2 * 3 * 8
    assert accounted["frozen_parameter_bytes"] == 3 * 8
    assert accounted["buffer_payload_bytes"] == 5 * 4 + 8
    assert accounted["parameter_payload_bytes"] == 72
    assert accounted["total_model_tensor_payload_bytes"] == 100
    assert accounted["nominal_dense_gradient_payload_bytes"] == 48
    assert accounted["nominal_adam_or_adamw_two_moment_payload_bytes"] == 96
    assert accounted["nominal_dense_adam_or_adamw_model_grad_moments_bytes"] == 244
    assert accounted["logical_bytes_by_dtype"] == {
        "torch.complex64": 24, "torch.float32": 20,
        "torch.float64": 48, "torch.int64": 8,
    }
    assert accounted["logical_bytes_by_device"] == {"cpu": 100}
    assert accounted["peak_training_memory_bytes"] is None
    assert accounted["unique_allocator_storage_bytes"] is None
    assert torch.equal(torch.get_rng_state(), before)
    assert all(torch.equal(v, m.state_dict()[name]) for name, v in state.items())


def test_reject_non_model():
    with pytest.raises(TypeError):
        visual_model_tensor_payload(object())


def spec():
    return ResidualAdapterSpec(3, 4, 3, 2, 8, 8, site_id="adapter:block1")


def settings():
    return SearchSettings(
        seed=41, device="cpu", init_width=2, init_depth=1,
        max_width=4, max_depth=2, max_neurons=12,
        max_candidates=1, max_epochs=1, batch_size=4,
        patience_es=1, patience_width_exp=1,
        patience_depth_exp=1,
    )


def data():
    rng = np.random.default_rng(701)
    images = rng.normal(size=(24, 3, 8, 8)).astype(np.float32)
    labels = np.asarray([0, 1] * 12)
    return images[:16], labels[:16], (images[16:], labels[16:])


def initial_bytes():
    m = build_visual_composite(spec(), (2,))
    return visual_model_tensor_payload(m)["total_model_tensor_payload_bytes"]


def test_optional_memory_cap_blocks_growth_but_preserves_initial_model_and_history(tmp_path):
    base = initial_bytes()
    x, y, valid = data()
    checkpoint = tmp_path / "memory-cap.pt"
    search = VisualResidualSearch(
        "ADP-W-RR", spec(), settings=settings(),
        max_model_tensor_payload_bytes=base,
    )
    search.fit(x, y, validation_data=valid, checkpoint_path=checkpoint)
    assert [row["architecture"] for row in search.history_] == [[2]]
    assert search.search_state_["training_budget"]["trained_stages"] == 1
    trace = search.search_state_["trained_model_memory_trace"]
    assert len(trace) == 1
    assert trace[0]["architecture"] == [2]
    assert trace[0]["logical_model_tensor_payload_bytes"] == base
    assert trace[0]["optimizer_steps"] == 4
    assert search.search_state_["trained_model_memory_trace_complete"] is True
    rejected = search.search_state_["memory_admission_rejections"]
    assert len(rejected) == 1
    assert rejected[0]["phase"] == "width"
    assert rejected[0]["architecture"] == [3]
    assert rejected[0]["cap_bytes"] == base
    assert rejected[0]["logical_model_tensor_payload_bytes"] > base
    assert rejected[0]["reason"] == "model-parameter-buffer-payload-limit"
    assert rejected[0]["before_candidate_construction"] is True
    assert rejected[0]["prediction_kind"] == "native-TabularMLP-geometry-v1"
    assert search.search_state_["proposals"] == 0  # proposal was never admitted
    assert checkpoint.is_file()

    same = VisualResidualSearch(
        "ADP-W-RR", spec(), settings=settings(),
        max_model_tensor_payload_bytes=base,
    )
    same.fit(x, y, validation_data=valid, checkpoint_path=checkpoint)
    assert same.history_ == search.history_
    assert same.search_state_["memory_admission_rejections"] == rejected
    assert same.search_state_["trained_model_memory_trace"] == trace

    with pytest.raises(ValueError, match="signature"):
        VisualResidualSearch(
            "ADP-W-RR", spec(), settings=settings(),
            max_model_tensor_payload_bytes=base + 1,
        ).fit(x, y, validation_data=valid, checkpoint_path=checkpoint)


def test_initial_model_must_fit_configured_payload_limit():
    x, y, valid = data()
    with pytest.raises(ValueError, match="initial visual model logical tensor payload"):
        VisualResidualSearch(
            "ADP-W-RR", spec(), settings=settings(),
            max_model_tensor_payload_bytes=initial_bytes() - 1,
        ).fit(x, y, validation_data=valid)


@pytest.mark.parametrize("bad", [True, False, 0, -1, 1.2, "1024"])
def test_invalid_byte_caps_rejected_before_training(bad):
    with pytest.raises(ValueError, match="positive integer"):
        VisualResidualSearch(
            "ADP-W-RR", spec(), settings=settings(),
            max_model_tensor_payload_bytes=bad,
        )


def test_optional_unlimited_policy_retains_source_search_signature():
    search = VisualResidualSearch("ADP-W-RR", spec(), settings=settings())
    capped = VisualResidualSearch(
        "ADP-W-RR", spec(), settings=settings(),
        max_model_tensor_payload_bytes=initial_bytes(),
    )
    x, y, valid = data()
    assert search._signature(x, y, *valid) != capped._signature(x, y, *valid)
    assert search.max_model_tensor_payload_bytes is None


@pytest.mark.parametrize("widths", [
    (2,), (3,), (2, 4), (4, 2, 5), (1, 1, 1),
])
@pytest.mark.parametrize("batch_norm", [False, True])
def test_static_tabular_payload_prediction_exact_for_new_widths(widths, batch_norm):
    model = TabularMLP(4, (2, 3), 5, batch_norm=batch_norm)
    predicted = projected_tabular_mlp_payload_bytes(model, widths)
    actual = rebuild_mlp(model, widths)
    assert predicted == visual_model_tensor_payload(actual)["total_model_tensor_payload_bytes"]
    assert projected_tabular_mlp_payload_bytes(
        model, tuple(model.hidden_widths)
    ) == visual_model_tensor_payload(model)["total_model_tensor_payload_bytes"]


def test_static_payload_returns_unknown_for_nondefault_dtype():
    model = TabularMLP(3, (2,), 2).double()
    assert projected_tabular_mlp_payload_bytes(model, (3,)) is None


@pytest.mark.parametrize("operator,source,expected", [
    ("all_layers", (2, 3), (3, 4)),
    ("round_robin", (2, 3), (3, 3)),
    ("den_round_robin", (2, 3), (3, 3)),
    ("D1", (2, 2), (2, 2, 1)),
    ("Dcur", (2, 3), (2, 3, 3)),
])
def test_static_growth_geometry_matches_real_native_growth(operator, source, expected):
    model = TabularMLP(3, source, 2)
    predicted = prospective_tabular_growth_widths(
        model, operator=operator, increment=1, max_width=4,
        max_depth=3, seed_width=1,
    )
    assert predicted == expected
    grow = {
        "all_layers": lambda: expand_width_all_layers(model, 1, 4),
        "round_robin": lambda: expand_width_round_robin(model, 1, 4),
        "den_round_robin": lambda: expand_width_den_round_robin(model, 1, 4),
        "D1": lambda: expand_depth(model, 3, 1),
        "Dcur": lambda: expand_depth_match_width(model, 3),
    }[operator]()
    assert grow is not None
    assert tuple(grow.hidden_widths) == predicted


def test_static_proposal_forecast_matches_full_composite_payload_with_mixer_both():
    spec_both = MixerSpec(
        input_channels=3, local_dim=4, classes=2,
        image_height=8, image_width=8, patch_size=4,
        site_id="both:block0",
    )
    original = build_visual_composite(spec_both, (2,))
    widened = original.grow_width(
        site_id="both:block0", operator="den_round_robin",
        increment=1, max_width=4,
    )
    assert widened is not None
    predicted = forecast_visual_growth_payload_bytes(
        original, (3,), other_hidden_widths=(3,),
    )
    assert predicted == visual_model_tensor_payload(widened)[
        "total_model_tensor_payload_bytes"
    ]


def test_preflight_blocks_growth_without_invoking_growth_allocator(monkeypatch):
    cap = initial_bytes()
    x, y, valid = data()

    def forbidden_growth(*args, **kwargs):
        raise AssertionError("oversized candidate growth must not be constructed")

    monkeypatch.setattr(VisualResidualAdapter, "grow_width", forbidden_growth)
    search = VisualResidualSearch(
        "ADP-W-RR", spec(), settings=settings(),
        max_model_tensor_payload_bytes=cap,
    )
    search.fit(x, y, validation_data=valid)
    assert search.search_state_["memory_admission_rejections"][0][
        "before_candidate_construction"
    ] is True


def test_memory_preflight_rejects_fixed_grid_before_new_model_construction(monkeypatch):
    from simple_adp import visual_search as source
    x, y, valid = data()
    cap = initial_bytes()
    original = source.build_visual_composite
    calls = []

    def tracked(*args, **kwargs):
        calls.append(args[1])
        if args[1] != (2,):
            raise AssertionError("over-cap fixed grid should be preflight-rejected")
        return original(*args, **kwargs)

    monkeypatch.setattr(source, "build_visual_composite", tracked)
    search = VisualResidualSearch(
        "MLP-Fixed", spec(), settings=settings(), fixed_grid=((2,), (3,)),
        max_model_tensor_payload_bytes=cap,
    )
    search.fit(x, y, validation_data=valid)
    assert len(search.history_) == 1
    assert calls == [(2,)]
    assert len(search.search_state_["memory_admission_rejections"]) == 1
    assert search.search_state_["memory_admission_rejections"][0][
        "before_candidate_construction"
    ] is True


@pytest.mark.parametrize("family,site", [
    ("V03", "ffn:block0"),
    ("V04", "channel:stage0.block0"),
    ("V05", "channel:poolformer.block0"),
    ("V06a", "channel:s2.block0"),
    ("V06b", "channel:axial.block0"),
    ("V10", "channel:resmlp.block0"),
    ("V11", "channel:gfnet.block0"),
])
def test_native_ffn_family_forecast_matches_actually_constructed_width(family, site):
    from simple_adp.visual_native_ffn import NativeFFNSpec
    family_spec = NativeFFNSpec(
        architecture_id=family, input_channels=3, local_dim=4,
        classes=2, image_height=8, image_width=8,
        patch_size=2, attention_heads=2, site_id=site,
    )
    original = build_visual_composite(family_spec, (2,))
    candidate = original.grow_width(
        site_id=site, operator="den_round_robin",
        increment=1, max_width=4,
    )
    assert candidate is not None
    forecast = forecast_visual_growth_payload_bytes(original, (3,))
    assert forecast == visual_model_tensor_payload(candidate)[
        "total_model_tensor_payload_bytes"
    ]


@pytest.mark.parametrize("variant", [
    "V02", "V07-token", "V07-channel", "V07-both",
    "V08-early", "V08-middle", "V08-late", "V08-schedule",
    "V09-bank0", "V09-bank1", "V09-bank2",
])
def test_remaining_visual_growth_sites_forecast_matches_real_bytes(variant):
    from simple_adp.visual_multistage import MultistageSpec
    from simple_adp.visual_adapter_bank import BankSpec
    from simple_adp.visual_multistage_schedule import ScheduledMultistageSpec

    if variant == "V02":
        candidate_spec = spec()
    elif variant.startswith("V07"):
        candidate_spec = MixerSpec(
            input_channels=3, local_dim=4, classes=2,
            image_height=8, image_width=8, patch_size=2,
            site_id=variant.split("-", 1)[1] + ":block0",
        )
    elif variant == "V08-schedule":
        candidate_spec = ScheduledMultistageSpec(
            input_channels=3, backbone_channels=4,
            classes=2, image_height=8, image_width=8,
        )
    elif variant.startswith("V08"):
        candidate_spec = MultistageSpec(
            input_channels=3, backbone_channels=4,
            classes=2, image_height=8, image_width=8,
            site_id="adapter:" + variant.split("-", 1)[1],
        )
    else:
        member = int(variant[-1])
        candidate_spec = BankSpec(
            input_channels=3, backbone_channels=4, local_dim=4,
            classes=2, image_height=8, image_width=8,
            site_id=f"bank:{member}",
        )
    original = build_visual_composite(candidate_spec, (2,))
    candidate = original.grow_width(
        site_id=candidate_spec.site_id, operator="den_round_robin",
        increment=1, max_width=4,
    )
    assert candidate is not None
    predicted = forecast_visual_growth_payload_bytes(
        original, (3,),
        other_hidden_widths=(3,) if variant == "V07-both" else None,
    )
    assert predicted == visual_model_tensor_payload(candidate)[
        "total_model_tensor_payload_bytes"
    ]


@pytest.mark.parametrize("kind", ["adam", "adamw"])
def test_nominal_gradient_and_adam_moment_bytes_match_actual_dense_optimizer_state(kind):
    module = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    module[0].requires_grad_(False)
    before = visual_model_tensor_payload(module)
    optimizer_class = torch.optim.Adam if kind == "adam" else torch.optim.AdamW
    optimizer = optimizer_class(p for p in module.parameters() if p.requires_grad)
    prediction = module(torch.ones((5, 3)))
    prediction.square().sum().backward()
    optimizer.step()
    observed_gradient_bytes = sum(
        p.grad.numel() * p.grad.element_size()
        for p in module.parameters() if p.grad is not None
    )
    observed_moments_bytes = sum(
        state[name].numel() * state[name].element_size()
        for state in optimizer.state.values()
        for name in ("exp_avg", "exp_avg_sq")
    )
    assert before["nominal_dense_gradient_payload_bytes"] == observed_gradient_bytes
    assert before["nominal_adam_or_adamw_two_moment_payload_bytes"] == observed_moments_bytes
    assert before["peak_training_memory_bytes"] is None
    assert "step scalars" in before["nominal_optimizer_assumption"]
