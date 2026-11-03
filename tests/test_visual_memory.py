"""CPU exact logical byte accounting and signed visual growth admission tests."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from simple_adp.config import SearchSettings
from simple_adp.visual_inner import ResidualAdapterSpec
from simple_adp.visual_native_ffn import build_visual_composite
from simple_adp.visual_memory import visual_model_tensor_payload
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
    rejected = search.search_state_["memory_admission_rejections"]
    assert len(rejected) == 1
    assert rejected[0]["phase"] == "width"
    assert rejected[0]["architecture"] == [3]
    assert rejected[0]["cap_bytes"] == base
    assert rejected[0]["logical_model_tensor_payload_bytes"] > base
    assert rejected[0]["reason"] == "model-parameter-buffer-payload-limit"
    assert search.search_state_["proposals"] == 0  # proposal was never admitted
    assert checkpoint.is_file()

    same = VisualResidualSearch(
        "ADP-W-RR", spec(), settings=settings(),
        max_model_tensor_payload_bytes=base,
    )
    same.fit(x, y, validation_data=valid, checkpoint_path=checkpoint)
    assert same.history_ == search.history_
    assert same.search_state_["memory_admission_rejections"] == rejected

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
