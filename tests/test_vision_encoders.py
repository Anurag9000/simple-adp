"""Real torchvision V01 backbone construction and local weights (no downloads)."""
import copy

import pytest
import torch
from torch import nn

from simple_adp.vision_encoders import create_frozen_vision_encoder


def test_resnet18_local_checkpoint_is_exactly_stripped_and_frozen(tmp_path):
    torchvision = pytest.importorskip("torchvision")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4)
        full = torchvision.models.resnet18(weights=None).eval()
    path = tmp_path / "resnet18_source_state.pt"
    torch.save(full.state_dict(), path)
    expected = copy.deepcopy(full)
    expected.fc = nn.Identity()
    before = torch.get_rng_state().clone()
    encoder, metadata = create_frozen_vision_encoder(
        "resnet18", weights_path=path, normalization="none",
    )
    assert torch.equal(before, torch.get_rng_state())
    assert metadata["pretrained"] is None  # a local checkpoint does not establish its training provenance
    assert metadata["weights_source"] == "unverified_local_state_dict"
    assert metadata["weights_file_sha256"]
    assert metadata["feature_dim"] == 512
    assert encoder.training is False
    assert all(not p.requires_grad for p in encoder.parameters())
    images = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        result, reference = encoder(images), expected(images)
    assert result.shape == (2, 512)
    assert torch.allclose(result, reference, atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError, match="SHA256"):
        create_frozen_vision_encoder("resnet18", weights_path=path,
                                     expected_weights_sha256="0" * 64)


def test_encoder_constructor_validates_architecture_and_shapes():
    pytest.importorskip("torchvision")
    with pytest.raises(ValueError, match="unknown frozen"):
        create_frozen_vision_encoder("unknown")
    with pytest.raises(ValueError, match="weights checksum requires"):
        create_frozen_vision_encoder("resnet18", expected_weights_sha256="1234")
    before = torch.get_rng_state().clone()
    encoder, metadata = create_frozen_vision_encoder("resnet18", seed=99)
    assert metadata["pretrained"] is False
    assert metadata["normalization"] == "none"
    assert torch.equal(before, torch.get_rng_state())
    with pytest.raises(ValueError, match=r"\[B,3,H,W\]"):
        encoder(torch.zeros(2, 1, 32, 32))
    with pytest.raises(ValueError, match="minimum"):
        encoder(torch.zeros(2, 3, 8, 8))
    encoder.train()
    assert encoder.training is False
