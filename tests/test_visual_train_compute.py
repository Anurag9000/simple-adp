"""Real forward-only visual training operator accounting and resume tests."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from simple_adp.visual_train_compute import VisualTrainingForwardMACCounter


def test_observed_linear_conv_counts_exact_actual_batches_not_evaluation():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 2, kernel_size=1)
            self.head = nn.Linear(2 * 4 * 4, 5)

        def forward(self, x):
            return self.head(self.conv(x).flatten(1))

    model = Tiny().train()
    before_rng = torch.get_rng_state().clone()
    before_state = {k: v.clone() for k, v in model.state_dict().items()}
    with VisualTrainingForwardMACCounter(model) as counter:
        # An unbracketed forward, including validation, contributes nothing.
        with torch.no_grad():
            model(torch.ones(1, 3, 4, 4))
        assert counter.snapshot()["partial_training_forward_macs"] == 0
        for batch in (2, 1):
            x = torch.ones(batch, 3, 4, 4)
            with counter.record_batch(batch):
                model(x)
    observed = counter.snapshot()
    conv = 3 * 2 * 4 * 4 * 3  # 3 images, 2 outputs, 3 input channels
    linear = 3 * (2 * 4 * 4) * 5
    assert observed["observed_linear_conv_training_forward_macs"] == conv + linear
    assert observed["analytical_mha_training_forward_macs"] == 0
    assert observed["recorded_training_forward_batches"] == 2
    assert observed["recorded_training_samples"] == 3
    assert observed["complete"] is True
    assert observed["full_training_flops"] is None
    assert observed["backward_flops"] is None
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert model.training
    assert all(torch.equal(before_state[k], v) for k, v in model.state_dict().items())
    assert not model.conv._forward_hooks
    assert not model.head._forward_hooks


def test_attention_counts_dense_shape_cost_once_in_training_forward():
    class Tokens(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.MultiheadAttention(8, 2, batch_first=True, dropout=0)

        def forward(self, tokens):
            y, _ = self.attention(tokens, tokens, tokens, need_weights=False)
            return y.sum()

    model = Tokens()
    with VisualTrainingForwardMACCounter(model) as observer:
        with observer.record_batch(2):
            result = model(torch.ones((2, 3, 8)))
            result.backward()
        # An extra forward used by another algorithm is intentionally excluded.
        model(torch.ones((2, 3, 8)))
    report = observer.snapshot()
    qkvout = 4 * 2 * 3 * 8 * 8
    qk_av = 2 * 2 * 3 * 3 * 8
    assert report["observed_linear_conv_training_forward_macs"] == 0
    assert report["analytical_mha_training_forward_macs"] == qkvout + qk_av
    assert report["partial_training_forward_macs"] == qkvout + qk_av
    assert report["recorded_training_forward_batches"] == 1
    assert not model.attention._forward_hooks
    assert not model.attention.out_proj._forward_hooks


def test_failed_forward_discards_uncommitted_partial_batch_and_removes_hooks():
    class Fails(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 1)

        def forward(self, x):
            self.conv(x)
            raise RuntimeError("deliberate in-forward failure")

    model = Fails()
    observer = VisualTrainingForwardMACCounter(model)
    with pytest.raises(RuntimeError, match="deliberate"):
        with observer:
            with observer.record_batch(1):
                model(torch.ones((1, 3, 4, 4)))
    assert observer.snapshot()["recorded_training_forward_batches"] == 0
    assert observer.snapshot()["partial_training_forward_macs"] == 0
    assert not model.conv._forward_hooks


def test_signed_exact_resume_and_missing_prior_counter_are_distinct():
    model = nn.Linear(4, 3)
    with VisualTrainingForwardMACCounter(model) as first:
        with first.record_batch(2):
            model(torch.ones(2, 4))
    state = first.snapshot()
    assert state["observed_linear_conv_training_forward_macs"] == 24

    with VisualTrainingForwardMACCounter(
        model, resume=state, prior_batches=1,
    ) as second:
        with second.record_batch(1):
            model(torch.ones(1, 4))
    resumed = second.snapshot()
    assert resumed["recorded_training_forward_batches"] == 2
    assert resumed["recorded_training_samples"] == 3
    assert resumed["observed_linear_conv_training_forward_macs"] == 36
    assert resumed["complete"] is True

    with VisualTrainingForwardMACCounter(model, prior_batches=3) as unknown:
        with unknown.record_batch(1):
            model(torch.ones(1, 4))
    assert unknown.snapshot()["complete"] is False
    assert unknown.snapshot()["recorded_training_forward_batches"] == 1

    with pytest.raises(ValueError, match="does not match"):
        VisualTrainingForwardMACCounter(model, resume=state, prior_batches=2)
    with pytest.raises(ValueError, match="format"):
        VisualTrainingForwardMACCounter(model, resume={"format": "wrong"}, prior_batches=1)


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_invalid_batch_sizes_rejected(invalid):
    model = nn.Linear(3, 2)
    with VisualTrainingForwardMACCounter(model) as observer:
        with pytest.raises(ValueError, match="positive integer"):
            with observer.record_batch(invalid):
                model(torch.ones(1, 3))


def test_nested_observations_are_rejected_without_leaking_first_batch():
    model = nn.Linear(3, 2)
    with VisualTrainingForwardMACCounter(model) as observer:
        with observer.record_batch(1):
            with pytest.raises(RuntimeError, match="nested"):
                with observer.record_batch(1):
                    pass
            model(torch.ones(1, 3))
    assert observer.snapshot()["recorded_training_forward_batches"] == 1
    assert observer.snapshot()["partial_training_forward_macs"] == 6
