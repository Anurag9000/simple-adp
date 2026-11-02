"""V09 fixed hard-routing adapter bank, strict isolation, source growth and resume."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from simple_adp.config import SearchSettings
from simple_adp.visual_adapter_bank import BankSpec, VisualAdapterBank
from simple_adp.visual_native_ffn import build_visual_composite, restore_visual_composite
from simple_adp.visual_search import VisualResidualSearch


def spec(member: int) -> BankSpec:
    return BankSpec(
        input_channels=3, backbone_channels=4, local_dim=4, classes=2,
        image_height=8, image_width=12, site_id=f"bank:{member}",
        bank_size=3, fixed_other_width=5, fixed_seed=29,
    )


def data():
    rng = np.random.default_rng(82)
    x = rng.normal(size=(24, 3, 8, 12)).astype(np.float32) / 4
    y = np.asarray([0, 1] * 12, dtype=np.int64)
    x[y == 1] += 0.2
    return x[:16], y[:16], (x[16:], y[16:])


def settings():
    return SearchSettings(
        init_width=3, init_depth=1, max_width=5, max_depth=2,
        max_neurons=30, max_epochs=1, max_candidates=1,
        batch_size=4, ex_k_width=1,
        patience_es=1, patience_width_exp=1, patience_depth_exp=1,
        device="cpu", seed=68,
    )


@pytest.mark.parametrize("member", [0, 1, 2])
def test_v09_router_is_label_free_hard_gated_and_member_isolated(member):
    model = VisualAdapterBank(spec(member), (3,))
    sha = model.frozen_state_sha256()
    assert len(model.other_adapters) == 2
    assert model.growth_targets == {f"bank:{member}": model.local_mlp}
    assert all(not p.requires_grad for p in model.router.parameters())
    assert all(not p.requires_grad for p in model.other_adapters.parameters())
    assert all(p.requires_grad for p in model.local_mlp.parameters())
    assert all(not module.training for module in model._fixed_modules())
    assert any(isinstance(m, nn.GELU) for m in model.local_mlp.modules())

    torch.manual_seed(111)
    images = torch.randn(512, 3, 8, 12) * 4
    route = model.route_indices(images)
    assert route.shape == (512,)
    assert route.dtype == torch.int64
    assert int(route.min()) >= 0 and int(route.max()) < 3
    counts = model.routing_histogram(images)
    assert len(counts) == 3 and sum(counts) == len(images)
    # Routing does not depend on labels, order or on the trainable adapter.
    perm = torch.arange(len(images) - 1, -1, -1)
    assert torch.equal(model.route_indices(images[perm]), route[perm])
    score = model.router(model.pre(images).mean(dim=(-2, -1)))
    assert torch.equal(route, score.argmax(dim=-1))

    # Ensure we train on samples genuinely assigned to the selected member.
    chosen = images[route == member]
    assert len(chosen) > 0, "frozen router failed to cover a bank member"
    out = model(chosen[:8])
    assert out.shape[1] == 2
    out.square().mean().backward()
    assert model.local_mlp.output_layer.weight.grad is not None
    assert model.local_mlp.output_layer.weight.grad.abs().sum() > 0
    assert all(p.grad is None for m in model._fixed_modules() for p in m.parameters())
    assert model.frozen_state_sha256() == sha
    with torch.no_grad():
        model.local_mlp.output_layer.weight.add_(0.15)
    assert model.frozen_state_sha256() == sha
    assert torch.equal(model.route_indices(images), route)

    with pytest.raises(ValueError, match="fixed NCHW"):
        model(torch.randn(2, 3, 8, 13))
    with pytest.raises(ValueError, match="growth site"):
        model.grow_width(site_id="bank:99", operator="all_layers",
                         increment=1, max_width=5)


@pytest.mark.parametrize("member", [0, 1, 2])
def test_v09_native_width_depth_and_router_snapshot_exact(member):
    model = build_visual_composite(spec(member), (3, 3))
    baseline = model.frozen_state_sha256()
    before_router = model.router.weight.detach().clone()
    before = model.local_mlp.hidden_layers[0].weight.detach().clone()
    rr = model.grow_width(site_id=f"bank:{member}", operator="round_robin",
                          increment=1, max_width=6)
    assert rr is not None and rr.hidden_widths == [4, 3]
    assert torch.equal(rr.local_mlp.hidden_layers[0].weight[:3], before)
    assert torch.equal(rr.router.weight, before_router)
    assert rr.frozen_state_sha256() == baseline
    all_layers = model.grow_width(site_id=f"bank:{member}",
                                  operator="all_layers", increment=1, max_width=6)
    den = model.grow_width(site_id=f"bank:{member}",
                            operator="den_round_robin", increment=1, max_width=6)
    assert all_layers is not None and all_layers.hidden_widths == [4, 4]
    assert den is not None and den.hidden_widths == [4, 3]
    d1 = model.grow_depth(site_id=f"bank:{member}", operator="D1", max_depth=3)
    dc = model.grow_depth(site_id=f"bank:{member}", operator="Dcur", max_depth=3)
    assert d1 is not None and d1.hidden_widths == [3, 3, 1]
    assert dc is not None and dc.hidden_widths == [3, 3, 3]
    rng = torch.get_rng_state().clone()
    reopened = restore_visual_composite(rr.snapshot(cursor=rr._width_cursor))
    assert torch.equal(torch.get_rng_state(), rng)
    assert reopened.spec == model.spec
    assert reopened.frozen_state_sha256() == baseline
    assert reopened._width_cursor == 1
    x = torch.randn(4, 3, 8, 12)
    with torch.no_grad():
        assert torch.equal(reopened.eval()(x), rr.eval()(x))
        assert torch.equal(reopened.route_indices(x), rr.route_indices(x))
        reopened.other_adapters[next(iter(reopened.other_adapters))].output_layer.weight.add_(1)
    with pytest.raises(ValueError, match="frozen"):
        reopened.verify_frozen()


@pytest.mark.parametrize("member", [0, 1, 2])
def test_v09_selected_member_search_and_exact_interrupted_batch(member, tmp_path):
    x, y, valid = data()
    baseline = VisualResidualSearch("ADP-W-RR", spec(member), settings=settings())
    baseline.fit(x, y, validation_data=valid)
    assert [r["architecture"] for r in baseline.history_] == [[3], [4]]
    assert baseline.history_[-1]["site"] == f"bank:{member}"
    assert baseline.model_.frozen_state_sha256() == baseline.model_._frozen_baseline_sha256
    checkpoint = tmp_path / "bank.ckpt"

    class Evicted(Exception):
        pass

    def interrupt(record):
        if record["phase"] == "width" and record["stage_status"] == "active" and record["batches_done"] == 1:
            raise Evicted()

    with pytest.raises(Evicted):
        VisualResidualSearch("ADP-W-RR", spec(member), settings=settings()).fit(
            x, y, validation_data=valid, checkpoint_path=checkpoint,
            progress_callback=interrupt,
        )
    recovered = VisualResidualSearch("ADP-W-RR", spec(member), settings=settings())
    recovered.fit(x, y, validation_data=valid, checkpoint_path=checkpoint)
    assert recovered.history_ == baseline.history_
    assert recovered.result_.score == baseline.result_.score
    assert np.array_equal(recovered.predict_proba(valid[0]),
                          baseline.predict_proba(valid[0]))
    other = (member + 1) % 3
    with pytest.raises(ValueError, match="signature"):
        VisualResidualSearch("ADP-W-RR", spec(other), settings=settings()).fit(
            x, y, validation_data=valid, checkpoint_path=checkpoint)


@pytest.mark.parametrize("member", [0, 1, 2])
def test_v09_fixed_and_generic_phase_search(member):
    x, y, valid = data()
    control = VisualResidualSearch(
        "MLP-Fixed", spec(member), settings=settings(), fixed_grid=((3,), (4,)),
    )
    control.fit(x, y, validation_data=valid)
    assert [r["architecture"] for r in control.history_] == [[3], [4]]
    generic = VisualResidualSearch(
        "ADP-WD-RR-Dcur", spec(member), settings=settings(),
        generic_profile="corrected",
    )
    generic.fit(x, y, validation_data=valid)
    assert len(generic.history_) == 2
    assert generic.history_[1]["profile"] == "corrected"


def test_v09_rejects_nonexisting_member_and_invalid_bank():
    with pytest.raises(ValueError, match="existing"):
        BankSpec(3, 4, 4, 2, 8, 12, "bank:3", bank_size=3)
    with pytest.raises(ValueError, match="at least two"):
        BankSpec(3, 4, 4, 2, 8, 12, "bank:0", bank_size=1)
    with pytest.raises(ValueError, match="positive"):
        BankSpec(3, 4, 0, 2, 8, 12, "bank:0")
