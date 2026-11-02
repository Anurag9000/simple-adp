"""Exact, side-effect-free parameter accounting for composite visual ADP.

These are *parameter counts*, not performance estimates. Linear MACs are
per invocation of a local MLP, not full-image or end-to-end inference FLOPs:
a spatial/channel FFN is reused at every spatial position/token.
"""
from __future__ import annotations

from collections.abc import Mapping

from torch import nn

from .models import TabularMLP


def visual_parameter_accounting(
    model: nn.Module, growth_targets: Mapping[str, TabularMLP],
) -> dict[str, object]:
    """Partition model parameters by frozen, eligible, and other-trainable.

    Every growth target must be an actual, unique, trainable submodule of the
    model. This catches accidental reporting of an external proxy MLP or a
    convolution as a source-equivalent growth site. No tensors are allocated,
    model modes changed, forward passes performed or RNG states touched.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    if not growth_targets:
        raise ValueError("explicit nonempty local MLP growth targets are required")
    owned = {id(parameter): parameter for parameter in model.parameters()}
    all_total = sum(parameter.numel() for parameter in owned.values())
    trainable = sum(parameter.numel() for parameter in owned.values() if parameter.requires_grad)
    site_records: dict[str, dict[str, object]] = {}
    eligible_ids: set[int] = set()

    for site_id, target in sorted(growth_targets.items()):
        if not isinstance(site_id, str) or not site_id.strip():
            raise ValueError("growth-site IDs must be nonempty strings")
        if not isinstance(target, TabularMLP):
            raise TypeError(f"growth target {site_id!r} must be a TabularMLP")
        local_ids = {id(p) for p in target.parameters()}
        if not local_ids or not local_ids.issubset(owned):
            raise ValueError(f"growth target {site_id!r} is not an owned model submodule")
        if eligible_ids & local_ids:
            raise ValueError("distinct growth-site IDs must not alias parameters")
        flags = {bool(owned[p].requires_grad) for p in local_ids}
        if len(flags) != 1:
            raise ValueError(f"growth target {site_id!r} has partially frozen coordinates")
        eligible_ids.update(local_ids)
        sites_total = sum(owned[p].numel() for p in local_ids)
        linear_macs = sum(
            layer.weight.numel()
            for layer in target.modules()
            if isinstance(layer, nn.Linear)
        )
        site_records[site_id] = {
            "input_dim": int(target.input_dim),
            "output_dim": int(target.output_dim),
            "hidden_widths": list(target.hidden_widths),
            "parameter_count": int(sites_total),
            "currently_trainable": bool(next(iter(flags))),
            "linear_macs_per_mlp_invocation": int(linear_macs),
        }

    eligible = sum(owned[p].numel() for p in eligible_ids)
    active_eligible = sum(owned[p].numel() for p in eligible_ids if owned[p].requires_grad)
    return {
        "total_parameters": int(all_total),
        "frozen_parameters": int(all_total - trainable),
        "trainable_parameters": int(trainable),
        "eligible_mlp_parameters": int(eligible),
        "currently_trainable_eligible_mlp_parameters": int(active_eligible),
        "other_trainable_parameters": int(trainable - active_eligible),
        "eligible_growth_sites": site_records,
        "linear_compute_note": (
            "Linear MACs are per MLP invocation; spatial/token repetition "
            "and non-MLP operators are excluded, not whole-model FLOPs."
        ),
    }
