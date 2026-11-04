"""Fail-closed binding of persisted runner job identity and loaded split.

A canonical run ID is a deterministic label, not a cryptographic signature.
This module rejects internally contradictory job specifications and split
archives before data is used for training. It does not attest that the
whole manifest came from an authorized publisher.
"""
from __future__ import annotations

from collections.abc import Mapping

from .pair_audit import VARIANCE_SUFFIX


def validate_rebuilt_spec(raw, identity, route) -> None:
    """Check that serialized labels and executable route/settings agree."""
    if not isinstance(raw, Mapping):
        raise ValueError("worker run specification must be an object")
    recorded = raw.get("identity")
    if not isinstance(recorded, Mapping) or recorded.get("run_id") != identity.run_id:
        raise ValueError("persisted worker run_id does not match canonical RunIdentity")
    route_data = raw.get("route")
    if not isinstance(route_data, Mapping) or route_data.get("route_label") != route.route_label:
        raise ValueError("persisted route_label does not match executable ModelRoute")
    settings = raw.get("worker_settings")
    if not isinstance(settings, Mapping):
        raise ValueError("persisted worker_settings must be a mapping")
    if "seed" in settings and (type(settings["seed"]) is not int
                               or settings["seed"] != identity.seed):
        raise ValueError("worker seed disagrees with its canonical identity")
    arm = "counterpart_of" in settings
    if arm:
        if not isinstance(settings["counterpart_of"], str) or not settings["counterpart_of"]:
            raise ValueError("Bayesian counterpart_of must name its source run")
        if settings.get("expansion_criterion") != "variance_stagnation":
            raise ValueError("Bayesian counterpart must use variance_stagnation")
        if settings.get("variance_posterior") != "shadow_bgd_sigma_not_exact_bayes":
            raise ValueError("Bayesian counterpart has an invalid posterior mode marker")
        if route.model_id == "MLP-Fixed" or route.screening_only:
            raise ValueError("fixed/screening model cannot be a Bayesian growth counterpart")
    else:
        if settings.get("expansion_criterion", "loss") != "loss":
            raise ValueError("noncounterpart worker cannot silently change growth criterion")
        if "variance_posterior" in settings:
            raise ValueError("original worker cannot claim shadow posterior state")
    required_route = route.route_label + (VARIANCE_SUFFIX if arm else "")
    if identity.model_route != required_route:
        raise ValueError("canonical worker model_route differs from the executable route")


def validate_loaded_split(split, identity, *, row_count: int) -> None:
    """Bind split contents to job identity and reject out-of-source row indices."""
    if split.split_sha256 != identity.split_sha256:
        raise ValueError("loaded split archive checksum differs from worker RunIdentity")
    if any(
        not isinstance(index, int) or index < 0 or index >= row_count
        for partition in (split.train_indices, split.validation_indices, split.test_indices)
        for index in partition
    ):
        raise ValueError("loaded split includes a row outside source data")
