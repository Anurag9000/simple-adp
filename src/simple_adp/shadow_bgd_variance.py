"""Auxiliary, checkpointable BGD-sigma observer for Adam/AdamW ADP variants.

Only sigma is updated using the BGD fixed-point positive root. Weight means,
optimizer gradients, model buffers and the native optimizer remain untouched.
This is a shadow approximation, NOT exact online variational Bayes.
"""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F
from torch.func import functional_call


class ShadowBGDVariance:
    def __init__(self, *, initial_std=0.5, mc_samples=2, tolerance=1e-4,
                 patience=3, seed=0):
        if not initial_std > 0 or mc_samples < 1 or patience < 1:
            raise ValueError("invalid BGD shadow variance settings")
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("invalid contraction tolerance")
        self.initial_std = float(initial_std)
        self.mc_samples = int(mc_samples)
        self.tolerance = float(tolerance)
        self.patience = int(patience)
        self.seed = int(seed)
        self.sigma = {}
        self.generators = {}
        self.streak = 0
        self.observations = 0
        self.contraction = None

    def _generator(self, device):
        key = str(device)
        if key not in self.generators:
            gen = torch.Generator(device=device)
            gen.manual_seed(self.seed)
            self.generators[key] = gen
        return self.generators[key]

    def _synchronize(self, model):
        result = {}
        for name, p in model.named_parameters():
            if not p.requires_grad or not p.is_floating_point():
                continue
            new = torch.full_like(p.detach(), self.initial_std)
            prev = self.sigma.get(name)
            if prev is not None and prev.ndim == new.ndim:
                sl = tuple(slice(0, min(a, b)) for a, b in zip(prev.shape, new.shape))
                new[sl] = prev.to(device=p.device, dtype=p.dtype)[sl]
            result[name] = new
        # Posterior values on overlapping coordinates remain useful after
        # widening, but their previous stagnation votes are not evidence for
        # the expanded architecture. Require a fresh persistence window.
        if self.sigma and (
            set(self.sigma) != set(result)
            or any(self.sigma[name].shape != tensor.shape
                   for name, tensor in result.items())
        ):
            self.streak = 0
            self.observations = 0
            self.contraction = None
        self.sigma = result

    def observe(self, model, x, y, loss_function=None):
        self._synchronize(model)
        params = {n: p for n, p in model.named_parameters() if n in self.sigma}
        if not params:
            self.contraction = None
            self.streak = 0
            return False
        before = {n: t.clone() for n, t in self.sigma.items()}
        cross = {n: torch.zeros_like(t) for n, t in before.items()}
        module_modes = {module: module.training for module in model.modules()}
        buffer_state = {n: b.detach().clone() for n, b in model.named_buffers()}
        try:
            model.eval()
            with torch.enable_grad():
                for _ in range(self.mc_samples):
                    eps = {
                        n: torch.randn(p.shape, device=p.device, dtype=p.dtype,
                                       generator=self._generator(p.device))
                        for n, p in params.items()
                    }
                    sampled = {
                        n: (p.detach() + before[n] * eps[n]).requires_grad_(True)
                        for n, p in params.items()
                    }
                    buffers = {n: b.clone() for n, b in buffer_state.items()}
                    logits = functional_call(model, (sampled, buffers), (x,))
                    loss = (F.cross_entropy(logits, y) if loss_function is None
                            else loss_function(logits, y))
                    if not bool(torch.isfinite(loss.detach()).item()):
                        self.contraction = None
                        self.streak = 0
                        return False
                    gradients = torch.autograd.grad(
                        loss, tuple(sampled.values()), allow_unused=True
                    )
                    for (n, _), grad in zip(sampled.items(), gradients):
                        if grad is not None:
                            cross[n] += grad.detach() * eps[n] / self.mc_samples
        finally:
            # Restore every child independently: calling model.train(mode)
            # would overwrite intentionally mixed training/eval submodules.
            for module, was_training in module_modes.items():
                module.training = was_training
        contraction_sum = 0.0
        total = 0
        new_sigma = {}
        for n, old in before.items():
            z = old * cross[n] / 2
            updated = old * torch.exp(-torch.asinh(z))
            if not bool(torch.isfinite(updated).all()):
                self.contraction = None
                self.streak = 0
                return False
            updated = updated.clamp(1e-7, 10.0)
            new_sigma[n] = updated
            contraction_sum += float(torch.log(
                (old.detach().double().square()+1e-12) /
                (updated.detach().double().square()+1e-12)
            ).sum().item())
            total += old.numel()
        self.sigma = new_sigma
        self.observations += 1
        self.contraction = contraction_sum / total
        self.streak = self.streak + 1 if self.contraction <= self.tolerance else 0
        return self.observations >= self.patience and self.streak >= self.patience

    def state_dict(self):
        return {
            "schema": 1,
            "config": [self.initial_std, self.mc_samples, self.tolerance,
                       self.patience, self.seed],
            "sigma": {n: t.detach().cpu().clone() for n, t in self.sigma.items()},
            "rng": {n: g.get_state().clone() for n, g in self.generators.items()},
            "streak": self.streak,
            "observations": self.observations,
            "contraction": self.contraction,
        }

    def load_state_dict(self, state, *, model=None):
        """Restore an intact posterior, checking model geometry on exact resume.

        Growth can migrate overlapping sigma via ``_synchronize``; checkpoint
        restoration must never silently substitute priors for missing neurons.
        """
        expected = [self.initial_std, self.mc_samples, self.tolerance,
                    self.patience, self.seed]
        if not isinstance(state, dict) or state.get("schema") != 1 or state.get("config") != expected:
            raise ValueError("incompatible BGD shadow state")
        raw_sigma = state.get("sigma")
        raw_rng = state.get("rng")
        if not isinstance(raw_sigma, dict) or not isinstance(raw_rng, dict):
            raise ValueError("invalid BGD shadow posterior or sampler state")
        sigma = {}
        for name, tensor in raw_sigma.items():
            if (not isinstance(name, str) or not name
                    or not torch.is_tensor(tensor)
                    or not tensor.is_floating_point()
                    or not bool(torch.isfinite(tensor).all().item())
                    or not bool((tensor > 0).all().item())):
                raise ValueError("invalid BGD shadow sigma coordinates")
            sigma[name] = tensor.detach().clone()
        try:
            observations = state["observations"]
            streak = state["streak"]
            contraction = state["contraction"]
        except KeyError as exc:
            raise ValueError("incomplete BGD shadow observation state") from exc
        if (type(observations) is not int or type(streak) is not int
                or observations < 0 or streak < 0 or streak > observations):
            raise ValueError("invalid BGD shadow observation counts")
        if (contraction is not None
                and (type(contraction) not in (float, int)
                     or not math.isfinite(contraction))):
            raise ValueError("invalid BGD shadow contraction evidence")
        if observations > 0 and (not sigma or contraction is None or not raw_rng):
            raise ValueError("incomplete BGD shadow posterior evidence")
        if model is not None and sigma:
            required = {
                name: param for name, param in model.named_parameters()
                if param.requires_grad and param.is_floating_point()
            }
            if sigma.keys() != required.keys():
                raise ValueError("BGD shadow posterior geometry differs from resumed model")
            for name, param in required.items():
                if sigma[name].shape != param.shape or sigma[name].dtype != param.dtype:
                    raise ValueError("BGD shadow posterior geometry differs from resumed model")
            if observations > 0 and set(raw_rng) != {str(p.device) for p in required.values()}:
                raise ValueError("BGD shadow sampler RNG devices differ from resumed model")
        generators = {}
        for device, rng in raw_rng.items():
            if (not isinstance(device, str) or not device
                    or not torch.is_tensor(rng) or rng.dtype != torch.uint8
                    or rng.ndim != 1):
                raise ValueError("invalid BGD shadow sampler RNG state")
            try:
                gen = torch.Generator(device=device)
                gen.set_state(rng)
            except (RuntimeError, ValueError, TypeError) as exc:
                raise ValueError("invalid BGD shadow sampler RNG state") from exc
            generators[device] = gen
        # Commit only after all validation: failures cannot partially restore.
        self.sigma = sigma
        self.generators = generators
        self.streak = streak
        self.observations = observations
        self.contraction = contraction
