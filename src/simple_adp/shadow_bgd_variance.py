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

    def load_state_dict(self, state):
        expected = [self.initial_std, self.mc_samples, self.tolerance,
                    self.patience, self.seed]
        if state.get("schema") != 1 or state.get("config") != expected:
            raise ValueError("incompatible BGD shadow state")
        self.sigma = {n: t.detach().clone() for n, t in state["sigma"].items()}
        self.generators = {}
        for device, rng in state["rng"].items():
            gen = torch.Generator(device=device)
            gen.set_state(rng)
            self.generators[device] = gen
        self.streak = int(state["streak"])
        self.observations = int(state["observations"])
        self.contraction = state["contraction"]
