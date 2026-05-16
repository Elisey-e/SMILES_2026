from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn


class ZeroOrderOptimizer:
    """
    Weight-only MeZO + SignSGD.

    Hypothesis:
      - bias update overfits random mini-batch class priors;
      - most useful transfer comes from rotating class hyperplanes in fc.weight.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 6e-4,
        eps: float = 1e-3,
        n_samples: int = 16,
        beta1: float = 0.9,
        warmup_steps: int = 4,
        total_steps: int = 128,
        min_lr_ratio: float = 0.15,
    ) -> None:
        self.model = model
        self.base_lr = lr
        self.eps = eps
        self.n_samples = n_samples
        self.beta1 = beta1
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio

        self.layer_names: list[str] = ["fc.weight"]

        self._m: dict[str, torch.Tensor] = {}
        self._step_count = 0
        self._last_loss = 0.0

    def _active_params(self) -> dict[str, nn.Parameter]:
        named = dict(self.model.named_parameters())
        return {name: named[name] for name in self.layer_names}

    def _sample_z(self, p: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(p).bernoulli_(0.5).mul_(2.0).sub_(1.0)

    def _perturb_inplace(
        self,
        params: dict[str, nn.Parameter],
        seed: int,
        scale: float,
    ) -> None:
        torch.manual_seed(seed)
        for p in params.values():
            p.data.add_(self._sample_z(p), alpha=scale)

    def _current_lr(self) -> float:
        if self._step_count < self.warmup_steps:
            return self.base_lr * (self._step_count + 1) / max(1, self.warmup_steps)

        progress = (self._step_count - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.base_lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine)

    def _estimate_grad(
        self,
        loss_fn: Callable[[], float],
        params: dict[str, nn.Parameter],
    ) -> dict[str, torch.Tensor]:
        grads = {name: torch.zeros_like(p) for name, p in params.items()}
        losses = []

        for _ in range(self.n_samples):
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())

            self._perturb_inplace(params, seed, +self.eps)
            f_plus = float(loss_fn())

            self._perturb_inplace(params, seed, -2.0 * self.eps)
            f_minus = float(loss_fn())

            self._perturb_inplace(params, seed, +self.eps)

            proj = (f_plus - f_minus) / (2.0 * self.eps)
            losses.append(0.5 * (f_plus + f_minus))

            torch.manual_seed(seed)
            for name, p in params.items():
                grads[name].add_(self._sample_z(p), alpha=proj / self.n_samples)

        self._last_loss = sum(losses) / len(losses)
        return grads

    def _sign_update(
        self,
        params: dict[str, nn.Parameter],
        grads: dict[str, torch.Tensor],
    ) -> None:
        lr = self._current_lr()

        with torch.no_grad():
            for name, p in params.items():
                if name not in self._m:
                    self._m[name] = torch.zeros_like(p)

                m = self._m[name]
                m.mul_(self.beta1).add_(grads[name], alpha=1.0 - self.beta1)
                p.data.sub_(m.sign(), alpha=lr)

    def step(self, loss_fn: Callable[[], float]) -> float:
        params = self._active_params()
        grads = self._estimate_grad(loss_fn, params)
        self._sign_update(params, grads)
        self._step_count += 1
        return self._last_loss