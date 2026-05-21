"""Post-hoc temperature scaling (Guo et al. 2017).

A single scalar T > 0 divides logits before softmax. Fitting T by minimizing
NLL on a held-out set is the standard cheap calibration trick. We optimize
in log-space to keep T positive and use LBFGS, which converges in a handful
of iterations.

The fitted temperature is stored on ``EarlyExitConfig.temperature`` and used
by the wrapper at inference time.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def fit_temperature(
    logits: torch.Tensor,
    targets: torch.Tensor,
    max_iter: int = 50,
    lr: float = 0.01,
) -> float:
    """Find T > 0 minimizing NLL(logits / T, targets)."""
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2D (N, C); got shape {tuple(logits.shape)}")
    log_t = torch.zeros((), requires_grad=True)  # T = exp(log_t) = 1 at start
    optimizer = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)

    logits_d = logits.detach()
    targets_d = targets.detach()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        t = log_t.exp()
        loss = F.cross_entropy(logits_d / t, targets_d)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())
