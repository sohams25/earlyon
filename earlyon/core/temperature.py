"""Post-hoc temperature scaling (Guo et al. 2017).

A single scalar T > 0 divides logits before softmax. Fitting T by minimizing
NLL on a held-out set is the standard cheap calibration trick. We optimize
in log-space to keep T positive and use LBFGS, which converges in a handful
of iterations.

The fitted temperature is stored on ``EarlyExitConfig.temperature`` and used
by the wrapper at inference time.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def fit_temperature(
    logits: torch.Tensor,
    targets: torch.Tensor,
    max_iter: int = 50,
    lr: float = 1.0,
    max_outer_steps: int = 10,
    tol: float = 1e-6,
) -> float:
    """Find T > 0 minimizing NLL(logits / T, targets).

    ``lr`` defaults to 1.0 (the PyTorch LBFGS default); the previous 0.01
    consistently underfit, returning T well below the NLL optimum on
    overconfident models. We also iterate ``optimizer.step`` until ``log_t``
    stabilises — LBFGS occasionally needs more than one outer step — which is
    the standard usage pattern and the only way to actually converge.

    Raises ``ValueError`` on non-finite logits: a single NaN/Inf would
    otherwise propagate to ``T`` and silently poison every downstream softmax.
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2D (N, C); got shape {tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        raise ValueError("logits contain NaN or Inf; cannot fit temperature")

    log_t = torch.zeros((), requires_grad=True)  # T = exp(log_t) = 1 at start
    optimizer = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)

    logits_d = logits.detach()
    targets_d = targets.detach()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.cross_entropy(logits_d / log_t.exp(), targets_d)
        loss.backward()  # type: ignore[no-untyped-call]  # torch stub gap
        return loss

    # Track the last finite iterate. LBFGS at lr=1.0 can diverge on tiny or
    # degenerate batches, producing a NaN/Inf log_t; without this guard that
    # NaN would propagate to T and poison every downstream softmax. On
    # divergence we keep the last good value (T=1.0 if even the first step
    # blew up — i.e. no calibration rather than a broken one).
    best_log_t = 0.0
    prev = float("inf")
    for _ in range(max_outer_steps):
        optimizer.step(closure)  # type: ignore[no-untyped-call]  # torch stub gap
        current = log_t.item()
        if not math.isfinite(current):
            break
        best_log_t = current
        if abs(current - prev) < tol:
            break
        prev = current

    return float(math.exp(best_log_t))
