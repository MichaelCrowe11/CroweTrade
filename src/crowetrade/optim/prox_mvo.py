from __future__ import annotations

"""Minimal proximal mean-variance optimizer (L2 only for now).

This is a simplified variant suitable for unit testing and initial integration.
Extensible to include L1/turnover if needed.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class MVOConfig:
    lam: float = 1.0  # risk aversion weight (higher -> more weight on mean)
    step: float = 0.05
    iters: int = 200
    tol: float = 1e-6


def prox_mvo(mu: np.ndarray, Sigma: np.ndarray, cfg: MVOConfig) -> np.ndarray:
    n = mu.shape[0]
    w = np.zeros(n)
    step = cfg.step
    for _ in range(cfg.iters):
        grad = Sigma @ w - cfg.lam * mu
        w_new = w - step * grad
        if np.linalg.norm(w_new - w, ord=np.inf) < cfg.tol:
            return w_new
        w = w_new
    return w
