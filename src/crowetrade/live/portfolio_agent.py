from __future__ import annotations

import numpy as np

from crowetrade.core.contracts import Signal


class PortfolioAgent:
    """Transforms signals into target sizes under a global risk budget.

    Simplified Kelly-tempered sizing using provided per-instrument volatility estimates.
    """

    def __init__(
        self, risk_budget: float, turnover_penalty: float = 0.0, lambda_temper: float = 0.25
    ):
        self.risk_budget = float(risk_budget)
        self.turnover_penalty = float(turnover_penalty)
        self.lambda_temper = float(lambda_temper)
        self.positions: dict[str, float] = {}

    def size(self, signals: dict[str, Signal], vol: dict[str, float]) -> dict[str, float]:
        targets: dict[str, float] = {}
        if not signals:
            return targets
        lam = self.lambda_temper
        for k, s in signals.items():
            v = max(float(vol.get(k, 1e-6)), 1e-6)
            kelly = s.mu / (v + 1e-9)
            targets[k] = float(
                np.clip(lam * kelly * self.risk_budget, -self.risk_budget, self.risk_budget)
            )
        return targets
