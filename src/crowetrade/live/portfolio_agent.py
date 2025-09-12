from __future__ import annotations

import numpy as np
from typing import Optional, Dict, TYPE_CHECKING

from crowetrade.core.contracts import Signal
try:  # pragma: no cover - import guarded
    from crowetrade.services.decision_service.optimizer import PortfolioOptimizer  # type: ignore
except Exception:  # pragma: no cover
    PortfolioOptimizer = None  # sentinel for runtime absence

if TYPE_CHECKING:  # for static type checkers only
    from crowetrade.services.decision_service.optimizer import PortfolioOptimizer as _PortfolioOptimizer


class PortfolioAgent:
    """Transforms signals into target sizes under a global risk budget.

    Simplified Kelly-tempered sizing using provided per-instrument volatility estimates.
    """

    def __init__(
        self,
        risk_budget: float,
        turnover_penalty: float = 0.0,
        lambda_temper: float = 0.25,
    optimizer: Optional["_PortfolioOptimizer"] = None,
    ):
        self.risk_budget = float(risk_budget)
        self.turnover_penalty = float(turnover_penalty)
        self.lambda_temper = float(lambda_temper)
        self.positions: dict[str, float] = {}
        self.optimizer = optimizer  # if provided, used for long-only allocation of positive-edge signals

    def size(self, signals: Dict[str, Signal], vol: Dict[str, float]) -> Dict[str, float]:
        if not signals:
            return {}

        # If no optimizer configured, fallback to Kelly-tempered sizing for all signals
        if self.optimizer is None:
            return self._kelly_tempered(signals, vol)

        # Optimizer currently supports only long-only weights. We therefore:
        #  - Allocate risk_budget * weights to signals with positive expected return (mu>0)
        #  - Apply Kelly-tempered sizing for negative mu signals (potentially short exposure)
        positive = [inst for inst, s in signals.items() if s.mu > 0]
        targets: Dict[str, float] = {}

        if positive:
            expected_returns = {inst: signals[inst].mu for inst in positive}
            # Build a diagonal covariance from provided vol (vol assumed to be std dev)
            covariance = {}
            for i in positive:
                vi = float(vol.get(i, 1e-6))
                covariance[(i, i)] = vi * vi
                for j in positive:
                    if i == j:
                        continue
                    # zero off-diagonal (no correlation assumption) for simplicity
                    covariance[(i, j)] = 0.0
            opt_port = self.optimizer.optimize(positive, expected_returns, covariance)
            for inst, w in opt_port.weights.items():
                targets[inst] = float(w * self.risk_budget)

        # Add sizing for negative mu signals (if any) via Kelly-tempered approach
        negatives = {k: s for k, s in signals.items() if s.mu <= 0}
        if negatives:
            neg_targets = self._kelly_tempered(negatives, vol)
            targets.update(neg_targets)

        return targets

    # ------------------------------------------------------------------
    def _kelly_tempered(self, signals: Dict[str, Signal], vol: Dict[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        lam = min(max(self.lambda_temper, 1e-6), 1.0)
        for k, s in signals.items():
            v = max(float(vol.get(k, 1e-6)), 1e-6)
            kelly = s.mu / (v + 1e-9)
            raw = lam * kelly
            scaled = np.clip(raw * self.risk_budget, -self.risk_budget, self.risk_budget)
            out[k] = float(scaled)
        return out
