"""Portfolio Manager Agent

Provides utility logic around assembling expected returns vectors from a
set of symbol predictions with a deterministic fallback default used when
an explicit model prediction is missing. Designed as a thin example of a
stateful agent inheriting from the core BaseAgent abstraction.

Current scope: only exposes `_get_expected_returns` used by tests to
validate partial prediction handling. Live event handling / optimization
loops can be layered on later.
"""
from __future__ import annotations

from typing import Iterable, Mapping

from crowetrade.core.agent import BaseAgent, AgentConfig


class PortfolioManagerAgent(BaseAgent):
    """Agent responsible for portfolio expected return preparation.

    Parameters
    ----------
    config : AgentConfig
        Standard agent configuration (id, policy, risk limits).
    default_expected_return : float, default 0.08
        Fallback expected return used when a symbol lacks a model
        prediction. Chosen (8%) purely for test illustration; adjust to
        align with research assumptions (e.g., annualized / periodized).
    """

    def __init__(self, config: AgentConfig, default_expected_return: float = 0.08):
        super().__init__(config)
        self.default_expected_return = float(default_expected_return)

    # ------------------------------------------------------------------
    # Core utility methods
    # ------------------------------------------------------------------
    def _get_expected_returns(
        self,
        symbols: Iterable[str],
        predictions: Mapping[str, float],
    ) -> dict[str, float]:
        """Build an expected returns dict for the provided symbols.

        Any symbol missing in `predictions` receives the configured
        `default_expected_return`.
        """
        out: dict[str, float] = {}
        for sym in symbols:
            out[sym] = float(predictions.get(sym, self.default_expected_return))
        return out

    # ------------------------------------------------------------------
    # BaseAgent abstract hooks (minimal implementations for now)
    # ------------------------------------------------------------------
    async def on_start(self):  # pragma: no cover - trivial
        return None

    async def on_stop(self):  # pragma: no cover - trivial
        return None

    async def on_error(self, error: Exception):  # pragma: no cover - trivial
        # Placeholder for future structured logging / alerting
        return None
