from __future__ import annotations

from crowetrade.core.contracts import FeatureVector


class Orchestrator:
    """Binds signal, portfolio, risk, and execution agents into an event-driven loop."""

    def __init__(self, signal_agent, portfolio_agent, risk_guard, exec_router):
        self.signal_agent = signal_agent
        self.portfolio_agent = portfolio_agent
        self.risk_guard = risk_guard
        self.exec_router = exec_router

    async def on_feature_batch(
        self, fvecs: dict[str, FeatureVector], vol: dict[str, float], prices: dict[str, float]
    ) -> None:
        signals = {k: s for k, v in fvecs.items() if (s := self.signal_agent.infer(v))}
        targets = self.portfolio_agent.size(signals, vol)
        var_est = sum(abs(q) * float(vol.get(k, 0.0)) for k, q in targets.items())
        exposure = sum(abs(q * float(prices.get(k, 0.0))) for k, q in targets.items())
        if not self.risk_guard.pretrade_check(exposure, var_est):
            return
        await self.exec_router.route(targets, prices)
