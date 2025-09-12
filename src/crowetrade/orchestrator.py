from __future__ import annotations

from crowetrade.core.contracts import FeatureVector
from crowetrade.monitoring.audit import AuditLogger


class Orchestrator:
    """Binds signal, portfolio, risk, and execution agents into an event-driven loop."""

    def __init__(self, signal_agent, portfolio_agent, risk_guard, exec_router, audit_logger: AuditLogger | None = None):
        self.signal_agent = signal_agent
        self.portfolio_agent = portfolio_agent
        self.risk_guard = risk_guard
        self.exec_router = exec_router
        self.audit = audit_logger

    async def on_feature_batch(
        self, fvecs: dict[str, FeatureVector], vol: dict[str, float], prices: dict[str, float]
    ) -> None:
        signals = {}
        for k, v in fvecs.items():
            sig = self.signal_agent.infer(v)
            if sig:
                signals[k] = sig
                if self.audit:
                    self.audit.log(
                        "signal.accepted",
                        {
                            "instrument": k,
                            "policy_id": sig.policy_id,
                            "policy_hash": getattr(sig, "policy_hash", None),
                            "mu": sig.mu,
                            "sigma": sig.sigma,
                            "pep": sig.prob_edge_pos,
                        },
                    )
            else:
                if self.audit:
                    self.audit.log("signal.gated", {"instrument": k})
        targets = self.portfolio_agent.size(signals, vol)
        var_est = sum(abs(q) * float(vol.get(k, 0.0)) for k, q in targets.items())
        exposure = sum(abs(q * float(prices.get(k, 0.0))) for k, q in targets.items())
        if not self.risk_guard.pretrade_check(exposure, var_est):
            if self.audit:
                self.audit.log("risk.block", {"exposure": exposure, "var_est": var_est})
            return
        if self.audit:
            self.audit.log("portfolio.targets", {"count": len(targets)})
        await self.exec_router.route(targets, prices)
        if self.audit:
            self.audit.log("execution.dispatched", {"orders": len(targets)})
