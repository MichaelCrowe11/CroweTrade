from __future__ import annotations

from datetime import UTC, datetime

from crowetrade.core.contracts import FeatureVector
from crowetrade.live.portfolio_agent import PortfolioAgent
from crowetrade.live.risk_guard import RiskGuard
from crowetrade.live.signal_agent import SignalAgent


def toy_model(values: dict[str, float]):
    # simple: mu = values.get('mom', 0), sigma fixed, pep from sign
    mu = float(values.get("mom", 0.0))
    sigma = 0.02
    pep = 0.6 if mu > 0 else 0.4
    return mu, sigma, pep


def make_fv(mu: float) -> FeatureVector:
    return FeatureVector(
        instrument="AAPL",
        asof=datetime.now(UTC),
        horizon="1d",
        values={"mom": mu},
        quality={"lag_ms": 100, "coverage": 1.0},
    )


def test_signal_agent_gating():
    sa = SignalAgent(
        model=toy_model, policy_id="p1", gates={"prob_edge_min": 0.55, "sigma_max": 0.03}
    )
    s1 = sa.infer(make_fv(0.001))
    assert s1 is not None and s1.mu > 0
    s2 = sa.infer(make_fv(-0.001))
    assert s2 is None  # fails prob gate


def test_portfolio_agent_sizing():
    sa = SignalAgent(
        model=toy_model, policy_id="p1", gates={"prob_edge_min": 0.55, "sigma_max": 0.03}
    )
    sig = sa.infer(make_fv(0.002))
    assert sig is not None
    pa = PortfolioAgent(risk_budget=1.0, lambda_temper=0.25)
    targets = pa.size({"AAPL": sig}, {"AAPL": 0.02})
    assert "AAPL" in targets
    assert abs(targets["AAPL"]) <= 1.0


def test_risk_guard_checks():
    rg = RiskGuard(dd_limit=0.02, var_limit=0.01)
    rg.update_pnl(0.0)
    rg.update_pnl(-0.01)  # 1% drawdown
    assert rg.pretrade_check(exposure=1000.0, var_est=0.005) is True
    assert rg.pretrade_check(exposure=1000.0, var_est=0.02) is False
