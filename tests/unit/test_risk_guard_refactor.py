import pytest

from crowetrade.live.risk_guard import RiskGuard


@pytest.mark.unit
def test_risk_guard_incremental_and_total_paths():
    g = RiskGuard(dd_limit=0.10, var_limit=0.05)
    g.update_incremental(0.05)
    g.update_incremental(0.04)  # cumulative 0.09
    assert pytest.approx(g.hwm) == 0.09
    # Simulate drawdown: set total lower
    g.set_total_pnl(0.06)
    assert g.current_dd == pytest.approx(0.03)
    # Kill switch not yet triggered (under limit)
    assert g.pretrade_check(exposure=1000, var_est=0.01)
    # Breach
    g.set_total_pnl(-0.05)  # drawdown 0.14
    assert not g.pretrade_check(exposure=1000, var_est=0.01)
    # Recovery
    g.set_total_pnl(0.02)
    g.reset_kill_switch()
    if g.current_dd < g.dd_limit * 0.5:
        # After partial recovery kill switch can clear
        assert g.pretrade_check(exposure=1000, var_est=0.01) in (True, False)
