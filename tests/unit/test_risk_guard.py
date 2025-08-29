import pytest
from hypothesis import given, strategies as st, assume, settings
import numpy as np

from crowetrade.live.risk_guard import RiskGuard


class TestRiskGuard:
    """Unit tests for RiskGuard with property-based testing."""
    
    @pytest.mark.unit
    def test_initialization(self):
        """Test RiskGuard initialization."""
        guard = RiskGuard(dd_limit=0.05, var_limit=0.02)
        assert guard.dd_limit == 0.05
        assert guard.var_limit == 0.02
        assert guard.hwm == 0.0
        assert guard.current_dd == 0.0
    
    @pytest.mark.unit
    def test_pnl_update_positive(self):
        """Test PnL update with positive returns."""
        guard = RiskGuard(dd_limit=0.05, var_limit=0.02)
        guard.update_pnl(0.01)
        guard.update_pnl(0.02)
        guard.update_pnl(0.015)
        
        assert guard.hwm == 0.03  # High water mark
        assert guard.current_dd == 0.015  # Drawdown from HWM
    
    @pytest.mark.unit
    def test_pnl_update_negative(self):
        """Test PnL update with negative returns."""
        guard = RiskGuard(dd_limit=0.05, var_limit=0.02)
        guard.update_pnl(0.02)
        guard.update_pnl(-0.01)
        guard.update_pnl(-0.005)
        
        assert guard.hwm == 0.02
        assert guard.current_dd == 0.015
    
    @pytest.mark.unit
    def test_pretrade_check_passes(self):
        """Test pre-trade check when within limits."""
        guard = RiskGuard(dd_limit=0.05, var_limit=0.02)
        guard.update_pnl(0.0)
        
        # Within limits
        assert guard.pretrade_check(exposure=1000, var_est=0.015) is True
    
    @pytest.mark.unit
    def test_pretrade_check_var_breach(self):
        """Test pre-trade check fails on VaR breach."""
        guard = RiskGuard(dd_limit=0.05, var_limit=0.02)
        guard.update_pnl(0.0)
        
        # VaR exceeds limit
        assert guard.pretrade_check(exposure=1000, var_est=0.025) is False
    
    @pytest.mark.unit
    def test_pretrade_check_drawdown_breach(self):
        """Test pre-trade check fails on drawdown breach."""
        guard = RiskGuard(dd_limit=0.05, var_limit=0.02)
        guard.update_pnl(0.10)  # Set high water mark
        guard.update_pnl(-0.06)  # Create 6% drawdown
        
        # Drawdown exceeds limit
        assert guard.pretrade_check(exposure=1000, var_est=0.01) is False
    
    @pytest.mark.unit
    @given(
        pnl_series=st.lists(
            st.floats(min_value=-0.1, max_value=0.1, allow_nan=False),
            min_size=1,
            max_size=100
        )
    )
    @settings(max_examples=50)
    def test_drawdown_calculation_invariants(self, pnl_series):
        """Property-based test for drawdown calculation invariants."""
        guard = RiskGuard(dd_limit=0.05, var_limit=0.02)
        
        cumulative_pnl = 0
        max_pnl = 0
        
        for pnl in pnl_series:
            guard.update_pnl(pnl)
            cumulative_pnl += pnl
            max_pnl = max(max_pnl, cumulative_pnl)
        
        # Invariants
        assert guard.hwm >= 0  # HWM can't be negative
        assert guard.current_dd >= 0  # Drawdown is non-negative
        assert guard.hwm == max_pnl  # HWM should track maximum
        
        if max_pnl > 0:
            assert guard.current_dd <= guard.hwm  # DD can't exceed HWM
    
    @pytest.mark.unit
    @given(
        dd_limit=st.floats(min_value=0.01, max_value=0.2, allow_nan=False),
        var_limit=st.floats(min_value=0.005, max_value=0.1, allow_nan=False),
        current_dd=st.floats(min_value=0, max_value=0.3, allow_nan=False),
        var_est=st.floats(min_value=0, max_value=0.2, allow_nan=False)
    )
    def test_pretrade_check_invariants(self, dd_limit, var_limit, current_dd, var_est):
        """Property-based test for pre-trade check invariants."""
        guard = RiskGuard(dd_limit=dd_limit, var_limit=var_limit)
        guard.current_dd = current_dd
        
        result = guard.pretrade_check(exposure=1000, var_est=var_est)
        
        # Check should fail if either limit is breached
        expected = (current_dd <= dd_limit) and (var_est <= var_limit)
        assert result == expected
    
    @pytest.mark.unit
    def test_kill_switch_activation(self):
        """Test kill switch activation on severe drawdown."""
        guard = RiskGuard(dd_limit=0.05, var_limit=0.02)
        guard.update_pnl(0.10)
        guard.update_pnl(-0.08)  # 8% drawdown
        
        # Should trigger kill switch
        assert guard.current_dd > guard.dd_limit
        assert guard.pretrade_check(exposure=100, var_est=0.001) is False
    
    @pytest.mark.unit
    def test_recovery_from_drawdown(self):
        """Test recovery from drawdown."""
        guard = RiskGuard(dd_limit=0.05, var_limit=0.02)
        guard.update_pnl(0.10)
        guard.update_pnl(-0.03)  # 3% drawdown
        guard.update_pnl(0.05)  # Recovery beyond HWM
        
        assert guard.hwm == 0.12
        assert guard.current_dd == 0  # No drawdown after recovery
        assert guard.pretrade_check(exposure=1000, var_est=0.015) is True
    
    @pytest.mark.unit
    def test_concurrent_limit_breaches(self):
        """Test behavior with multiple concurrent limit breaches."""
        guard = RiskGuard(dd_limit=0.05, var_limit=0.02)
        guard.update_pnl(0.10)
        guard.update_pnl(-0.06)  # 6% drawdown - exceeds limit
        
        # Both limits breached
        assert guard.pretrade_check(exposure=1000, var_est=0.025) is False
        
        # Only VaR breached
        assert guard.pretrade_check(exposure=1000, var_est=0.025) is False
        
        # Only DD breached
        assert guard.pretrade_check(exposure=1000, var_est=0.015) is False
        
        # Neither breached after recovery
        guard.update_pnl(0.07)  # Recovery
        assert guard.pretrade_check(exposure=1000, var_est=0.015) is True