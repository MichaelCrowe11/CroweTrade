"""Tests for RiskGuard with explicit PnL semantics"""
import pytest
from crowetrade.live_agents.risk_guard import RiskGuard, PnLUpdate, PnLMode


class TestRiskGuardExplicit:
    """Test RiskGuard with explicit incremental/absolute semantics"""
    
    def test_incremental_updates(self):
        """Test pure incremental PnL updates"""
        guard = RiskGuard()
        
        # Series of incremental updates
        guard.update_pnl_incremental(100)
        assert guard.pnl_state.cumulative == 100
        assert guard.pnl_state.high_water_mark == 100
        
        guard.update_pnl_incremental(50)
        assert guard.pnl_state.cumulative == 150
        assert guard.pnl_state.high_water_mark == 150
        
        guard.update_pnl_incremental(-30)
        assert guard.pnl_state.cumulative == 120
        assert guard.pnl_state.high_water_mark == 150  # HWM unchanged
        assert guard.pnl_state.drawdown == pytest.approx(0.2)  # 20% drawdown
    
    def test_absolute_updates(self):
        """Test absolute PnL updates (snapshots)"""
        guard = RiskGuard()
        
        # Absolute updates
        guard.update_pnl_absolute(100)
        assert guard.pnl_state.cumulative == 100
        assert guard.pnl_state.high_water_mark == 100
        
        guard.update_pnl_absolute(150)
        assert guard.pnl_state.cumulative == 150
        assert guard.pnl_state.high_water_mark == 150
        
        guard.update_pnl_absolute(120)
        assert guard.pnl_state.cumulative == 120
        assert guard.pnl_state.high_water_mark == 150  # HWM unchanged
        assert guard.pnl_state.drawdown == pytest.approx(0.2)  # 20% drawdown
    
    def test_mixed_updates(self):
        """Test mixing incremental and absolute updates"""
        guard = RiskGuard()
        
        # Start with absolute
        guard.update_pnl_absolute(1000)
        assert guard.pnl_state.cumulative == 1000
        
        # Add incremental
        guard.update_pnl_incremental(200)
        assert guard.pnl_state.cumulative == 1200
        
        # Another absolute (resets to this value)
        guard.update_pnl_absolute(1100)
        assert guard.pnl_state.cumulative == 1100
        
        # More incremental
        guard.update_pnl_incremental(-50)
        assert guard.pnl_state.cumulative == 1050
        
        assert guard.pnl_state.high_water_mark == 1200
        assert guard.pnl_state.drawdown == pytest.approx(0.125)  # 12.5% drawdown
    
    def test_daily_pnl_tracking(self):
        """Test daily PnL tracking with both update modes"""
        guard = RiskGuard()
        
        # Incremental updates affect daily PnL directly
        guard.update_pnl_incremental(100)
        assert guard.daily_pnl == 100
        
        guard.update_pnl_incremental(-30)
        assert guard.daily_pnl == 70
        
        # Absolute update computes delta for daily PnL
        guard.update_pnl_absolute(100)  # Was 70, now 100, delta = 30
        assert guard.daily_pnl == 100  # 70 + 30
        
        # Reset daily
        guard.reset_daily_pnl()
        assert guard.daily_pnl == 0
        assert guard.pnl_state.cumulative == 100  # Cumulative unchanged
    
    def test_kill_switch_with_explicit_modes(self):
        """Test kill switch activation with different update modes"""
        guard = RiskGuard(kill_switch_loss=1000)
        
        # Use absolute to set a starting point
        guard.update_pnl_absolute(500)
        assert not guard.kill_switch_active
        
        # Incremental losses
        guard.update_pnl_incremental(-700)
        assert guard.pnl_state.cumulative == -200
        assert not guard.kill_switch_active
        
        guard.update_pnl_incremental(-850)
        assert guard.pnl_state.cumulative == -1050
        assert guard.kill_switch_active  # Triggered
        
        # Recovery via absolute
        guard.update_pnl_absolute(-700)
        assert not guard.kill_switch_active  # Recovered (above 80% threshold: -700 > -1000 * 0.8 = -800)
        
        # Can re-trigger
        guard.update_pnl_absolute(-1100)
        assert guard.kill_switch_active  # Triggered again
    
    def test_pretrade_checks_with_drawdown(self):
        """Test pretrade checks consider drawdown correctly"""
        guard = RiskGuard(max_drawdown=0.1)  # 10% max drawdown
        
        # Set up a drawdown scenario
        guard.update_pnl_absolute(10000)  # Start with 10k
        guard.update_pnl_absolute(8900)   # Now at 8900, 11% drawdown
        
        # Should reject due to drawdown
        passed, reason = guard.pretrade_check("AAPL", 10, 150)
        assert not passed
        assert "Drawdown" in reason
        
        # Recover partially
        guard.update_pnl_incremental(200)  # Now at 9100, 9% drawdown
        
        # Should pass now
        passed, reason = guard.pretrade_check("AAPL", 10, 150)
        assert passed
    
    def test_structured_pnl_update(self):
        """Test using structured PnLUpdate objects"""
        guard = RiskGuard()
        
        # Create structured updates
        update1 = PnLUpdate(value=100, mode=PnLMode.ABSOLUTE, metadata={"source": "initial"})
        update2 = PnLUpdate(value=50, mode=PnLMode.INCREMENTAL, metadata={"source": "trade1"})
        update3 = PnLUpdate(value=120, mode=PnLMode.ABSOLUTE, metadata={"source": "reconciliation"})
        
        guard.update_pnl(update1)
        assert guard.pnl_state.cumulative == 100
        
        guard.update_pnl(update2)
        assert guard.pnl_state.cumulative == 150
        
        guard.update_pnl(update3)
        assert guard.pnl_state.cumulative == 120
        
        assert guard.pnl_state.high_water_mark == 150


class TestPropertyBasedWithExplicitSemantics:
    """Property-based tests with clear semantics"""
    
    def test_incremental_property_hwm_monotonic(self):
        """HWM never decreases with incremental updates"""
        guard = RiskGuard()
        values = [10, -5, 20, -15, 30, -25]
        
        prev_hwm = 0
        for val in values:
            guard.update_pnl_incremental(val)
            assert guard.pnl_state.high_water_mark >= prev_hwm
            prev_hwm = guard.pnl_state.high_water_mark
    
    def test_absolute_property_hwm_is_max(self):
        """HWM equals max of all absolute values seen"""
        guard = RiskGuard()
        values = [100, 150, 120, 180, 160]
        
        for val in values:
            guard.update_pnl_absolute(val)
        
        assert guard.pnl_state.high_water_mark == max(values)
    
    def test_drawdown_bounds(self):
        """Drawdown always between 0 and 1"""
        guard = RiskGuard()
        
        # Various scenarios
        scenarios = [
            [PnLUpdate(100, PnLMode.ABSOLUTE), PnLUpdate(-20, PnLMode.INCREMENTAL)],
            [PnLUpdate(1000, PnLMode.ABSOLUTE), PnLUpdate(-2000, PnLMode.INCREMENTAL)],
            [PnLUpdate(-100, PnLMode.ABSOLUTE), PnLUpdate(200, PnLMode.INCREMENTAL)],
        ]
        
        for updates in scenarios:
            guard = RiskGuard()  # Fresh guard
            for update in updates:
                guard.update_pnl(update)
                assert 0 <= guard.pnl_state.drawdown <= 1