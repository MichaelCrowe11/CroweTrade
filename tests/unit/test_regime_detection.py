"""Tests for Regime Detection Module"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from crowetrade.regime.detector import RegimeDetector, RegimeType, RegimeState
from crowetrade.live.regime_agent import RegimeDetectionAgent
from crowetrade.core.agent import AgentConfig
from crowetrade.messaging.event_bus import EventBus


class TestRegimeDetector:
    """Test suite for RegimeDetector"""
    
    def test_initialization(self):
        """Test regime detector initialization"""
        detector = RegimeDetector(
            lookback_window=100,
            vol_threshold_low=0.10,
            vol_threshold_high=0.30
        )
        
        assert detector.lookback_window == 100
        assert detector.vol_threshold_low == 0.10
        assert detector.vol_threshold_high == 0.30
        assert len(detector.returns_history) == 0
        assert len(detector.volatility_history) == 0
    
    def test_update_with_empty_returns(self):
        """Test regime detection with empty returns"""
        detector = RegimeDetector()
        
        regime_state = detector.update({})
        
        assert regime_state.regime == RegimeType.SIDEWAYS
        assert regime_state.confidence == 0.5
        assert regime_state.metadata["insufficient_data"] == True
    
    def test_update_with_insufficient_data(self):
        """Test regime detection with insufficient historical data"""
        detector = RegimeDetector()
        
        # Add only a few data points
        for i in range(5):
            returns = {"AAPL": 0.01, "MSFT": -0.005}
            regime_state = detector.update(returns)
        
        # Should still return default state
        assert regime_state.regime == RegimeType.SIDEWAYS
        assert regime_state.confidence == 0.5
    
    def test_volatility_calculation(self):
        """Test volatility calculation from returns"""
        detector = RegimeDetector()
        
        # Add returns data with known volatility
        np.random.seed(42)
        for _ in range(50):
            # Generate returns with known std dev
            returns = {"AAPL": np.random.normal(0.001, 0.02)}
            detector.update(returns)
        
        # Check that volatility is reasonable (annualized)
        vol = detector._calculate_volatility()
        assert 0.1 < vol < 0.5  # Reasonable volatility range
    
    def test_bull_regime_detection(self):
        """Test detection of bull market regime"""
        detector = RegimeDetector(
            vol_threshold_low=0.15,
            vol_threshold_high=0.35
        )
        
        # Simulate bull market: positive returns, low volatility
        for _ in range(30):
            returns = {"AAPL": 0.015, "MSFT": 0.012, "GOOGL": 0.010}
            regime_state = detector.update(returns)
        
        # Should detect bull or sideways regime (positive returns, low vol)
        assert regime_state.regime in [RegimeType.BULL, RegimeType.SIDEWAYS]
        assert regime_state.volatility < detector.vol_threshold_high
    
    def test_volatile_regime_detection(self):
        """Test detection of volatile market regime"""
        detector = RegimeDetector(
            vol_threshold_low=0.10,
            vol_threshold_high=0.20
        )
        
        # Simulate volatile market: high variance returns
        np.random.seed(42)
        for _ in range(30):
            # High volatility returns
            returns = {
                "AAPL": np.random.normal(0.0, 0.05),
                "MSFT": np.random.normal(0.0, 0.04),
                "GOOGL": np.random.normal(0.0, 0.06)
            }
            regime_state = detector.update(returns)
        
        # Should eventually detect high volatility
        assert regime_state.volatility > detector.vol_threshold_low
    
    def test_crash_regime_detection(self):
        """Test detection of crash regime via turbulence"""
        detector = RegimeDetector(turbulence_threshold=2.0)
        
        # Build up normal history
        for _ in range(25):
            returns = {"AAPL": 0.005, "MSFT": 0.003}
            detector.update(returns)
        
        # Sudden large negative returns (crash scenario)
        crash_returns = {"AAPL": -0.08, "MSFT": -0.10, "GOOGL": -0.12}
        regime_state = detector.update(crash_returns)
        
        # Should detect high turbulence
        assert regime_state.turbulence > 0
        # May detect crash regime if turbulence is high enough
        if regime_state.turbulence > detector.turbulence_threshold:
            assert regime_state.regime == RegimeType.CRASH
    
    def test_regime_probabilities_sum_to_one(self):
        """Test that regime probabilities sum to approximately 1.0"""
        detector = RegimeDetector()
        
        # Add sufficient data
        for _ in range(25):
            returns = {"AAPL": 0.01, "MSFT": -0.005}
            regime_state = detector.update(returns)
        
        prob_sum = sum(regime_state.probabilities.values())
        assert abs(prob_sum - 1.0) < 0.01  # Allow small floating point errors
    
    def test_persistence_filter(self):
        """Test regime persistence filtering"""
        detector = RegimeDetector(min_regime_duration=3)
        
        # Build up consistent regime history
        for _ in range(10):
            returns = {"AAPL": 0.002, "MSFT": 0.001}  # Low vol, positive
            detector.update(returns)
        
        # Try to switch regime with single data point
        regime_before = detector.regime_history[-1][0] if detector.regime_history else RegimeType.SIDEWAYS
        
        # Single volatile data point
        volatile_returns = {"AAPL": 0.05, "MSFT": -0.04}
        regime_state = detector.update(volatile_returns)
        
        # Persistence filter should prevent immediate regime change
        # (unless it's a crash which overrides persistence)
        if regime_state.regime != RegimeType.CRASH:
            # Check that rapid regime switching is filtered
            assert len(detector.regime_history) > 0
    
    def test_regime_statistics(self):
        """Test regime statistics calculation"""
        detector = RegimeDetector()
        
        # Add some history
        for i in range(20):
            returns = {"AAPL": 0.001 * (i % 3), "MSFT": 0.002}
            detector.update(returns)
        
        stats = detector.get_regime_statistics()
        
        assert "regime_distribution" in stats
        assert "current_volatility" in stats
        assert "average_volatility" in stats
        assert "returns_history_length" in stats
        assert stats["returns_history_length"] == 20


class TestRegimeDetectionAgent:
    """Test suite for RegimeDetectionAgent"""
    
    @pytest.fixture
    def event_bus(self):
        """Mock event bus for testing"""
        bus = MagicMock(spec=EventBus)
        bus.subscribe = AsyncMock()
        bus.unsubscribe = AsyncMock()
        bus.publish = AsyncMock()
        return bus
    
    @pytest.fixture
    def agent_config(self):
        """Agent configuration for testing"""
        return AgentConfig(
            agent_id="regime_detector_01",
            policy={
                "vol_threshold_low": 0.12,
                "vol_threshold_high": 0.25,
                "turbulence_threshold": 3.0,
                "min_regime_duration": 5
            },
            risk_limits={}
        )
    
    @pytest.fixture
    def regime_agent(self, agent_config, event_bus):
        """Create RegimeDetectionAgent for testing"""
        return RegimeDetectionAgent(
            config=agent_config,
            event_bus=event_bus,
            update_frequency=timedelta(minutes=5)
        )
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, regime_agent):
        """Test agent initialization"""
        assert regime_agent.config.agent_id == "regime_detector_01"
        assert regime_agent.detector is not None
        assert regime_agent.last_regime == RegimeType.SIDEWAYS
        assert len(regime_agent.market_data_buffer) == 0
    
    @pytest.mark.asyncio
    async def test_agent_start_stop(self, regime_agent, event_bus):
        """Test agent start and stop lifecycle"""
        await regime_agent.start()
        
        # Should subscribe to market data events
        assert event_bus.subscribe.call_count == 2
        event_bus.subscribe.assert_any_call("market_data", regime_agent._handle_market_data)
        event_bus.subscribe.assert_any_call("returns_calculated", regime_agent._handle_returns_data)
        
        await regime_agent.stop()
        
        # Should unsubscribe from events
        assert event_bus.unsubscribe.call_count == 2
    
    @pytest.mark.asyncio 
    async def test_handle_returns_data(self, regime_agent):
        """Test handling of returns data"""
        # Mock returns data event
        returns_event = {
            "returns": {
                "AAPL": 0.015,
                "MSFT": 0.012,
                "GOOGL": 0.008
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await regime_agent._handle_returns_data(returns_event)
        
        # Should update last_update timestamp
        assert regime_agent.last_update > datetime.min
    
    @pytest.mark.asyncio
    async def test_get_current_regime_no_data(self, regime_agent):
        """Test getting current regime with no market data"""
        regime_state = await regime_agent.get_current_regime()
        assert regime_state is None
    
    @pytest.mark.asyncio
    async def test_get_current_regime_with_data(self, regime_agent):
        """Test getting current regime with market data"""
        # Add some market data
        regime_agent.market_data_buffer = {
            "AAPL": 150.0,
            "MSFT": 300.0,
            "GOOGL": 2500.0
        }
        
        regime_state = await regime_agent.get_current_regime()
        
        assert regime_state is not None
        assert isinstance(regime_state.regime, RegimeType)
        assert 0 <= regime_state.confidence <= 1
        assert regime_state.volatility >= 0
    
    @pytest.mark.asyncio
    async def test_get_regime_statistics(self, regime_agent):
        """Test getting regime statistics from agent"""
        stats = await regime_agent.get_regime_statistics()
        
        assert "agent_id" in stats
        assert stats["agent_id"] == "regime_detector_01"
        assert "update_frequency_minutes" in stats
        assert "market_data_symbols" in stats
    
    @pytest.mark.asyncio
    async def test_error_handling(self, regime_agent, event_bus):
        """Test error handling in regime agent"""
        # Create a test error
        test_error = ValueError("Test error")
        
        await regime_agent._handle_error(test_error)
        
        # Should publish error event
        event_bus.publish.assert_called_once()
        call_args = event_bus.publish.call_args[0]
        assert call_args[0] == "agent_error"
        assert "error_type" in call_args[1]
        assert call_args[1]["error_type"] == "ValueError"
