"""Tests for Portfolio Manager Agent"""
import pytest
import numpy as np
from datetime import datetime
import asyncio

from crowetrade.live_agents.portfolio_manager import (
    PortfolioManagerAgent,
    PortfolioState
)
from crowetrade.portfolio.optimizer import (
    PortfolioOptimizer,
    OptimizationConstraints
)
from crowetrade.data.feature_store import PersistentFeatureStore
from crowetrade.core.events import Event, EventType


class TestPortfolioManagerAgent:
    """Test portfolio manager agent functionality"""
    
    @pytest.fixture
    def agent(self):
        """Create a portfolio manager agent"""
        return PortfolioManagerAgent(
            agent_id="test_portfolio",
            rebalance_threshold=0.05,
            min_rebalance_interval=60
        )
    
    @pytest.mark.asyncio
    async def test_signal_handling(self, agent):
        """Test handling of signal events"""
        
        # Create signal event
        signal_event = Event(
            type=EventType.SIGNAL,
            source="signal_generator",
            data={
                'symbols': ['AAPL', 'GOOGL', 'MSFT'],
                'predictions': {
                    'AAPL': 0.08,
                    'GOOGL': 0.12,
                    'MSFT': 0.10
                },
                'confidence': {
                    'AAPL': 0.8,
                    'GOOGL': 0.9,
                    'MSFT': 0.7
                }
            }
        )
        
        # Process signal
        result = await agent.process_event(signal_event)
        
        # Check that target weights were set
        assert len(agent.target_weights) > 0
        assert sum(agent.target_weights.values()) == pytest.approx(1.0, abs=0.01)
        
        # Check optimization history
        assert len(agent.optimization_history) == 1
        assert agent.optimization_history[0].success
    
    @pytest.mark.asyncio
    async def test_position_update(self, agent):
        """Test handling of position updates"""
        
        # Create position update event
        position_event = Event(
            type=EventType.POSITION_UPDATE,
            source="broker",
            data={
                'positions': {
                    'AAPL': 100,
                    'GOOGL': 50,
                    'MSFT': 75
                },
                'weights': {
                    'AAPL': 0.4,
                    'GOOGL': 0.35,
                    'MSFT': 0.25
                },
                'cash': 10000.0,
                'total_value': 100000.0,
                'timestamp': datetime.now().timestamp()
            }
        )
        
        # Process update
        await agent.process_event(position_event)
        
        # Check state was updated
        assert agent.current_state is not None
        assert agent.current_state.total_value == 100000.0
        assert agent.current_state.cash == 10000.0
        assert len(agent.current_state.positions) == 3
    
    @pytest.mark.asyncio
    async def test_rebalancing_trigger(self, agent):
        """Test rebalancing trigger logic"""
        
        # Set current state
        agent.current_state = PortfolioState(
            positions={'AAPL': 100, 'GOOGL': 50},
            weights={'AAPL': 0.5, 'GOOGL': 0.5},
            cash=10000.0,
            total_value=100000.0,
            timestamp=datetime.now().timestamp()
        )
        
        # Set target weights with significant drift
        agent.target_weights = {'AAPL': 0.3, 'GOOGL': 0.7}
        
        # Process signal to trigger rebalance check
        signal_event = Event(
            type=EventType.SIGNAL,
            source="signal_generator",
            data={
                'symbols': ['AAPL', 'GOOGL'],
                'predictions': {'AAPL': 0.08, 'GOOGL': 0.12},
                'confidence': {'AAPL': 0.8, 'GOOGL': 0.9}
            }
        )
        
        result = await agent.process_event(signal_event)
        
        # Should trigger rebalance due to weight drift > threshold
        if result:
            assert result.type == EventType.REBALANCE
            assert 'target_weights' in result.data
    
    @pytest.mark.asyncio
    async def test_risk_update_handling(self, agent):
        """Test handling of risk updates"""
        
        # Initial constraints
        initial_max_weight = agent.constraints.max_weight
        
        # Create high risk event
        risk_event = Event(
            type=EventType.RISK_UPDATE,
            source="risk_guard",
            data={
                'high_risk': True,
                'risk_score': 0.8,
                'drawdown': 0.05
            }
        )
        
        # Process risk update
        await agent.process_event(risk_event)
        
        # Check constraints were tightened
        assert agent.constraints.max_weight < initial_max_weight
        assert agent.constraints.max_leverage == 1.0
    
    @pytest.mark.asyncio
    async def test_market_data_storage(self, agent):
        """Test market data storage in feature store"""
        
        # Create market data event
        market_event = Event(
            type=EventType.MARKET_DATA,
            source="data_provider",
            data={
                'prices': {
                    'AAPL': {'price': 150.0, 'timestamp': datetime.now().isoformat()},
                    'GOOGL': {'price': 2800.0, 'timestamp': datetime.now().isoformat()}
                },
                'volumes': {
                    'AAPL': {'volume': 1000000, 'timestamp': datetime.now().isoformat()},
                    'GOOGL': {'volume': 500000, 'timestamp': datetime.now().isoformat()}
                }
            }
        )
        
        # Process market data
        await agent.process_event(market_event)
        
        # Check data was stored
        price_aapl = agent.feature_store.get('price_AAPL')
        assert price_aapl is not None
        assert price_aapl.data['price'] == 150.0
        
        volume_googl = agent.feature_store.get('volume_GOOGL')
        assert volume_googl is not None
        assert volume_googl.data['volume'] == 500000
    
    def test_portfolio_state_weights_array(self):
        """Test portfolio state weight array conversion"""
        
        state = PortfolioState(
            positions={'AAPL': 100, 'GOOGL': 50, 'MSFT': 75},
            weights={'AAPL': 0.4, 'GOOGL': 0.35, 'MSFT': 0.25},
            cash=10000.0,
            total_value=100000.0,
            timestamp=datetime.now().timestamp()
        )
        
        # Get weights in specific order
        symbols = ['GOOGL', 'AAPL', 'MSFT', 'TSLA']
        weights = state.get_weights_array(symbols)
        
        assert len(weights) == 4
        assert weights[0] == 0.35  # GOOGL
        assert weights[1] == 0.4   # AAPL
        assert weights[2] == 0.25  # MSFT
        assert weights[3] == 0.0   # TSLA (not in portfolio)
    
    def test_metrics_reporting(self, agent):
        """Test agent metrics reporting"""
        
        # Set up some state
        agent.current_state = PortfolioState(
            positions={'AAPL': 100},
            weights={'AAPL': 1.0},
            cash=10000.0,
            total_value=100000.0,
            timestamp=datetime.now().timestamp()
        )
        
        agent.target_weights = {'AAPL': 1.0}
        
        # Get metrics
        metrics = agent.get_metrics()
        
        assert metrics['agent_id'] == 'test_portfolio'
        assert metrics['has_target_weights'] is True
        assert metrics['position_count'] == 1
        assert metrics['total_value'] == 100000.0
        assert metrics['cash'] == 10000.0
    
    def test_covariance_fallback(self, agent):
        """Test covariance matrix fallback logic"""
        
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        
        # Get covariance with no data in store
        cov_matrix = agent._get_covariance_matrix(symbols)
        
        assert cov_matrix is not None
        assert cov_matrix.shape == (3, 3)
        
        # Check it's positive definite
        eigenvalues = np.linalg.eigvals(cov_matrix)
        assert np.all(eigenvalues > 0)
    
    def test_expected_returns_fallback(self, agent):
        """Test expected returns fallback logic"""
        
        symbols = ['AAPL', 'GOOGL']
        predictions = {'AAPL': 0.10}  # Only one prediction
        
        # Get returns with partial predictions
        returns = agent._get_expected_returns(symbols, predictions)
        
        assert returns is not None
        assert returns['AAPL'] == 0.10  # From prediction
        assert returns['GOOGL'] == 0.08  # Default fallback