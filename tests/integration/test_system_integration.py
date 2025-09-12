"""Integration Tests for CroweTrade System Components

This module tests the integration between Portfolio Optimizer, Regime Detection,
and Execution Scheduler to ensure they work together properly for live trading.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from crowetrade.services.decision_service.optimizer import (
    PortfolioOptimizer, OptimizationConfig, OptimizationMethod
)
from crowetrade.regime.detector import RegimeDetector, RegimeType
from crowetrade.execution.scheduler import (
    ExecutionScheduler, ExecutionParams, ExecutionStrategy, create_execution_params
)
from crowetrade.live.portfolio_manager import PortfolioManagerAgent
from crowetrade.live.regime_agent import RegimeDetectionAgent
from crowetrade.core.agent import AgentConfig


class TestCroweTradeIntegration:
    """Integration tests for the complete trading system"""
    
    @pytest.fixture
    def sample_market_data(self):
        """Sample market data for testing"""
        return {
            "AAPL": {"price": 150.0, "volume": 2_000_000, "volatility": 0.18},
            "MSFT": {"price": 300.0, "volume": 1_500_000, "volatility": 0.16}, 
            "GOOGL": {"price": 2500.0, "volume": 800_000, "volatility": 0.22},
            "TSLA": {"price": 800.0, "volume": 3_000_000, "volatility": 0.35}
        }
    
    @pytest.fixture
    def portfolio_optimizer(self):
        """Create portfolio optimizer for testing"""
        config = OptimizationConfig(
            method=OptimizationMethod.MEAN_VARIANCE,
            constraints={"min_weight": 0.05, "max_weight": 0.40},
            risk_aversion=1.0,
            rebalance_threshold=0.05
        )
        return PortfolioOptimizer(config)
    
    @pytest.fixture
    def regime_detector(self):
        """Create regime detector for testing"""
        return RegimeDetector(
            lookback_window=50,
            vol_threshold_low=0.15,
            vol_threshold_high=0.25,
            turbulence_threshold=2.5
        )
    
    @pytest.fixture
    def execution_scheduler(self):
        """Create execution scheduler for testing"""
        return ExecutionScheduler(
            default_participation_rate=0.08,
            market_impact_coefficient=0.1,
            temporary_impact_coefficient=0.3
        )
    
    def test_portfolio_optimization_integration(self, portfolio_optimizer, sample_market_data):
        """Test portfolio optimization with realistic market data"""
        assets = list(sample_market_data.keys())
        expected_returns = {
            "AAPL": 0.12, "MSFT": 0.10, "GOOGL": 0.08, "TSLA": 0.15
        }
        
        # Create covariance matrix
        covariance_matrix = {}
        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                if i == j:
                    vol = sample_market_data[asset1]["volatility"]
                    covariance_matrix[(asset1, asset2)] = vol ** 2
                else:
                    vol1 = sample_market_data[asset1]["volatility"]
                    vol2 = sample_market_data[asset2]["volatility"]
                    covariance_matrix[(asset1, asset2)] = 0.3 * vol1 * vol2  # 30% correlation
        
        # Optimize portfolio
        portfolio = portfolio_optimizer.optimize(
            assets=assets,
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix
        )
        
        # Verify optimization results
        assert portfolio is not None
        assert len(portfolio.weights) == len(assets)
        assert abs(sum(portfolio.weights.values()) - 1.0) < 0.01
        assert portfolio.expected_return > 0
        assert portfolio.expected_risk > 0
        assert portfolio.sharpe_ratio >= 0
        
        # Check constraint compliance
        for weight in portfolio.weights.values():
            assert 0.05 <= weight <= 0.40
    
    def test_regime_detection_integration(self, regime_detector, sample_market_data):
        """Test regime detection with market data"""
        # Simulate market returns over time
        returns_history = []
        
        # Bull market phase
        for _ in range(20):
            returns = {"AAPL": 0.008, "MSFT": 0.006, "GOOGL": 0.004, "TSLA": 0.010}
            regime_state = regime_detector.update(returns)
            returns_history.append((regime_state.regime, regime_state.volatility))
        
        # Should detect bull or sideways market
        recent_regimes = [r[0] for r in returns_history[-5:]]
        assert any(r in [RegimeType.BULL, RegimeType.SIDEWAYS] for r in recent_regimes)
        
        # Volatile market phase
        for _ in range(15):
            returns = {"AAPL": 0.025, "MSFT": -0.020, "GOOGL": 0.015, "TSLA": -0.030}
            regime_state = regime_detector.update(returns)
            returns_history.append((regime_state.regime, regime_state.volatility))
        
        # Should detect increased volatility
        final_regime_state = regime_detector.update(returns)
        assert final_regime_state.volatility > 0.1
        
        # Verify regime statistics
        stats = regime_detector.get_regime_statistics()
        assert "regime_distribution" in stats
        assert stats["returns_history_length"] == 35
    
    def test_execution_scheduling_integration(self, execution_scheduler, sample_market_data):
        """Test execution scheduling for portfolio rebalancing"""
        # Create execution parameters for portfolio rebalancing
        rebalancing_orders = [
            ("AAPL", 500, "twap"),    # Buy 500 shares AAPL with TWAP
            ("MSFT", -200, "vwap"),   # Sell 200 shares MSFT with VWAP
            ("GOOGL", 50, "is"),      # Buy 50 shares GOOGL with Implementation Shortfall
            ("TSLA", -100, "pov")     # Sell 100 shares TSLA with POV
        ]
        
        execution_plans = []
        
        for symbol, quantity, strategy in rebalancing_orders:
            market_data = {
                "price": sample_market_data[symbol]["price"],
                "volume": sample_market_data[symbol]["volume"],
                "volatility": sample_market_data[symbol]["volatility"],
                "spread": sample_market_data[symbol]["price"] * 0.001,
                "time_elapsed_today": timedelta(hours=2)
            }
            
            params = create_execution_params(
                strategy_name=strategy,
                target_quantity=quantity,
                time_horizon_minutes=30,
                max_participation_rate=0.10,
                urgency=0.6
            )
            
            plan = execution_scheduler.create_execution_plan(symbol, params, market_data)
            execution_plans.append((symbol, plan))
        
        # Verify all plans created successfully
        assert len(execution_plans) == 4
        
        for symbol, plan in execution_plans:
            assert plan.total_quantity != 0
            assert len(plan.slices) > 0
            assert plan.expected_cost >= 0
            assert plan.expected_risk >= 0
            
            # Verify slices sum to target
            total_slices = sum(slice.quantity for slice in plan.slices)
            target = next(q for s, q, _ in rebalancing_orders if s == symbol)
            assert abs(total_slices - target) < 0.01
    
    def test_regime_adaptive_optimization(self, portfolio_optimizer, regime_detector):
        """Test regime-aware portfolio optimization"""
        assets = ["AAPL", "MSFT", "GOOGL"]
        base_returns = {"AAPL": 0.10, "MSFT": 0.08, "GOOGL": 0.06}
        
        # Create base covariance matrix
        covariance_matrix = {}
        for asset1 in assets:
            for asset2 in assets:
                if asset1 == asset2:
                    covariance_matrix[(asset1, asset2)] = 0.04  # 20% volatility
                else:
                    covariance_matrix[(asset1, asset2)] = 0.012  # 30% correlation
        
        # Test bull market regime
        bull_returns = {k: v * 1.2 for k, v in base_returns.items()}  # Higher expected returns
        bull_portfolio = portfolio_optimizer.optimize(assets, bull_returns, covariance_matrix)
        
        # Test bear market regime  
        bear_returns = {k: v * 0.5 for k, v in base_returns.items()}  # Lower expected returns
        bear_portfolio = portfolio_optimizer.optimize(assets, bear_returns, covariance_matrix)
        
        # Bull market should have higher expected return
        assert bull_portfolio.expected_return > bear_portfolio.expected_return
        
        # Test volatile market (adjust risk aversion)
        volatile_config = OptimizationConfig(
            method=OptimizationMethod.MIN_VARIANCE,  # Focus on risk minimization
            constraints={"min_weight": 0.05, "max_weight": 0.40},
            risk_aversion=2.0,  # Higher risk aversion
            rebalance_threshold=0.05
        )
        volatile_optimizer = PortfolioOptimizer(volatile_config)
        volatile_portfolio = volatile_optimizer.optimize(assets, base_returns, covariance_matrix)
        
        # Volatile market optimization should have lower risk
        assert volatile_portfolio.expected_risk <= bull_portfolio.expected_risk
    
    def test_full_system_workflow(self, portfolio_optimizer, regime_detector, execution_scheduler):
        """Test complete trading workflow: regime detection -> optimization -> execution"""
        assets = ["AAPL", "MSFT", "GOOGL", "TSLA"]
        
        # Step 1: Market data arrives
        market_returns = {"AAPL": 0.015, "MSFT": 0.008, "GOOGL": -0.005, "TSLA": 0.020}
        
        # Step 2: Update regime detection
        regime_state = regime_detector.update(market_returns)
        
        # Step 3: Adjust optimization parameters based on regime
        if regime_state.regime == RegimeType.VOLATILE:
            risk_aversion = 2.0
            method = OptimizationMethod.MIN_VARIANCE
        elif regime_state.regime == RegimeType.BULL:
            risk_aversion = 0.8
            method = OptimizationMethod.MAX_SHARPE
        else:
            risk_aversion = 1.0
            method = OptimizationMethod.MEAN_VARIANCE
        
        # Step 4: Portfolio optimization with regime-adjusted parameters
        config = OptimizationConfig(
            method=method,
            constraints={"min_weight": 0.1, "max_weight": 0.4},
            risk_aversion=risk_aversion,
            rebalance_threshold=0.05
        )
        optimizer = PortfolioOptimizer(config)
        
        expected_returns = {"AAPL": 0.12, "MSFT": 0.10, "GOOGL": 0.08, "TSLA": 0.15}
        covariance_matrix = {
            ("AAPL", "AAPL"): 0.0324, ("MSFT", "MSFT"): 0.0256, 
            ("GOOGL", "GOOGL"): 0.0484, ("TSLA", "TSLA"): 0.1225,
            ("AAPL", "MSFT"): 0.0144, ("AAPL", "GOOGL"): 0.0198,
            ("AAPL", "TSLA"): 0.0315, ("MSFT", "GOOGL"): 0.0176,
            ("MSFT", "TSLA"): 0.0280, ("GOOGL", "TSLA"): 0.0385,
            # Symmetric entries
            ("MSFT", "AAPL"): 0.0144, ("GOOGL", "AAPL"): 0.0198,
            ("TSLA", "AAPL"): 0.0315, ("GOOGL", "MSFT"): 0.0176,
            ("TSLA", "MSFT"): 0.0280, ("TSLA", "GOOGL"): 0.0385
        }
        
        portfolio = optimizer.optimize(assets, expected_returns, covariance_matrix)
        
        # Step 5: Generate execution plans for portfolio rebalancing
        current_positions = {"AAPL": 0.20, "MSFT": 0.30, "GOOGL": 0.25, "TSLA": 0.25}
        target_weights = portfolio.weights
        
        # Calculate position changes needed
        total_portfolio_value = 1_000_000  # $1M portfolio
        execution_orders = []
        
        for asset in assets:
            current_weight = current_positions[asset]
            target_weight = target_weights[asset]
            weight_change = target_weight - current_weight
            
            if abs(weight_change) > 0.02:  # 2% threshold for rebalancing
                dollar_change = weight_change * total_portfolio_value
                price = {"AAPL": 150, "MSFT": 300, "GOOGL": 2500, "TSLA": 800}[asset]
                shares_change = dollar_change / price
                
                execution_orders.append((asset, shares_change))
        
        # Step 6: Create execution plans
        execution_plans = []
        for asset, quantity in execution_orders:
            if abs(quantity) >= 1:  # Minimum 1 share
                market_data = {
                    "price": {"AAPL": 150, "MSFT": 300, "GOOGL": 2500, "TSLA": 800}[asset],
                    "volume": 1_000_000,
                    "volatility": 0.20,
                    "spread": 0.05
                }
                
                # Choose execution strategy based on regime
                if regime_state.regime in [RegimeType.VOLATILE, RegimeType.CRASH]:
                    strategy = "is"  # Implementation Shortfall for volatile markets
                    urgency = 0.8
                elif regime_state.volatility > 0.25:
                    strategy = "vwap"  # VWAP for high volatility
                    urgency = 0.6
                else:
                    strategy = "twap"  # TWAP for normal conditions
                    urgency = 0.4
                
                params = create_execution_params(
                    strategy_name=strategy,
                    target_quantity=quantity,
                    time_horizon_minutes=60,
                    max_participation_rate=0.10,
                    urgency=urgency
                )
                
                plan = execution_scheduler.create_execution_plan(asset, params, market_data)
                execution_plans.append((asset, plan))
        
        # Verify complete workflow
        assert regime_state is not None
        assert portfolio is not None
        assert len(portfolio.weights) == len(assets)
        
        # Should have execution plans for assets needing rebalancing
        if execution_plans:
            for asset, plan in execution_plans:
                assert plan.total_quantity != 0
                assert len(plan.slices) > 0
                
                # Verify regime-appropriate strategy selection
                if regime_state.regime == RegimeType.VOLATILE:
                    assert plan.strategy in [ExecutionStrategy.IMPLEMENTATION_SHORTFALL, 
                                           ExecutionStrategy.VWAP]
        
        # Verify system coherence
        assert abs(sum(portfolio.weights.values()) - 1.0) < 0.01
        assert portfolio.expected_return > 0
        assert 0 <= regime_state.confidence <= 1
    
    @pytest.mark.asyncio
    async def test_agent_integration(self):
        """Test integration between Portfolio Manager and Regime Detection agents"""
        # Mock event bus
        event_bus = MagicMock()
        event_bus.subscribe = AsyncMock()
        event_bus.publish = AsyncMock()
        
        # Create agents
        portfolio_config = AgentConfig(
            agent_id="portfolio_mgr_01",
            policy={"rebalance_threshold": 0.05},
            risk_limits={}
        )
        
        regime_config = AgentConfig(
            agent_id="regime_detector_01", 
            policy={
                "vol_threshold_low": 0.15,
                "vol_threshold_high": 0.25
            },
            risk_limits={}
        )
        
        portfolio_agent = PortfolioManagerAgent(portfolio_config, 0.08)
        regime_agent = RegimeDetectionAgent(regime_config, event_bus)
        
        # Test agent interaction
        symbols = ["AAPL", "MSFT", "GOOGL"]
        predictions = {"AAPL": 0.12, "MSFT": 0.10}  # Missing GOOGL prediction
        
        expected_returns = portfolio_agent._get_expected_returns(symbols, predictions)
        
        # Should use default for missing prediction
        assert expected_returns["GOOGL"] == 0.08  # Default value
        assert expected_returns["AAPL"] == 0.12
        assert expected_returns["MSFT"] == 0.10
        
        # Test regime agent statistics
        stats = await regime_agent.get_regime_statistics()
        assert "agent_id" in stats
        assert stats["agent_id"] == "regime_detector_01"


if __name__ == "__main__":
    # Run basic integration test
    pytest.main([__file__, "-v"])
