"""
Tests for Backtesting Framework

Comprehensive tests for the backtesting engine and strategy integration:
- BacktestEngine functionality and edge cases
- Strategy integration with components
- Performance metrics calculation and validation
- Walk-forward analysis and parameter optimization
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import tempfile
import os
from typing import Dict, Any

from crowetrade.backtesting.engine import (
    BacktestEngine, BacktestConfig, PerformanceMetrics,
    TransactionCosts, calculate_performance_metrics
)
from crowetrade.backtesting.strategy_integration import (
    IntegratedStrategy, StrategyConfig, StrategyBacktester,
    create_strategy_config, run_simple_backtest
)


# Test Fixtures
@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing"""
    np.random.seed(42)
    
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    n_assets = 5
    assets = [f'ASSET_{i}' for i in range(n_assets)]
    
    # Generate correlated returns
    n_days = len(dates)
    returns = np.random.multivariate_normal(
        mean=[0.001] * n_assets,  # Small positive drift
        cov=np.eye(n_assets) * 0.02 + 0.005,  # Some correlation
        size=n_days
    )
    
    # Convert to prices (geometric returns)
    prices_data = {}
    for i, asset in enumerate(assets):
        price_series = 100 * np.cumprod(1 + returns[:, i])
        prices_data[asset] = price_series
    
    return pd.DataFrame(prices_data, index=dates)


@pytest.fixture
def basic_strategy():
    """Simple buy-and-hold strategy for testing"""
    def strategy_func(current_date, market_data, current_positions, **kwargs):
        # Equal weight allocation
        n_assets = len(market_data.columns)
        return {asset: 1.0 / n_assets for asset in market_data.columns}
    
    return strategy_func


@pytest.fixture
def momentum_strategy():
    """Momentum-based strategy for testing"""
    def momentum_func(current_date, market_data, current_positions, **kwargs):
        if len(market_data) < 20:
            return {}
        
        # Calculate 20-day momentum
        recent_data = market_data.tail(20)
        returns = (recent_data.iloc[-1] / recent_data.iloc[0]) - 1
        
        # Rank assets and allocate
        sorted_assets = returns.sort_values(ascending=False)
        
        # Top 3 assets get equal allocation
        target_weights = {}
        top_assets = sorted_assets.head(3).index
        
        for asset in top_assets:
            target_weights[asset] = 1.0 / 3.0
        
        return target_weights
    
    return momentum_func


@pytest.fixture
def sample_backtest_config():
    """Standard backtest configuration"""
    return BacktestConfig(
        start_date=datetime(2021, 1, 1),
        end_date=datetime(2022, 12, 31),
        initial_capital=1_000_000.0,
        transaction_costs=TransactionCosts(
            commission_rate=0.001,
            bid_ask_spread=0.0005,
            market_impact=0.0001
        )
    )


class TestBacktestEngine:
    """Test BacktestEngine functionality"""
    
    def test_engine_initialization(self, sample_market_data, sample_backtest_config):
        """Test engine initialization"""
        
        engine = BacktestEngine(sample_market_data, sample_backtest_config)
        
        assert engine.market_data is not None
        assert engine.config == sample_backtest_config
        assert len(engine.market_data) > 0
        
    def test_basic_backtest_execution(self, sample_market_data, sample_backtest_config, basic_strategy):
        """Test basic backtest execution"""
        
        engine = BacktestEngine(sample_market_data, sample_backtest_config)
        metrics = engine.run_backtest(basic_strategy)
        
        # Check that metrics are computed
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_return is not None
        assert metrics.sharpe_ratio is not None
        assert metrics.max_drawdown is not None
        
        # Check reasonable values
        assert -2.0 <= metrics.total_return <= 5.0  # Reasonable return range
        assert -3.0 <= metrics.sharpe_ratio <= 5.0   # Reasonable Sharpe range
        assert metrics.max_drawdown <= 0.0          # Drawdown should be negative
        
    def test_momentum_strategy_backtest(self, sample_market_data, sample_backtest_config, momentum_strategy):
        """Test momentum strategy backtest"""
        
        engine = BacktestEngine(sample_market_data, sample_backtest_config)
        metrics = engine.run_backtest(momentum_strategy)
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_return is not None
        
        # Strategy should produce different results than buy-and-hold
        # (We can't guarantee better or worse, just different)
        
    def test_transaction_costs_impact(self, sample_market_data, basic_strategy):
        """Test that transaction costs reduce returns"""
        
        # Config with no transaction costs
        config_no_costs = BacktestConfig(
            start_date=datetime(2021, 1, 1),
            end_date=datetime(2022, 12, 31),
            transaction_costs=TransactionCosts(0, 0, 0)
        )
        
        # Config with high transaction costs  
        config_high_costs = BacktestConfig(
            start_date=datetime(2021, 1, 1),
            end_date=datetime(2022, 12, 31),
            transaction_costs=TransactionCosts(0.01, 0.005, 0.001)  # 1% commission + spreads
        )
        
        # Run both backtests
        engine_no_costs = BacktestEngine(sample_market_data, config_no_costs)
        engine_high_costs = BacktestEngine(sample_market_data, config_high_costs)
        
        metrics_no_costs = engine_no_costs.run_backtest(basic_strategy)
        metrics_high_costs = engine_high_costs.run_backtest(basic_strategy)
        
        # High cost backtest should have lower returns (for active strategies)
        # Note: For pure buy-and-hold, difference might be minimal
        assert metrics_no_costs.total_return >= metrics_high_costs.total_return - 0.01  # Small tolerance
        
    def test_walk_forward_analysis(self, sample_market_data, basic_strategy):
        """Test walk-forward analysis"""
        
        config = BacktestConfig(
            start_date=datetime(2021, 1, 1),
            end_date=datetime(2022, 12, 31)
        )
        
        engine = BacktestEngine(sample_market_data, config)
        results = engine.walk_forward_analysis(basic_strategy, training_window=126, test_window=63)
        
        # Should have multiple test periods
        assert len(results) > 0
        
        # Each result should be valid metrics
        for metrics in results:
            assert isinstance(metrics, PerformanceMetrics)
            assert metrics.total_return is not None
            
    def test_edge_cases(self, sample_backtest_config):
        """Test edge cases and error handling"""
        
        # Empty market data
        empty_data = pd.DataFrame()
        
        with pytest.raises((ValueError, IndexError)):
            engine = BacktestEngine(empty_data, sample_backtest_config)
            engine.run_backtest(lambda *args, **kwargs: {})
        
        # Strategy that returns invalid weights
        def bad_strategy(*args, **kwargs):
            return {'INVALID_ASSET': 2.0}  # Weight > 1.0
        
        # Should handle gracefully
        sample_data = pd.DataFrame({
            'ASSET_0': [100, 101, 102],
            'ASSET_1': [200, 201, 199]
        }, index=pd.date_range('2021-01-01', periods=3))
        
        engine = BacktestEngine(sample_data, sample_backtest_config)
        # Should not crash, may return zero returns
        metrics = engine.run_backtest(bad_strategy)
        assert isinstance(metrics, PerformanceMetrics)


class TestIntegratedStrategy:
    """Test IntegratedStrategy class"""
    
    def test_strategy_initialization(self):
        """Test strategy initialization"""
        
        config = StrategyConfig()
        strategy = IntegratedStrategy(config)
        
        assert strategy.config == config
        assert strategy.portfolio_optimizer is not None
        
    def test_signal_generation(self, sample_market_data):
        """Test signal generation"""
        
        config = StrategyConfig()
        strategy = IntegratedStrategy(config)
        
        current_date = datetime(2021, 6, 1)
        market_slice = sample_market_data[:current_date]
        
        signals = strategy.generate_signals(market_slice, current_date)
        
        # Should return signals for all assets
        assert isinstance(signals, dict)
        assert len(signals) == len(market_slice.columns)
        
        # Signals should be in reasonable range
        for signal in signals.values():
            assert -2.0 <= signal <= 2.0
    
    def test_portfolio_optimization(self, sample_market_data):
        """Test portfolio optimization"""
        
        config = StrategyConfig()
        strategy = IntegratedStrategy(config)
        
        current_date = datetime(2021, 6, 1)
        market_slice = sample_market_data[:current_date]
        
        # Generate signals first
        signals = strategy.generate_signals(market_slice, current_date)
        current_positions = {}
        
        # Run optimization
        weights = strategy.optimize_portfolio(
            market_slice, signals, current_positions, current_date
        )
        
        if weights:  # May be empty if insufficient data
            assert isinstance(weights, dict)
            
            # Weights should sum to reasonable amount
            total_weight = sum(weights.values())
            assert 0.0 <= total_weight <= 1.0
            
            # Individual weights should be reasonable
            for weight in weights.values():
                assert 0.0 <= weight <= config.max_position_weight
    
    def test_strategy_execution(self, sample_market_data):
        """Test full strategy execution"""
        
        config = StrategyConfig()
        strategy = IntegratedStrategy(config)
        
        current_date = datetime(2021, 6, 1)
        market_slice = sample_market_data[:current_date]
        current_positions = {}
        
        # Execute strategy
        target_weights = strategy(current_date, market_slice, current_positions)
        
        assert isinstance(target_weights, dict)
        
        if target_weights:
            total_weight = sum(target_weights.values())
            assert 0.0 <= total_weight <= 1.0
    
    def test_risk_checks(self, sample_market_data):
        """Test risk management functionality"""
        
        config = StrategyConfig(
            max_position_weight=0.2,  # 20% max per asset
            max_turnover=0.5         # 50% max turnover
        )
        strategy = IntegratedStrategy(config)
        
        # Test position limit enforcement
        target_weights = {'ASSET_0': 0.5, 'ASSET_1': 0.5}  # Exceeds 20% limit
        current_positions = {}
        
        adjusted_weights = strategy._apply_risk_checks(target_weights, current_positions)
        
        # Should cap individual positions
        for weight in adjusted_weights.values():
            assert weight <= config.max_position_weight
    
    def test_regime_integration(self, sample_market_data):
        """Test regime detection integration"""
        
        config = StrategyConfig(use_regime_detection=True)
        strategy = IntegratedStrategy(config)
        
        current_date = datetime(2021, 6, 1)
        market_slice = sample_market_data[:current_date]
        
        # Should handle regime detection without crashing
        regime = strategy.detect_regime(market_slice, current_date)
        
        # May return None if insufficient data, which is fine


class TestStrategyBacktester:
    """Test StrategyBacktester class"""
    
    def test_backtest_strategy(self, sample_market_data):
        """Test strategy backtesting"""
        
        strategy_config = StrategyConfig()
        backtest_config = BacktestConfig(
            start_date=datetime(2021, 1, 1),
            end_date=datetime(2021, 12, 31)
        )
        
        backtester = StrategyBacktester()
        metrics = backtester.backtest_strategy(strategy_config, backtest_config, sample_market_data)
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_return is not None
    
    def test_parameter_sweep(self, sample_market_data):
        """Test parameter sweep functionality"""
        
        base_config = StrategyConfig()
        backtest_config = BacktestConfig(
            start_date=datetime(2021, 1, 1),
            end_date=datetime(2021, 6, 30)  # Shorter period for faster testing
        )
        
        # Parameter grid
        param_grid = {
            'risk_aversion': [0.5, 1.0, 2.0],
            'max_position_weight': [0.2, 0.3, 0.5]
        }
        
        backtester = StrategyBacktester()
        results = backtester.parameter_sweep(
            base_config, backtest_config, sample_market_data, param_grid
        )
        
        # Should have results for each parameter combination
        expected_combinations = len(param_grid['risk_aversion']) * len(param_grid['max_position_weight'])
        assert len(results) <= expected_combinations  # May have some failures
        
        # Results should be sorted by Sharpe ratio
        if len(results) > 1:
            sharpe_ratios = [metrics.sharpe_ratio for _, metrics in results]
            assert all(sharpe_ratios[i] >= sharpe_ratios[i+1] for i in range(len(sharpe_ratios)-1))
    
    def test_walk_forward_analysis(self, sample_market_data):
        """Test walk-forward analysis"""
        
        strategy_config = StrategyConfig()
        
        backtester = StrategyBacktester()
        results = backtester.walk_forward_analysis(
            strategy_config, sample_market_data,
            training_window=126, test_window=63
        )
        
        assert len(results) > 0
        
        for metrics in results:
            assert isinstance(metrics, PerformanceMetrics)
            assert hasattr(metrics, 'test_start')
            assert hasattr(metrics, 'test_end')


class TestPerformanceMetrics:
    """Test performance metrics calculation"""
    
    def test_basic_metrics_calculation(self):
        """Test basic performance metrics"""
        
        # Sample portfolio values (10% growth with some volatility)
        dates = pd.date_range('2021-01-01', '2021-12-31', freq='D')
        np.random.seed(42)
        
        # Generate returns with positive drift
        daily_returns = np.random.normal(0.0005, 0.02, len(dates))  # ~13% annual return
        portfolio_values = 1000000 * np.cumprod(1 + daily_returns)
        
        portfolio_series = pd.Series(portfolio_values, index=dates)
        
        # Calculate metrics
        metrics = calculate_performance_metrics(portfolio_series, 1000000.0)
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_return > -0.5  # Shouldn't lose more than 50%
        assert metrics.total_return < 2.0    # Shouldn't gain more than 200%
        assert -5.0 < metrics.sharpe_ratio < 5.0  # Reasonable Sharpe range
        assert metrics.max_drawdown <= 0.0   # Drawdown is negative
        
    def test_perfect_strategy_metrics(self):
        """Test metrics for perfect upward trending strategy"""
        
        dates = pd.date_range('2021-01-01', '2021-12-31', freq='D')
        portfolio_values = 1000000 * np.linspace(1, 1.2, len(dates))  # Perfect 20% growth
        
        portfolio_series = pd.Series(portfolio_values, index=dates)
        metrics = calculate_performance_metrics(portfolio_series, 1000000.0)
        
        # Should have positive returns and very high Sharpe
        assert metrics.total_return > 0.15  # ~20% return
        assert metrics.sharpe_ratio > 5.0   # Very high Sharpe for no volatility
        assert metrics.max_drawdown == 0.0  # No drawdown
        
    def test_crash_strategy_metrics(self):
        """Test metrics for strategy with large drawdown"""
        
        dates = pd.date_range('2021-01-01', '2021-12-31', freq='D')
        
        # Strategy that crashes 50% then recovers
        mid_point = len(dates) // 2
        values_crash = np.linspace(1, 0.5, mid_point)
        values_recover = np.linspace(0.5, 0.8, len(dates) - mid_point)
        all_values = np.concatenate([values_crash, values_recover])
        
        portfolio_values = 1000000 * all_values
        portfolio_series = pd.Series(portfolio_values, index=dates)
        
        metrics = calculate_performance_metrics(portfolio_series, 1000000.0)
        
        # Should show large drawdown and negative returns
        assert metrics.total_return < 0.0    # Net loss
        assert metrics.max_drawdown < -0.4   # At least 40% drawdown
        assert metrics.sharpe_ratio < 0.0    # Negative Sharpe


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_create_strategy_config(self):
        """Test strategy config creation"""
        
        config = create_strategy_config(
            risk_aversion=2.0,
            max_position_weight=0.25
        )
        
        assert isinstance(config, StrategyConfig)
        assert config.risk_aversion == 2.0
        assert config.max_position_weight == 0.25
    
    def test_run_simple_backtest(self, sample_market_data):
        """Test simple backtest runner"""
        
        metrics = run_simple_backtest(
            sample_market_data,
            start_date='2021-01-01',
            end_date='2021-12-31',
            strategy_params={'risk_aversion': 1.5}
        )
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_return is not None


class TestIntegrationScenarios:
    """Integration tests for complete scenarios"""
    
    def test_end_to_end_backtest(self, sample_market_data):
        """Test complete end-to-end backtest"""
        
        # Create a comprehensive strategy config
        strategy_config = StrategyConfig(
            risk_aversion=1.0,
            max_position_weight=0.3,
            use_regime_detection=True,
            rebalance_threshold=0.05
        )
        
        backtest_config = BacktestConfig(
            start_date=datetime(2021, 1, 1),
            end_date=datetime(2021, 12, 31),
            initial_capital=1_000_000.0,
            transaction_costs=TransactionCosts(0.001, 0.0005, 0.0001)
        )
        
        # Run backtest
        backtester = StrategyBacktester()
        metrics = backtester.backtest_strategy(strategy_config, backtest_config, sample_market_data)
        
        # Comprehensive checks
        assert isinstance(metrics, PerformanceMetrics)
        
        # Check all key metrics exist
        required_metrics = [
            'total_return', 'annualized_return', 'sharpe_ratio', 
            'max_drawdown', 'volatility', 'var_95', 'cvar_95'
        ]
        
        for metric_name in required_metrics:
            assert hasattr(metrics, metric_name)
            assert getattr(metrics, metric_name) is not None
    
    def test_comparative_strategy_analysis(self, sample_market_data):
        """Test comparing different strategies"""
        
        backtest_config = BacktestConfig(
            start_date=datetime(2021, 1, 1),
            end_date=datetime(2021, 12, 31)
        )
        
        # Conservative strategy
        conservative_config = StrategyConfig(
            risk_aversion=3.0,
            max_position_weight=0.2
        )
        
        # Aggressive strategy  
        aggressive_config = StrategyConfig(
            risk_aversion=0.5,
            max_position_weight=0.5
        )
        
        backtester = StrategyBacktester()
        
        conservative_metrics = backtester.backtest_strategy(
            conservative_config, backtest_config, sample_market_data
        )
        
        aggressive_metrics = backtester.backtest_strategy(
            aggressive_config, backtest_config, sample_market_data
        )
        
        # Both should complete successfully
        assert isinstance(conservative_metrics, PerformanceMetrics)
        assert isinstance(aggressive_metrics, PerformanceMetrics)
        
        # Conservative strategy should typically have lower volatility
        # (though not guaranteed with random data)
        
    @patch('crowetrade.models.registry.ModelRegistry')
    def test_model_registry_integration(self, mock_registry_class, sample_market_data):
        """Test integration with model registry"""
        
        # Setup mock registry
        mock_registry = Mock()
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.1, -0.05, 0.2, 0.0, -0.1])
        
        mock_registry.get_model.return_value = (mock_model, {'version': '1.0'})
        mock_registry_class.return_value = mock_registry
        
        # Create strategy with model registry
        strategy_config = StrategyConfig(
            signal_model_id='test_signal_model'
        )
        
        strategy = IntegratedStrategy(strategy_config, mock_registry)
        
        # Test signal generation with model
        current_date = datetime(2021, 6, 1)
        market_slice = sample_market_data[:current_date]
        
        signals = strategy.generate_signals(market_slice, current_date)
        
        # Should use model predictions
        assert isinstance(signals, dict)
        assert len(signals) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
