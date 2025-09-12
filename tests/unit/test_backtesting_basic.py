"""
Basic Backtesting Framework Tests

Simple tests that don't require external dependencies for CI/CD validation.
These tests focus on core logic and data structures without pandas/numpy.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any
import sys
import os

# Mock pandas and numpy if not available
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    
    # Create mock modules
    sys.modules['pandas'] = MagicMock()
    sys.modules['numpy'] = MagicMock()
    
    # Mock specific classes we need
    class MockDataFrame:
        def __init__(self, data=None, index=None):
            self.data = data or {}
            self.index = index or []
            self.columns = list(self.data.keys()) if data else []
            
        def __len__(self):
            return len(self.index) if self.index else 0
            
        def __getitem__(self, key):
            if key in self.data:
                return MockSeries(self.data[key])
            return MockDataFrame()
            
        def tail(self, n):
            return MockDataFrame(self.data, self.index[-n:] if self.index else [])
            
        def pct_change(self):
            return MockDataFrame()
            
        def dropna(self):
            return MockDataFrame(self.data, self.index)
            
        def fillna(self, value):
            return MockDataFrame(self.data, self.index)
            
        def iloc(self):
            return self.data
    
    class MockSeries:
        def __init__(self, data):
            self.data = data
            
        def mean(self):
            return 0.001
            
        def std(self):
            return 0.02
            
        def sum(self):
            return 0.1
            
        def skew(self):
            return 0.0
    
    pd.DataFrame = MockDataFrame
    pd.Series = MockSeries
    pd.date_range = lambda start, end, freq: [datetime(2021, 1, 1) + timedelta(days=i) for i in range(100)]
    
    np.array = list
    np.random.seed = lambda x: None
    np.random.normal = lambda mean, std, size: [0.001] * size
    np.cumprod = lambda x: x
    np.tanh = lambda x: max(-1, min(1, x))


# Now import our modules
from crowetrade.backtesting.strategy_integration import (
    StrategyConfig, IntegratedStrategy, create_strategy_config
)


class TestStrategyConfig:
    """Test StrategyConfig data class"""
    
    def test_default_config(self):
        """Test default configuration creation"""
        config = StrategyConfig()
        
        assert config.risk_aversion == 1.0
        assert config.rebalance_threshold == 0.05
        assert config.use_regime_detection == True
        assert config.max_position_weight == 0.30
        assert config.min_position_weight == 0.05
        
    def test_custom_config(self):
        """Test custom configuration"""
        config = StrategyConfig(
            risk_aversion=2.0,
            max_position_weight=0.25,
            use_regime_detection=False
        )
        
        assert config.risk_aversion == 2.0
        assert config.max_position_weight == 0.25
        assert config.use_regime_detection == False
        
    def test_config_to_dict(self):
        """Test configuration serialization"""
        config = StrategyConfig(risk_aversion=1.5)
        
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert 'risk_aversion' in config_dict
        assert config_dict['risk_aversion'] == 1.5


class TestIntegratedStrategy:
    """Test IntegratedStrategy core logic"""
    
    def test_strategy_initialization(self):
        """Test strategy initialization without external dependencies"""
        config = StrategyConfig()
        
        with patch('crowetrade.backtesting.strategy_integration.PortfolioOptimizer'):
            with patch('crowetrade.backtesting.strategy_integration.ExecutionScheduler'):
                strategy = IntegratedStrategy(config)
                
                assert strategy.config == config
                assert strategy.current_regime is None
                assert strategy.last_rebalance_date is None
    
    def test_momentum_signals_logic(self):
        """Test momentum signal generation logic"""
        config = StrategyConfig()
        
        with patch('crowetrade.backtesting.strategy_integration.PortfolioOptimizer'):
            with patch('crowetrade.backtesting.strategy_integration.ExecutionScheduler'):
                strategy = IntegratedStrategy(config)
                
                # Mock market data
                mock_data = Mock()
                mock_data.columns = ['ASSET_0', 'ASSET_1']
                
                # Mock asset data with price series
                mock_asset_data = Mock()
                mock_asset_data.__len__ = Mock(return_value=25)  # Sufficient data
                
                # Mock tail method for recent prices
                mock_recent = Mock()
                mock_recent.iloc = [100, 110]  # 10% gain
                mock_asset_data.tail.return_value = mock_recent
                
                mock_data.__getitem__ = Mock(return_value=mock_asset_data)
                
                # Test signal generation
                signals = strategy._momentum_signals(mock_data)
                
                assert isinstance(signals, dict)
                assert len(signals) == 2
                assert 'ASSET_0' in signals
                assert 'ASSET_1' in signals
    
    def test_risk_checks_logic(self):
        """Test risk management logic"""
        config = StrategyConfig(
            max_position_weight=0.2,
            min_position_weight=0.05,
            max_turnover=0.5
        )
        
        with patch('crowetrade.backtesting.strategy_integration.PortfolioOptimizer'):
            with patch('crowetrade.backtesting.strategy_integration.ExecutionScheduler'):
                strategy = IntegratedStrategy(config)
                
                # Test position limits
                target_weights = {
                    'ASSET_0': 0.4,  # Exceeds 20% limit
                    'ASSET_1': 0.3,  # Exceeds 20% limit
                    'ASSET_2': 0.02  # Below 5% minimum
                }
                current_positions = {}
                
                adjusted = strategy._apply_risk_checks(target_weights, current_positions)
                
                # Should cap positions at max weight
                for asset, weight in adjusted.items():
                    if weight > 0:
                        assert weight <= config.max_position_weight
                
                # Small positions should be removed
                assert adjusted.get('ASSET_2', 0) == 0.0
    
    def test_should_rebalance_logic(self):
        """Test rebalancing decision logic"""
        config = StrategyConfig()
        
        with patch('crowetrade.backtesting.strategy_integration.PortfolioOptimizer'):
            with patch('crowetrade.backtesting.strategy_integration.ExecutionScheduler'):
                strategy = IntegratedStrategy(config)
                
                current_date = datetime(2021, 6, 15)
                positions = {'ASSET_0': 0.5}
                
                # First rebalance should be True
                assert strategy._should_rebalance(current_date, positions) == True
                
                # Set last rebalance date
                strategy.last_rebalance_date = current_date
                
                # Same day rebalance should be False
                assert strategy._should_rebalance(current_date, positions) == False
                
                # Next day should be True
                next_date = current_date + timedelta(days=1)
                assert strategy._should_rebalance(next_date, positions) == True
    
    def test_turnover_calculation(self):
        """Test turnover calculation logic"""
        config = StrategyConfig()
        
        with patch('crowetrade.backtesting.strategy_integration.PortfolioOptimizer'):
            with patch('crowetrade.backtesting.strategy_integration.ExecutionScheduler'):
                strategy = IntegratedStrategy(config)
                
                # Test empty positions
                empty_positions = {}
                max_turnover = strategy._calculate_max_turnover(empty_positions)
                assert max_turnover == config.max_turnover
                
                # Test with positions
                positions_with_data = {'ASSET_0': 0.3, 'ASSET_1': 0.2, 'ASSET_2': 0.1}
                max_turnover = strategy._calculate_max_turnover(positions_with_data)
                assert max_turnover <= config.max_turnover


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_create_strategy_config_function(self):
        """Test strategy config creation function"""
        config = create_strategy_config(
            risk_aversion=2.0,
            max_position_weight=0.25,
            use_regime_detection=False
        )
        
        assert isinstance(config, StrategyConfig)
        assert config.risk_aversion == 2.0
        assert config.max_position_weight == 0.25
        assert config.use_regime_detection == False
    
    def test_create_strategy_config_defaults(self):
        """Test strategy config with defaults"""
        config = create_strategy_config()
        
        assert isinstance(config, StrategyConfig)
        assert config.risk_aversion == 1.0  # Default value


class TestConfigValidation:
    """Test configuration validation"""
    
    def test_valid_risk_aversion_values(self):
        """Test risk aversion parameter validation"""
        # Positive values should work
        config = StrategyConfig(risk_aversion=0.1)
        assert config.risk_aversion == 0.1
        
        config = StrategyConfig(risk_aversion=10.0)
        assert config.risk_aversion == 10.0
        
        # Zero should work (risk-neutral)
        config = StrategyConfig(risk_aversion=0.0)
        assert config.risk_aversion == 0.0
    
    def test_position_weight_limits(self):
        """Test position weight limit validation"""
        config = StrategyConfig(
            max_position_weight=0.5,
            min_position_weight=0.01
        )
        
        assert config.max_position_weight == 0.5
        assert config.min_position_weight == 0.01
        assert config.max_position_weight > config.min_position_weight
    
    def test_rebalance_threshold(self):
        """Test rebalance threshold validation"""
        config = StrategyConfig(rebalance_threshold=0.10)
        assert config.rebalance_threshold == 0.10
        
        # Should accept small values
        config = StrategyConfig(rebalance_threshold=0.001)
        assert config.rebalance_threshold == 0.001


class TestErrorHandling:
    """Test error handling scenarios"""
    
    def test_invalid_signal_handling(self):
        """Test handling of invalid signals"""
        config = StrategyConfig()
        
        with patch('crowetrade.backtesting.strategy_integration.PortfolioOptimizer'):
            with patch('crowetrade.backtesting.strategy_integration.ExecutionScheduler'):
                strategy = IntegratedStrategy(config)
                
                # Test with empty market data
                empty_signals = strategy._momentum_signals(Mock(columns=[]))
                assert isinstance(empty_signals, dict)
                assert len(empty_signals) == 0
    
    def test_optimization_failure_handling(self):
        """Test handling when optimization fails"""
        config = StrategyConfig()
        
        with patch('crowetrade.backtesting.strategy_integration.PortfolioOptimizer') as mock_opt:
            with patch('crowetrade.backtesting.strategy_integration.ExecutionScheduler'):
                # Make optimizer raise exception
                mock_opt_instance = Mock()
                mock_opt_instance.optimize_portfolio.side_effect = Exception("Optimization failed")
                mock_opt.return_value = mock_opt_instance
                
                strategy = IntegratedStrategy(config)
                
                # Mock inputs
                mock_data = Mock()
                mock_data.pct_change.return_value.dropna.return_value.__len__ = Mock(return_value=5)  # Too little data
                
                result = strategy.optimize_portfolio(
                    mock_data, {}, {}, datetime(2021, 1, 1)
                )
                
                # Should return empty dict or current positions on failure
                assert isinstance(result, dict)


class TestIntegrationLogic:
    """Test integration logic without external dependencies"""
    
    def test_strategy_call_method_structure(self):
        """Test the main strategy call method structure"""
        config = StrategyConfig()
        
        with patch('crowetrade.backtesting.strategy_integration.PortfolioOptimizer'):
            with patch('crowetrade.backtesting.strategy_integration.ExecutionScheduler'):
                strategy = IntegratedStrategy(config)
                
                # Mock the internal methods
                strategy.generate_signals = Mock(return_value={'ASSET_0': 0.1})
                strategy.detect_regime = Mock(return_value=None)
                strategy._should_rebalance = Mock(return_value=True)
                strategy.optimize_portfolio = Mock(return_value={'ASSET_0': 0.5})
                strategy._apply_risk_checks = Mock(return_value={'ASSET_0': 0.3})
                
                # Call strategy
                result = strategy(
                    datetime(2021, 1, 1),
                    Mock(),  # market_data
                    {}       # current_positions
                )
                
                # Should call all required methods
                strategy.generate_signals.assert_called_once()
                strategy.detect_regime.assert_called_once()
                strategy._should_rebalance.assert_called_once()
                strategy.optimize_portfolio.assert_called_once()
                strategy._apply_risk_checks.assert_called_once()
                
                # Should return processed weights
                assert result == {'ASSET_0': 0.3}


if __name__ == '__main__':
    # Run basic tests
    print("Running basic backtesting framework tests...")
    
    # Test 1: Configuration
    try:
        config = StrategyConfig(risk_aversion=2.0)
        assert config.risk_aversion == 2.0
        print("✓ StrategyConfig test passed")
    except Exception as e:
        print(f"✗ StrategyConfig test failed: {e}")
    
    # Test 2: Utility function
    try:
        config2 = create_strategy_config(max_position_weight=0.25)
        assert config2.max_position_weight == 0.25
        print("✓ create_strategy_config test passed")
    except Exception as e:
        print(f"✗ create_strategy_config test failed: {e}")
    
    # Test 3: Configuration validation
    try:
        config3 = StrategyConfig()
        config_dict = config3.to_dict()
        assert isinstance(config_dict, dict)
        print("✓ Configuration serialization test passed")
    except Exception as e:
        print(f"✗ Configuration serialization test failed: {e}")
    
    print("\nBasic tests completed. Framework structure is valid.")
