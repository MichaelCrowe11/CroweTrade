# Backtesting Framework Documentation

## Overview

The CroweTrade Backtesting Framework provides comprehensive historical simulation capabilities for systematic trading strategy validation. It integrates with all major CroweTrade components including the Model Registry, A/B Testing system, Portfolio Optimizer, and Regime Detector.

## Architecture

### Core Components

1. **BacktestEngine** - Main simulation engine
2. **IntegratedStrategy** - Unified strategy interface connecting all components
3. **StrategyBacktester** - High-level backtesting orchestrator
4. **PerformanceMetrics** - Comprehensive performance evaluation

### Key Features

- **Historical Simulation**: Event-driven backtesting with realistic market conditions
- **Transaction Costs**: Configurable commission, bid-ask spreads, and market impact
- **Walk-Forward Analysis**: Time-series cross-validation for robust validation
- **Parameter Optimization**: Grid search and optimization across strategy parameters
- **Model Integration**: Seamless integration with Model Registry for systematic testing
- **Risk Management**: Built-in position limits, turnover constraints, and drawdown controls

## Quick Start

### Basic Backtest

```python
from datetime import datetime
from crowetrade.backtesting.strategy_integration import run_simple_backtest
import pandas as pd

# Load market data
market_data = pd.read_csv('market_data.csv', index_col='date', parse_dates=True)

# Run backtest
metrics = run_simple_backtest(
    market_data=market_data,
    start_date='2021-01-01',
    end_date='2021-12-31',
    strategy_params={
        'risk_aversion': 1.0,
        'max_position_weight': 0.25
    }
)

print(f"Total Return: {metrics.total_return:.3f}")
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
print(f"Max Drawdown: {metrics.max_drawdown:.3f}")
```

### Advanced Strategy Configuration

```python
from crowetrade.backtesting.strategy_integration import (
    StrategyConfig, StrategyBacktester
)
from crowetrade.backtesting.engine import BacktestConfig, TransactionCosts

# Configure strategy
strategy_config = StrategyConfig(
    # Portfolio optimization
    optimization_method=OptimizationMethod.MEAN_VARIANCE,
    risk_aversion=1.5,
    
    # Risk management
    max_position_weight=0.20,
    min_position_weight=0.05,
    max_turnover=0.50,
    
    # Regime detection
    use_regime_detection=True,
    regime_lookback=252,
    
    # Model integration
    signal_model_id='momentum_v1.2',
    regime_model_id='hmm_regime_v2.1'
)

# Configure backtest
backtest_config = BacktestConfig(
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2023, 12, 31),
    initial_capital=10_000_000.0,
    transaction_costs=TransactionCosts(
        commission_rate=0.0005,  # 5 bps commission
        bid_ask_spread=0.0010,   # 10 bps bid-ask spread
        market_impact=0.0001     # 1 bp market impact
    )
)

# Run backtest
backtester = StrategyBacktester(model_registry=model_registry)
metrics = backtester.backtest_strategy(strategy_config, backtest_config, market_data)
```

## Strategy Integration

### IntegratedStrategy Class

The `IntegratedStrategy` class unifies all CroweTrade components:

```python
from crowetrade.backtesting.strategy_integration import IntegratedStrategy, StrategyConfig

# Create strategy
config = StrategyConfig(
    optimization_method=OptimizationMethod.RISK_PARITY,
    risk_aversion=2.0,
    use_regime_detection=True
)

strategy = IntegratedStrategy(config, model_registry)

# Use as callable function
target_weights = strategy(
    current_date=datetime(2021, 6, 15),
    market_data=market_data[:datetime(2021, 6, 15)],
    current_positions={'AAPL': 0.2, 'TSLA': 0.15}
)
```

### Strategy Workflow

1. **Signal Generation**: Extract trading signals from market data or ML models
2. **Regime Detection**: Identify current market regime (optional)
3. **Portfolio Optimization**: Determine optimal allocation using various methods
4. **Risk Checks**: Apply position limits, turnover constraints, and other risk controls
5. **Execution Planning**: Generate execution plans with cost estimates

### Available Optimization Methods

- `MEAN_VARIANCE`: Classic mean-variance optimization
- `RISK_PARITY`: Risk parity allocation
- `EQUAL_WEIGHT`: Simple equal-weight allocation
- `MINIMUM_VARIANCE`: Minimum variance portfolio
- `MAXIMUM_DIVERSIFICATION`: Maximum diversification strategy

## Performance Metrics

The framework calculates comprehensive performance statistics:

### Returns Metrics
- **Total Return**: Cumulative return over backtest period
- **Annualized Return**: Compound annual growth rate
- **Excess Return**: Return above risk-free rate

### Risk Metrics
- **Volatility**: Annualized standard deviation of returns
- **Sharpe Ratio**: Risk-adjusted return measure
- **Sortino Ratio**: Downside risk-adjusted return
- **Calmar Ratio**: Return to maximum drawdown ratio

### Drawdown Metrics
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Average Drawdown**: Mean drawdown during losing periods
- **Drawdown Duration**: Time to recover from maximum drawdown

### Risk Measures
- **Value at Risk (VaR)**: 95th percentile loss
- **Conditional VaR (CVaR)**: Expected loss beyond VaR
- **Skewness**: Return distribution asymmetry
- **Kurtosis**: Return distribution tail thickness

### Example Performance Report

```python
# Print comprehensive performance report
def print_performance_report(metrics):
    print("=== PERFORMANCE REPORT ===")
    print(f"Total Return:        {metrics.total_return:.2%}")
    print(f"Annualized Return:   {metrics.annualized_return:.2%}")
    print(f"Volatility:          {metrics.volatility:.2%}")
    print(f"Sharpe Ratio:        {metrics.sharpe_ratio:.3f}")
    print(f"Sortino Ratio:       {metrics.sortino_ratio:.3f}")
    print(f"Calmar Ratio:        {metrics.calmar_ratio:.3f}")
    print(f"Max Drawdown:        {metrics.max_drawdown:.2%}")
    print(f"VaR (95%):          {metrics.var_95:.2%}")
    print(f"CVaR (95%):         {metrics.cvar_95:.2%}")
    print(f"Skewness:           {metrics.skewness:.3f}")
    print(f"Kurtosis:           {metrics.kurtosis:.3f}")

print_performance_report(metrics)
```

## Walk-Forward Analysis

Perform robust out-of-sample validation using walk-forward analysis:

```python
# Run walk-forward analysis
results = backtester.walk_forward_analysis(
    strategy_config=strategy_config,
    market_data=market_data,
    training_window=252,  # 1 year training
    test_window=63       # 3 months testing
)

# Analyze results across periods
total_returns = [r.total_return for r in results]
sharpe_ratios = [r.sharpe_ratio for r in results]

print(f"Average Return: {np.mean(total_returns):.3f}")
print(f"Return Std: {np.std(total_returns):.3f}")
print(f"Average Sharpe: {np.mean(sharpe_ratios):.3f}")
print(f"Sharpe Consistency: {np.std(sharpe_ratios):.3f}")
```

## Parameter Optimization

Systematically optimize strategy parameters:

```python
# Define parameter grid
param_grid = {
    'risk_aversion': [0.5, 1.0, 1.5, 2.0, 3.0],
    'max_position_weight': [0.15, 0.20, 0.25, 0.30],
    'rebalance_threshold': [0.02, 0.05, 0.10],
    'use_regime_detection': [True, False]
}

# Run parameter sweep
results = backtester.parameter_sweep(
    base_config=base_config,
    backtest_config=backtest_config,
    market_data=market_data,
    param_grid=param_grid
)

# Analyze best parameters
best_params, best_metrics = results[0]  # Results sorted by Sharpe ratio
print("Best Parameters:", best_params)
print(f"Best Sharpe: {best_metrics.sharpe_ratio:.3f}")
print(f"Best Return: {best_metrics.total_return:.3f}")
```

## Model Registry Integration

### Using Models from Registry

```python
from crowetrade.models.registry import ModelRegistry

# Initialize model registry
model_registry = ModelRegistry('/path/to/models')

# Register models
model_registry.register_model(
    name='momentum_signals',
    model_object=momentum_model,
    metadata={'version': '1.2', 'type': 'signal_generator'}
)

# Use in strategy
strategy_config = StrategyConfig(
    signal_model_id='momentum_signals'
)

strategy = IntegratedStrategy(strategy_config, model_registry)
```

### Systematic Model Testing

```python
# Test all signal models
signal_models = model_registry.list_models()['signal_generator']

results = {}
for model_id in signal_models:
    config = StrategyConfig(signal_model_id=model_id)
    metrics = backtester.backtest_strategy(config, backtest_config, market_data)
    results[model_id] = metrics

# Compare model performance
best_model = max(results.keys(), key=lambda m: results[m].sharpe_ratio)
print(f"Best signal model: {best_model}")
print(f"Sharpe ratio: {results[best_model].sharpe_ratio:.3f}")
```

## Transaction Costs

### Cost Configuration

```python
from crowetrade.backtesting.engine import TransactionCosts

# Conservative cost model
conservative_costs = TransactionCosts(
    commission_rate=0.001,   # 10 bps commission
    bid_ask_spread=0.002,    # 20 bps spread
    market_impact=0.0005     # 5 bps impact
)

# Institutional cost model
institutional_costs = TransactionCosts(
    commission_rate=0.0002,  # 2 bps commission
    bid_ask_spread=0.0008,   # 8 bps spread
    market_impact=0.0001     # 1 bp impact
)
```

### Cost Impact Analysis

```python
# Compare strategies with different cost assumptions
configs = [
    BacktestConfig(transaction_costs=TransactionCosts(0, 0, 0)),  # No costs
    BacktestConfig(transaction_costs=conservative_costs),          # High costs
    BacktestConfig(transaction_costs=institutional_costs)         # Low costs
]

for i, config in enumerate(configs):
    metrics = backtester.backtest_strategy(strategy_config, config, market_data)
    print(f"Cost Model {i}: Return={metrics.total_return:.3f}, "
          f"Sharpe={metrics.sharpe_ratio:.3f}")
```

## Best Practices

### Strategy Development

1. **Start Simple**: Begin with basic equal-weight or momentum strategies
2. **Incremental Complexity**: Add components (regime detection, optimization) gradually
3. **Out-of-Sample Testing**: Always validate on unseen data
4. **Multiple Timeframes**: Test across different market conditions
5. **Transaction Costs**: Include realistic cost assumptions

### Performance Evaluation

1. **Risk-Adjusted Returns**: Focus on Sharpe ratio, not just returns
2. **Drawdown Analysis**: Monitor maximum drawdown and recovery time
3. **Regime Performance**: Analyze performance across different market regimes
4. **Parameter Stability**: Ensure results are stable across parameter ranges
5. **Walk-Forward Validation**: Use rolling window validation for robustness

### Model Integration

1. **Version Control**: Use Model Registry for systematic model management
2. **A/B Testing**: Compare models using statistical significance testing
3. **Ensemble Methods**: Combine multiple models for improved performance
4. **Regular Retraining**: Update models periodically with new data
5. **Graceful Degradation**: Handle model failures and missing predictions

### Risk Management

1. **Position Limits**: Set maximum individual position sizes
2. **Turnover Constraints**: Limit portfolio turnover to control costs
3. **Diversification**: Maintain appropriate diversification across assets
4. **Stress Testing**: Test strategies under extreme market conditions
5. **Real-Time Monitoring**: Implement live performance monitoring

## Advanced Topics

### Custom Strategy Implementation

```python
class CustomMomentumStrategy(IntegratedStrategy):
    """Custom momentum strategy with volatility filtering"""
    
    def generate_signals(self, market_data, current_date):
        # Custom signal logic
        returns = market_data.pct_change(20)  # 20-day returns
        volatility = returns.rolling(60).std()  # 60-day volatility
        
        # Generate momentum signals but filter by volatility
        signals = {}
        for asset in market_data.columns:
            momentum = returns[asset].iloc[-1]
            vol = volatility[asset].iloc[-1]
            
            # Reduce signal strength in high volatility periods
            vol_factor = min(1.0, 0.02 / vol) if vol > 0 else 1.0
            signals[asset] = np.tanh(momentum * 10) * vol_factor
            
        return signals
```

### Multi-Asset Class Strategies

```python
# Configure strategy for multiple asset classes
strategy_config = StrategyConfig(
    max_position_weight=0.15,  # Lower individual limits
    min_position_weight=0.02,   # Require meaningful positions
    use_regime_detection=True,  # Important for multi-asset
    optimization_method=OptimizationMethod.RISK_PARITY  # Good for diversification
)

# Different constraints by asset class
def asset_class_constraints(asset):
    if asset.startswith('EQUITY_'):
        return {'max_weight': 0.20, 'min_weight': 0.02}
    elif asset.startswith('BOND_'):
        return {'max_weight': 0.30, 'min_weight': 0.05}
    elif asset.startswith('COMMODITY_'):
        return {'max_weight': 0.10, 'min_weight': 0.01}
    else:
        return {'max_weight': 0.15, 'min_weight': 0.02}
```

### Performance Attribution

```python
def calculate_attribution(portfolio_returns, benchmark_returns, weights):
    """Calculate performance attribution"""
    
    # Asset contribution to portfolio return
    asset_contributions = {}
    for date in portfolio_returns.index:
        date_weights = weights.loc[date] if date in weights.index else {}
        date_returns = portfolio_returns.loc[date]
        
        for asset in date_weights.keys():
            if asset not in asset_contributions:
                asset_contributions[asset] = 0
            asset_contributions[asset] += date_weights[asset] * date_returns
    
    return asset_contributions
```

This comprehensive backtesting framework provides the foundation for systematic strategy development, validation, and optimization in the CroweTrade platform.
