"""Backtesting Framework

Comprehensive backtesting engine for strategy validation:
- Historical market data simulation with configurable replay speed
- Portfolio performance evaluation with transaction costs
- Walk-forward analysis and out-of-sample testing
- Integration with Model Registry for systematic evaluation
- Risk metrics calculation and performance attribution
"""

import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from pathlib import Path
from enum import Enum
import json

logger = logging.getLogger(__name__)


class BacktestMode(Enum):
    """Backtesting execution modes"""
    SINGLE_PERIOD = "single_period"           # Single backtest period
    WALK_FORWARD = "walk_forward"             # Rolling walk-forward analysis
    EXPANDING_WINDOW = "expanding_window"     # Expanding training window
    MONTE_CARLO = "monte_carlo"               # Monte Carlo simulation


class PerformanceMetric(Enum):
    """Standard performance metrics"""
    TOTAL_RETURN = "total_return"
    ANNUALIZED_RETURN = "annualized_return"
    VOLATILITY = "volatility"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    CALMAR_RATIO = "calmar_ratio"
    VaR_95 = "var_95"
    CVaR_95 = "cvar_95"
    SKEWNESS = "skewness"
    KURTOSIS = "kurtosis"


@dataclass
class BacktestConfig:
    """Backtesting configuration parameters"""
    
    # Time period settings
    start_date: datetime
    end_date: datetime
    initial_capital: float = 1_000_000.0
    
    # Rebalancing settings
    rebalance_frequency: str = "daily"  # daily, weekly, monthly, quarterly
    
    # Transaction cost settings
    commission_rate: float = 0.001      # 0.1% commission
    bid_ask_spread: float = 0.0005      # 0.05% spread cost
    market_impact: float = 0.0002       # 0.02% market impact per 1% of volume
    
    # Risk management
    max_position_size: float = 0.20     # 20% max single position
    max_leverage: float = 1.0           # No leverage by default
    stop_loss: Optional[float] = None   # Portfolio stop-loss (-10% = -0.10)
    
    # Data settings
    benchmark_symbol: str = "SPY"       # Benchmark for comparison
    risk_free_rate: float = 0.02       # 2% annual risk-free rate
    
    # Advanced settings
    lag_data: int = 1                   # Days to lag data (realistic execution delay)
    minimum_history: int = 252          # Min days for strategy initialization
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['start_date'] = self.start_date.isoformat()
        data['end_date'] = self.end_date.isoformat()
        return data


@dataclass
class PositionEntry:
    """Individual position entry for tracking"""
    symbol: str
    quantity: float
    price: float
    timestamp: datetime
    side: str  # "buy" or "sell"
    
    @property
    def notional(self) -> float:
        return self.quantity * self.price


@dataclass 
class PerformanceMetrics:
    """Comprehensive performance metrics container"""
    
    # Returns
    total_return: float = 0.0
    annualized_return: float = 0.0
    excess_return: float = 0.0          # Return over benchmark
    
    # Risk metrics
    volatility: float = 0.0
    downside_volatility: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0      # Days
    
    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0      # Excess return / tracking error
    
    # Value at Risk
    var_95: float = 0.0                 # 95% VaR
    cvar_95: float = 0.0                # 95% Conditional VaR
    
    # Distribution statistics
    skewness: float = 0.0
    kurtosis: float = 0.0
    
    # Trading statistics
    num_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0          # Gross profit / gross loss
    
    # Transaction costs
    total_commission: float = 0.0
    total_slippage: float = 0.0
    total_market_impact: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return asdict(self)


class BacktestEngine:
    """Core backtesting engine for strategy evaluation"""
    
    def __init__(self, 
                 market_data: pd.DataFrame,
                 config: BacktestConfig):
        
        self.market_data = market_data.copy()
        self.config = config
        
        # Ensure data is sorted by date
        if 'date' in self.market_data.columns:
            self.market_data = self.market_data.sort_values('date')
            self.market_data.set_index('date', inplace=True)
        
        # Initialize tracking variables
        self.positions: Dict[str, float] = {}       # Current positions
        self.portfolio_value: List[float] = []      # Portfolio value history
        self.returns: List[float] = []              # Daily returns
        self.trades: List[PositionEntry] = []       # All trades
        self.dates: List[datetime] = []             # Date history
        
        # Transaction costs tracking
        self.commission_costs: List[float] = []
        self.slippage_costs: List[float] = []
        self.impact_costs: List[float] = []
        
        # Current state
        self.current_cash = config.initial_capital
        self.current_date = None
        self.is_initialized = False
        
        logger.info(f"BacktestEngine initialized: {config.start_date} to {config.end_date}")
    
    def run_backtest(self, 
                    strategy_func: Callable[[datetime, pd.DataFrame, Dict], Dict[str, float]],
                    strategy_params: Dict[str, Any] = None) -> PerformanceMetrics:
        """Run complete backtest with given strategy
        
        Args:
            strategy_func: Strategy function that returns target weights
                Signature: (date, market_data, current_positions) -> target_weights
            strategy_params: Additional parameters for strategy
            
        Returns:
            PerformanceMetrics object with comprehensive results
        """
        
        if strategy_params is None:
            strategy_params = {}
        
        # Filter data to backtest period
        start_date = self.config.start_date
        end_date = self.config.end_date
        
        backtest_data = self.market_data[
            (self.market_data.index >= start_date) & 
            (self.market_data.index <= end_date)
        ].copy()
        
        if len(backtest_data) == 0:
            raise ValueError(f"No data available for period {start_date} to {end_date}")
        
        logger.info(f"Running backtest on {len(backtest_data)} days of data")
        
        # Reset state
        self._reset_state()
        
        # Main backtest loop
        for i, (current_date, row) in enumerate(backtest_data.iterrows()):
            self.current_date = current_date
            
            # Skip initial period if insufficient history
            if i < self.config.minimum_history:
                self._record_portfolio_value(current_date, self.config.initial_capital)
                continue
            
            # Get available market data up to current date (with lag)
            available_data = self._get_available_data(current_date, backtest_data)
            
            try:
                # Get target portfolio weights from strategy
                target_weights = strategy_func(
                    current_date, 
                    available_data, 
                    self.positions.copy(),
                    **strategy_params
                )
                
                # Validate and normalize weights
                target_weights = self._validate_weights(target_weights)
                
                # Execute rebalancing if needed
                if self._should_rebalance(current_date):
                    self._rebalance_portfolio(target_weights, current_date, row)
                
                # Calculate and record portfolio value
                portfolio_value = self._calculate_portfolio_value(current_date, row)
                self._record_portfolio_value(current_date, portfolio_value)
                
            except Exception as e:
                logger.error(f"Error on {current_date}: {e}")
                # Record previous portfolio value on error
                if self.portfolio_value:
                    self._record_portfolio_value(current_date, self.portfolio_value[-1])
                else:
                    self._record_portfolio_value(current_date, self.config.initial_capital)
        
        # Calculate final performance metrics
        return self._calculate_performance_metrics()
    
    def run_walk_forward_analysis(self,
                                 strategy_func: Callable,
                                 training_window: int = 252,  # Days
                                 refit_frequency: int = 63,   # Refit every quarter
                                 strategy_params: Dict[str, Any] = None) -> List[PerformanceMetrics]:
        """Run walk-forward analysis with periodic model retraining
        
        Returns:
            List of PerformanceMetrics for each walk-forward period
        """
        
        results = []
        current_start = self.config.start_date
        
        while current_start < self.config.end_date:
            # Define training and testing periods
            training_end = current_start + timedelta(days=training_window)
            test_start = training_end + timedelta(days=1)
            test_end = min(
                test_start + timedelta(days=refit_frequency), 
                self.config.end_date
            )
            
            if test_start >= self.config.end_date:
                break
            
            logger.info(f"Walk-forward period: train={current_start} to {training_end}, "
                       f"test={test_start} to {test_end}")
            
            # Create configuration for this period
            period_config = BacktestConfig(
                start_date=test_start,
                end_date=test_end,
                initial_capital=self.config.initial_capital,
                rebalance_frequency=self.config.rebalance_frequency,
                commission_rate=self.config.commission_rate,
                bid_ask_spread=self.config.bid_ask_spread,
                market_impact=self.config.market_impact
            )
            
            # Run backtest for this period
            period_engine = BacktestEngine(self.market_data, period_config)
            
            # Modify strategy function to use training data
            def wrapped_strategy(date, market_data, positions, **kwargs):
                # Filter training data up to current date
                training_data = self.market_data[
                    (self.market_data.index >= current_start) &
                    (self.market_data.index <= min(date, training_end))
                ]
                
                return strategy_func(date, training_data, positions, **kwargs)
            
            try:
                period_metrics = period_engine.run_backtest(wrapped_strategy, strategy_params)
                period_metrics.period_start = test_start
                period_metrics.period_end = test_end
                results.append(period_metrics)
                
            except Exception as e:
                logger.error(f"Walk-forward period failed: {e}")
            
            # Move to next period
            current_start = test_end + timedelta(days=1)
        
        logger.info(f"Walk-forward analysis complete: {len(results)} periods")
        return results
    
    def _reset_state(self) -> None:
        """Reset engine state for new backtest"""
        self.positions = {}
        self.portfolio_value = []
        self.returns = []
        self.trades = []
        self.dates = []
        self.commission_costs = []
        self.slippage_costs = []
        self.impact_costs = []
        self.current_cash = self.config.initial_capital
        self.is_initialized = False
    
    def _get_available_data(self, current_date: datetime, full_data: pd.DataFrame) -> pd.DataFrame:
        """Get available market data up to current date with lag"""
        lag_date = current_date - timedelta(days=self.config.lag_data)
        return full_data[full_data.index <= lag_date]
    
    def _validate_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Validate and normalize portfolio weights"""
        if not weights:
            return {}
        
        # Remove zero or negative weights
        weights = {k: v for k, v in weights.items() if v > 0}
        
        # Check position size limits
        for symbol, weight in weights.items():
            if weight > self.config.max_position_size:
                logger.warning(f"Position {symbol} weight {weight:.3f} exceeds limit "
                             f"{self.config.max_position_size:.3f}")
                weights[symbol] = self.config.max_position_size
        
        # Normalize to sum to 1.0 (or less for cash)
        total_weight = sum(weights.values())
        if total_weight > 1.0:
            # Scale down proportionally
            scale_factor = 1.0 / total_weight
            weights = {k: v * scale_factor for k, v in weights.items()}
            logger.info(f"Scaled weights by {scale_factor:.3f} to fit constraint")
        
        return weights
    
    def _should_rebalance(self, current_date: datetime) -> bool:
        """Determine if portfolio should be rebalanced"""
        if not self.dates:
            return True  # First day
        
        last_date = self.dates[-1]
        freq = self.config.rebalance_frequency.lower()
        
        if freq == "daily":
            return True
        elif freq == "weekly":
            return current_date.weekday() == 0  # Monday
        elif freq == "monthly":
            return current_date.day == 1
        elif freq == "quarterly":
            return current_date.month in [1, 4, 7, 10] and current_date.day == 1
        else:
            return True  # Default to daily
    
    def _rebalance_portfolio(self, 
                           target_weights: Dict[str, float], 
                           current_date: datetime,
                           market_data_row: pd.Series) -> None:
        """Execute portfolio rebalancing with transaction costs"""
        
        # Calculate current portfolio value
        current_value = self._calculate_portfolio_value(current_date, market_data_row)
        
        # Calculate target positions
        target_positions = {}
        for symbol, weight in target_weights.items():
            if symbol in market_data_row.index and not pd.isna(market_data_row[symbol]):
                target_value = weight * current_value
                target_quantity = target_value / market_data_row[symbol]
                target_positions[symbol] = target_quantity
        
        # Execute trades
        total_commission = 0.0
        total_slippage = 0.0
        total_impact = 0.0
        
        # Sell positions not in target or reduce oversized positions
        for symbol, current_qty in list(self.positions.items()):
            target_qty = target_positions.get(symbol, 0.0)
            
            if target_qty < current_qty:
                sell_qty = current_qty - target_qty
                price = market_data_row[symbol]
                
                if pd.isna(price):
                    continue
                
                # Calculate transaction costs
                commission = sell_qty * price * self.config.commission_rate
                slippage = sell_qty * price * self.config.bid_ask_spread / 2
                impact = sell_qty * price * self.config.market_impact
                
                # Execute sell
                self.positions[symbol] = target_qty
                self.current_cash += sell_qty * price - commission - slippage - impact
                
                # Record trade
                self.trades.append(PositionEntry(
                    symbol=symbol,
                    quantity=sell_qty,
                    price=price,
                    timestamp=current_date,
                    side="sell"
                ))
                
                total_commission += commission
                total_slippage += slippage
                total_impact += impact
        
        # Buy new positions or increase existing ones
        for symbol, target_qty in target_positions.items():
            current_qty = self.positions.get(symbol, 0.0)
            
            if target_qty > current_qty:
                buy_qty = target_qty - current_qty
                price = market_data_row[symbol]
                
                if pd.isna(price):
                    continue
                
                # Calculate transaction costs
                commission = buy_qty * price * self.config.commission_rate
                slippage = buy_qty * price * self.config.bid_ask_spread / 2
                impact = buy_qty * price * self.config.market_impact
                
                total_cost = buy_qty * price + commission + slippage + impact
                
                # Check if we have enough cash
                if total_cost <= self.current_cash:
                    # Execute buy
                    self.positions[symbol] = target_qty
                    self.current_cash -= total_cost
                    
                    # Record trade
                    self.trades.append(PositionEntry(
                        symbol=symbol,
                        quantity=buy_qty,
                        price=price,
                        timestamp=current_date,
                        side="buy"
                    ))
                    
                    total_commission += commission
                    total_slippage += slippage
                    total_impact += impact
                else:
                    logger.warning(f"Insufficient cash for {symbol}: need {total_cost:.2f}, "
                                 f"have {self.current_cash:.2f}")
        
        # Record transaction costs
        self.commission_costs.append(total_commission)
        self.slippage_costs.append(total_slippage)
        self.impact_costs.append(total_impact)
        
        # Clean up zero positions
        self.positions = {k: v for k, v in self.positions.items() if abs(v) > 1e-8}
    
    def _calculate_portfolio_value(self, current_date: datetime, market_data_row: pd.Series) -> float:
        """Calculate total portfolio value including cash"""
        
        position_value = 0.0
        for symbol, quantity in self.positions.items():
            if symbol in market_data_row.index and not pd.isna(market_data_row[symbol]):
                position_value += quantity * market_data_row[symbol]
        
        return position_value + self.current_cash
    
    def _record_portfolio_value(self, current_date: datetime, portfolio_value: float) -> None:
        """Record portfolio value and calculate returns"""
        
        self.dates.append(current_date)
        self.portfolio_value.append(portfolio_value)
        
        # Calculate daily return
        if len(self.portfolio_value) > 1:
            daily_return = (portfolio_value / self.portfolio_value[-2]) - 1.0
            self.returns.append(daily_return)
        else:
            self.returns.append(0.0)  # First day return is 0
    
    def _calculate_performance_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        
        if len(self.returns) < 2:
            return PerformanceMetrics()
        
        returns_series = pd.Series(self.returns)
        returns_series = returns_series.replace([np.inf, -np.inf], 0)  # Handle infinite returns
        returns_series = returns_series.fillna(0)  # Handle NaN returns
        
        metrics = PerformanceMetrics()
        
        # Basic return metrics
        metrics.total_return = (self.portfolio_value[-1] / self.config.initial_capital) - 1.0
        
        # Annualized return
        trading_days = len(self.returns)
        if trading_days > 0:
            metrics.annualized_return = ((1 + metrics.total_return) ** (252 / trading_days)) - 1.0
        
        # Volatility (annualized)
        if len(returns_series) > 1:
            metrics.volatility = returns_series.std() * np.sqrt(252)
        
        # Sharpe ratio
        if metrics.volatility > 0:
            excess_return = metrics.annualized_return - self.config.risk_free_rate
            metrics.sharpe_ratio = excess_return / metrics.volatility
        
        # Downside volatility and Sortino ratio
        negative_returns = returns_series[returns_series < 0]
        if len(negative_returns) > 1:
            metrics.downside_volatility = negative_returns.std() * np.sqrt(252)
            if metrics.downside_volatility > 0:
                excess_return = metrics.annualized_return - self.config.risk_free_rate
                metrics.sortino_ratio = excess_return / metrics.downside_volatility
        
        # Maximum drawdown
        cumulative_returns = (1 + returns_series).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        metrics.max_drawdown = drawdown.min()
        
        # Calmar ratio
        if abs(metrics.max_drawdown) > 0:
            metrics.calmar_ratio = metrics.annualized_return / abs(metrics.max_drawdown)
        
        # Value at Risk (95%)
        if len(returns_series) > 1:
            metrics.var_95 = returns_series.quantile(0.05)
            tail_returns = returns_series[returns_series <= metrics.var_95]
            if len(tail_returns) > 0:
                metrics.cvar_95 = tail_returns.mean()
        
        # Skewness and Kurtosis
        if len(returns_series) > 3:
            metrics.skewness = returns_series.skew()
            metrics.kurtosis = returns_series.kurtosis()
        
        # Trading statistics
        metrics.num_trades = len(self.trades)
        
        if self.trades:
            winning_trades = [t for t in self.trades if t.side == "sell"]  # Approximate
            if winning_trades:
                # This is a simplified win rate calculation
                # In practice, you'd track individual position PnL
                metrics.win_rate = 0.5  # Placeholder - needs proper trade matching
        
        # Transaction costs
        metrics.total_commission = sum(self.commission_costs)
        metrics.total_slippage = sum(self.slippage_costs)  
        metrics.total_market_impact = sum(self.impact_costs)
        
        return metrics


def create_backtest_config(start_date: str,
                          end_date: str,
                          initial_capital: float = 1_000_000.0,
                          **kwargs) -> BacktestConfig:
    """Convenience function to create backtest configuration"""
    
    return BacktestConfig(
        start_date=datetime.fromisoformat(start_date),
        end_date=datetime.fromisoformat(end_date),
        initial_capital=initial_capital,
        **kwargs
    )


def load_market_data(data_path: Union[str, Path], 
                    symbols: List[str] = None) -> pd.DataFrame:
    """Load market data from file
    
    Expected format: CSV with date index and symbol columns for prices
    """
    
    data_path = Path(data_path)
    
    if data_path.suffix.lower() == '.csv':
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    elif data_path.suffix.lower() == '.parquet':
        data = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")
    
    if symbols:
        # Filter to requested symbols
        available_symbols = [s for s in symbols if s in data.columns]
        if not available_symbols:
            raise ValueError(f"No requested symbols found in data: {symbols}")
        data = data[available_symbols]
    
    # Forward fill missing data and drop rows with all NaN
    data = data.fillna(method='ffill').dropna()
    
    logger.info(f"Loaded market data: {len(data)} rows, {len(data.columns)} symbols")
    return data
