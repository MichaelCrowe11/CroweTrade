"""Strategy Integration Module

Connects CroweTrade components with the backtesting framework:
- Portfolio optimization strategies with regime detection
- Model registry integration for systematic testing
- Execution cost modeling and performance attribution  
- Strategy parameter optimization and validation
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

from .engine import BacktestEngine, BacktestConfig, PerformanceMetrics
from ..portfolio.optimizer import PortfolioOptimizer, OptimizationMethod
from ..regime.detector import RegimeDetector, RegimeState
from ..models.registry import ModelRegistry
from ..execution.scheduler import ExecutionScheduler, ExecutionPlan

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Configuration for integrated trading strategy"""
    
    # Portfolio optimization settings
    optimization_method: OptimizationMethod = OptimizationMethod.MEAN_VARIANCE
    risk_aversion: float = 1.0
    rebalance_threshold: float = 0.05
    
    # Regime detection settings
    use_regime_detection: bool = True
    regime_lookback: int = 252
    regime_min_duration: int = 5
    
    # Risk management
    max_position_weight: float = 0.30
    min_position_weight: float = 0.05
    max_turnover: float = 1.0
    
    # Model selection
    signal_model_id: Optional[str] = None
    regime_model_id: Optional[str] = None
    portfolio_model_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, Enum):
                data[field_name] = field_value.value
            else:
                data[field_name] = field_value
        return data


class IntegratedStrategy:
    """Integrated trading strategy combining all CroweTrade components"""
    
    def __init__(self,
                 config: StrategyConfig,
                 model_registry: Optional[ModelRegistry] = None):
        
        self.config = config
        self.model_registry = model_registry
        
        # Initialize components
        self.portfolio_optimizer = PortfolioOptimizer()
        
        if config.use_regime_detection:
            self.regime_detector = RegimeDetector(
                lookback_window=config.regime_lookback,
                min_regime_duration=config.regime_min_duration
            )
        else:
            self.regime_detector = None
        
        self.execution_scheduler = ExecutionScheduler()
        
        # Load models from registry if specified
        self.signal_model = None
        self.regime_model = None
        self.portfolio_model = None
        
        if model_registry:
            self._load_models()
        
        # Strategy state
        self.current_regime = None
        self.last_rebalance_date = None
        
        logger.info(f"IntegratedStrategy initialized with {config.optimization_method}")
    
    def generate_signals(self, 
                        market_data: pd.DataFrame,
                        current_date: datetime) -> Dict[str, float]:
        """Generate trading signals for assets
        
        Args:
            market_data: Historical market data up to current_date
            current_date: Current simulation date
            
        Returns:
            Dictionary of asset signals (-1 to 1, where 0 is neutral)
        """
        
        if self.signal_model and len(market_data) >= 30:
            try:
                # Prepare features for signal model
                features = self._prepare_signal_features(market_data, current_date)
                
                # Generate signals using loaded model
                signals = self.signal_model.predict(features)
                
                # Convert to dictionary format
                if len(signals) == len(market_data.columns):
                    return dict(zip(market_data.columns, signals))
                    
            except Exception as e:
                logger.warning(f"Signal model prediction failed: {e}")
        
        # Fallback: momentum-based signals
        return self._momentum_signals(market_data)
    
    def detect_regime(self, 
                     market_data: pd.DataFrame,
                     current_date: datetime) -> Optional[RegimeState]:
        """Detect current market regime
        
        Args:
            market_data: Historical market data
            current_date: Current simulation date
            
        Returns:
            Current regime state or None if insufficient data
        """
        
        if not self.regime_detector or len(market_data) < self.config.regime_lookback:
            return None
        
        try:
            # Use regime detector
            regime_probs = self.regime_detector.detect_regime(market_data.values)
            
            # Update current regime
            self.current_regime = regime_probs
            return regime_probs
            
        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            return None
    
    def optimize_portfolio(self,
                          market_data: pd.DataFrame,
                          signals: Dict[str, float],
                          current_positions: Dict[str, float],
                          current_date: datetime) -> Dict[str, float]:
        """Optimize portfolio allocation
        
        Args:
            market_data: Historical price data
            signals: Asset signals from signal generation
            current_positions: Current portfolio positions
            current_date: Current simulation date
            
        Returns:
            Dictionary of target portfolio weights
        """
        
        if len(market_data) < 30:  # Need minimum history
            return {}
        
        try:
            # Prepare optimization inputs
            returns_data = market_data.pct_change().dropna()
            
            if len(returns_data) < 10:
                return {}
            
            # Calculate expected returns incorporating signals
            expected_returns = self._calculate_expected_returns(returns_data, signals)
            
            # Adjust parameters based on regime if available
            optimization_params = self._get_regime_adjusted_params()
            
            # Create constraints
            constraints = {
                'min_weight': self.config.min_position_weight,
                'max_weight': self.config.max_position_weight,
                'max_turnover': self._calculate_max_turnover(current_positions)
            }
            
            # Run optimization
            optimal_weights = self.portfolio_optimizer.optimize_portfolio(
                expected_returns=expected_returns,
                returns_data=returns_data,
                method=self.config.optimization_method,
                risk_aversion=optimization_params['risk_aversion'],
                constraints=constraints
            )
            
            return optimal_weights
            
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            return current_positions  # Return current positions as fallback
    
    def __call__(self, 
                current_date: datetime,
                market_data: pd.DataFrame, 
                current_positions: Dict[str, float],
                **kwargs) -> Dict[str, float]:
        """Main strategy function compatible with BacktestEngine
        
        This is the function called by the backtesting engine
        """
        
        try:
            # Step 1: Generate trading signals
            signals = self.generate_signals(market_data, current_date)
            
            # Step 2: Detect market regime
            regime = self.detect_regime(market_data, current_date)
            
            # Step 3: Check if we should rebalance
            if not self._should_rebalance(current_date, current_positions):
                return current_positions
            
            # Step 4: Optimize portfolio
            target_weights = self.optimize_portfolio(
                market_data, signals, current_positions, current_date
            )
            
            # Step 5: Apply risk checks
            target_weights = self._apply_risk_checks(target_weights, current_positions)
            
            # Update last rebalance date
            self.last_rebalance_date = current_date
            
            return target_weights
            
        except Exception as e:
            logger.error(f"Strategy execution failed on {current_date}: {e}")
            return current_positions  # Safe fallback
    
    def _load_models(self) -> None:
        """Load models from registry"""
        
        if not self.model_registry:
            return
        
        try:
            if self.config.signal_model_id:
                self.signal_model, _ = self.model_registry.get_model(
                    self.config.signal_model_id
                )
                logger.info(f"Loaded signal model: {self.config.signal_model_id}")
                
            if self.config.regime_model_id:
                self.regime_model, _ = self.model_registry.get_model(
                    self.config.regime_model_id
                )
                logger.info(f"Loaded regime model: {self.config.regime_model_id}")
                
            if self.config.portfolio_model_id:
                self.portfolio_model, _ = self.model_registry.get_model(
                    self.config.portfolio_model_id
                )
                logger.info(f"Loaded portfolio model: {self.config.portfolio_model_id}")
                
        except Exception as e:
            logger.error(f"Failed to load models from registry: {e}")
    
    def _prepare_signal_features(self, 
                                market_data: pd.DataFrame,
                                current_date: datetime) -> np.ndarray:
        """Prepare features for signal model prediction"""
        
        # Calculate basic technical features
        returns = market_data.pct_change().fillna(0)
        
        # Rolling statistics (last 20 days)
        features = []
        for col in market_data.columns:
            if col in returns.columns:
                recent_returns = returns[col].tail(20)
                
                # Momentum features
                features.extend([
                    recent_returns.mean(),           # Average return
                    recent_returns.std(),            # Volatility
                    recent_returns.sum(),           # Cumulative return
                    recent_returns.skew() if len(recent_returns) > 3 else 0,  # Skewness
                ])
                
                # Price-based features
                recent_prices = market_data[col].tail(20)
                if len(recent_prices) >= 2:
                    features.extend([
                        (recent_prices.iloc[-1] / recent_prices.iloc[0]) - 1,  # Total period return
                        (recent_prices.iloc[-1] / recent_prices.mean()) - 1,   # Price vs mean
                    ])
                else:
                    features.extend([0, 0])
        
        return np.array(features).reshape(1, -1)  # Single sample for prediction
    
    def _momentum_signals(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Generate momentum-based signals as fallback"""
        
        signals = {}
        
        for asset in market_data.columns:
            if len(market_data[asset]) >= 20:
                recent_prices = market_data[asset].tail(20)
                
                # Simple momentum: 20-day return
                momentum_return = (recent_prices.iloc[-1] / recent_prices.iloc[0]) - 1
                
                # Normalize to [-1, 1] range
                signal = np.tanh(momentum_return * 10)  # Scale factor of 10
                signals[asset] = float(signal)
            else:
                signals[asset] = 0.0
        
        return signals
    
    def _calculate_expected_returns(self,
                                  returns_data: pd.DataFrame,
                                  signals: Dict[str, float]) -> pd.Series:
        """Calculate expected returns incorporating signals"""
        
        # Base expected returns (historical mean)
        base_returns = returns_data.mean() * 252  # Annualize
        
        # Adjust based on signals
        signal_adjustment = pd.Series(signals, index=base_returns.index).fillna(0)
        
        # Combine base returns with signal overlay
        expected_returns = base_returns + (signal_adjustment * 0.05)  # 5% signal impact
        
        return expected_returns
    
    def _get_regime_adjusted_params(self) -> Dict[str, float]:
        """Get optimization parameters adjusted for current regime"""
        
        base_params = {
            'risk_aversion': self.config.risk_aversion
        }
        
        if self.current_regime and hasattr(self.current_regime, 'regime_type'):
            regime_type = self.current_regime.regime_type
            
            # Adjust risk aversion based on regime
            if regime_type == "high_vol" or regime_type == "crash":
                base_params['risk_aversion'] *= 2.0  # More conservative
            elif regime_type == "low_vol":
                base_params['risk_aversion'] *= 0.8  # Slightly more aggressive
        
        return base_params
    
    def _calculate_max_turnover(self, current_positions: Dict[str, float]) -> float:
        """Calculate maximum allowed turnover based on current positions"""
        
        # Higher turnover allowed when we have fewer positions
        num_positions = len([w for w in current_positions.values() if abs(w) > 0.01])
        
        if num_positions == 0:
            return self.config.max_turnover  # Full turnover allowed from cash
        else:
            # Scale down turnover as we have more positions
            return min(self.config.max_turnover, 0.5 + 0.1 * num_positions)
    
    def _should_rebalance(self, 
                         current_date: datetime,
                         current_positions: Dict[str, float]) -> bool:
        """Determine if portfolio should be rebalanced"""
        
        # Always rebalance on first day
        if self.last_rebalance_date is None:
            return True
        
        # Check if enough time has passed (minimum 1 day)
        days_since_rebalance = (current_date - self.last_rebalance_date).days
        if days_since_rebalance < 1:
            return False
        
        # Could add drift-based rebalancing here
        # For now, use simple time-based rebalancing
        
        return True
    
    def _apply_risk_checks(self, 
                          target_weights: Dict[str, float],
                          current_positions: Dict[str, float]) -> Dict[str, float]:
        """Apply final risk checks to target weights"""
        
        # Ensure weights sum to <= 1.0
        total_weight = sum(target_weights.values())
        if total_weight > 1.0:
            # Scale proportionally
            scale_factor = 0.99 / total_weight  # Leave 1% cash buffer
            target_weights = {k: v * scale_factor for k, v in target_weights.items()}
        
        # Check individual position limits
        for asset, weight in target_weights.items():
            if weight > self.config.max_position_weight:
                target_weights[asset] = self.config.max_position_weight
            elif weight < self.config.min_position_weight and weight > 0:
                target_weights[asset] = 0.0  # Remove small positions
        
        # Check turnover limit
        current_total = sum(abs(w) for w in current_positions.values())
        target_total = sum(abs(w) for w in target_weights.values())
        
        if current_total > 0:
            turnover = sum(abs(target_weights.get(k, 0) - current_positions.get(k, 0))
                          for k in set(list(target_weights.keys()) + list(current_positions.keys())))
            
            if turnover > self.config.max_turnover:
                # Reduce turnover by interpolating between current and target
                blend_factor = self.config.max_turnover / turnover
                
                blended_weights = {}
                for asset in set(list(target_weights.keys()) + list(current_positions.keys())):
                    current_w = current_positions.get(asset, 0.0)
                    target_w = target_weights.get(asset, 0.0)
                    blended_w = current_w + blend_factor * (target_w - current_w)
                    
                    if abs(blended_w) > 0.001:  # Keep only meaningful positions
                        blended_weights[asset] = blended_w
                
                target_weights = blended_weights
        
        return target_weights


class StrategyBacktester:
    """Specialized backtester for CroweTrade strategies"""
    
    def __init__(self, 
                 model_registry: Optional[ModelRegistry] = None):
        self.model_registry = model_registry
    
    def backtest_strategy(self,
                         strategy_config: StrategyConfig,
                         backtest_config: BacktestConfig,
                         market_data: pd.DataFrame) -> PerformanceMetrics:
        """Run backtest with integrated strategy"""
        
        # Create strategy instance
        strategy = IntegratedStrategy(strategy_config, self.model_registry)
        
        # Create and run backtest
        engine = BacktestEngine(market_data, backtest_config)
        
        return engine.run_backtest(strategy)
    
    def parameter_sweep(self,
                       base_config: StrategyConfig,
                       backtest_config: BacktestConfig,
                       market_data: pd.DataFrame,
                       param_grid: Dict[str, List[Any]]) -> List[Tuple[Dict, PerformanceMetrics]]:
        """Run parameter sweep across strategy configurations"""
        
        results = []
        
        # Generate all parameter combinations
        import itertools
        
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for param_combo in itertools.product(*param_values):
            # Create configuration with current parameters
            config_dict = base_config.to_dict()
            
            for name, value in zip(param_names, param_combo):
                config_dict[name] = value
            
            # Create new config object
            test_config = StrategyConfig(**config_dict)
            
            try:
                # Run backtest
                metrics = self.backtest_strategy(test_config, backtest_config, market_data)
                
                # Store results
                param_dict = dict(zip(param_names, param_combo))
                results.append((param_dict, metrics))
                
                logger.info(f"Parameter combo {param_dict}: Sharpe={metrics.sharpe_ratio:.3f}")
                
            except Exception as e:
                logger.error(f"Parameter combo {param_combo} failed: {e}")
        
        # Sort by Sharpe ratio (descending)
        results.sort(key=lambda x: x[1].sharpe_ratio, reverse=True)
        
        return results
    
    def walk_forward_analysis(self,
                             strategy_config: StrategyConfig,
                             market_data: pd.DataFrame,
                             training_window: int = 252,
                             test_window: int = 63) -> List[PerformanceMetrics]:
        """Run walk-forward analysis"""
        
        strategy = IntegratedStrategy(strategy_config, self.model_registry)
        
        results = []
        start_date = market_data.index[training_window]
        
        while start_date < market_data.index[-test_window]:
            # Define test period
            test_start_idx = market_data.index.get_loc(start_date)
            test_end_idx = min(test_start_idx + test_window, len(market_data) - 1)
            
            test_start = market_data.index[test_start_idx]
            test_end = market_data.index[test_end_idx]
            
            # Create backtest config for this period
            period_config = BacktestConfig(
                start_date=test_start,
                end_date=test_end,
                initial_capital=1_000_000.0
            )
            
            try:
                # Run backtest
                engine = BacktestEngine(market_data, period_config)
                metrics = engine.run_backtest(strategy)
                
                # Add period information
                metrics.test_start = test_start
                metrics.test_end = test_end
                results.append(metrics)
                
                logger.info(f"Walk-forward {test_start} to {test_end}: "
                           f"Return={metrics.total_return:.3f}, Sharpe={metrics.sharpe_ratio:.3f}")
                
            except Exception as e:
                logger.error(f"Walk-forward period {test_start} to {test_end} failed: {e}")
            
            # Move to next period
            if test_end_idx + 1 < len(market_data):
                start_date = market_data.index[test_end_idx + 1]
            else:
                break
        
        return results


def create_strategy_config(**kwargs) -> StrategyConfig:
    """Convenience function to create strategy configuration"""
    return StrategyConfig(**kwargs)


def run_simple_backtest(market_data: pd.DataFrame,
                       start_date: str,
                       end_date: str,
                       strategy_params: Dict[str, Any] = None) -> PerformanceMetrics:
    """Run a simple backtest with default parameters"""
    
    # Create configurations
    strategy_config = create_strategy_config(**(strategy_params or {}))
    
    backtest_config = BacktestConfig(
        start_date=datetime.fromisoformat(start_date),
        end_date=datetime.fromisoformat(end_date)
    )
    
    # Run backtest
    backtester = StrategyBacktester()
    return backtester.backtest_strategy(strategy_config, backtest_config, market_data)
