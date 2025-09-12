"""Portfolio Manager Agent with integrated optimization"""
import logging
from typing import Dict, List, Optional, Any
import numpy as np
from dataclasses import dataclass
from datetime import datetime

from crowetrade.core.agent import BaseAgent
from crowetrade.core.events import Event, EventType
from crowetrade.portfolio.optimizer import (
    PortfolioOptimizer,
    OptimizationConstraints,
    OptimizationResult
)
from crowetrade.data.feature_store import PersistentFeatureStore
from crowetrade.config.policy_manager import PolicyHotReloadManager

logger = logging.getLogger(__name__)


@dataclass
class PortfolioState:
    """Current portfolio state"""
    positions: Dict[str, float]  # Symbol -> quantity
    weights: Dict[str, float]  # Symbol -> weight
    cash: float
    total_value: float
    timestamp: float
    
    def get_weights_array(self, symbols: List[str]) -> np.ndarray:
        """Get weights as numpy array in symbol order"""
        return np.array([self.weights.get(s, 0.0) for s in symbols])


class PortfolioManagerAgent(BaseAgent):
    """Agent responsible for portfolio optimization and rebalancing"""
    
    def __init__(
        self,
        agent_id: str,
        optimizer: Optional[PortfolioOptimizer] = None,
        feature_store: Optional[PersistentFeatureStore] = None,
        policy_manager: Optional[PolicyHotReloadManager] = None,
        rebalance_threshold: float = 0.05,
        min_rebalance_interval: int = 3600  # seconds
    ):
        super().__init__(agent_id)
        
        # Core components
        self.optimizer = optimizer or PortfolioOptimizer()
        self.feature_store = feature_store or PersistentFeatureStore()
        self.policy_manager = policy_manager
        
        # Configuration
        self.rebalance_threshold = rebalance_threshold
        self.min_rebalance_interval = min_rebalance_interval
        
        # State tracking
        self.current_state: Optional[PortfolioState] = None
        self.target_weights: Dict[str, float] = {}
        self.last_rebalance_time: float = 0
        self.optimization_history: List[OptimizationResult] = []
        
        # Constraints from policy
        self.constraints = OptimizationConstraints()
        
        # Register for policy updates
        if self.policy_manager:
            self.policy_manager.register_change_callback(
                self._on_policy_change
            )
    
    async def process_event(self, event: Event) -> Optional[Event]:
        """Process incoming events"""
        
        if event.type == EventType.SIGNAL:
            return await self._handle_signal(event)
        
        elif event.type == EventType.MARKET_DATA:
            return await self._handle_market_data(event)
        
        elif event.type == EventType.POSITION_UPDATE:
            return await self._handle_position_update(event)
        
        elif event.type == EventType.RISK_UPDATE:
            return await self._handle_risk_update(event)
        
        return None
    
    async def _handle_signal(self, event: Event) -> Optional[Event]:
        """Handle incoming signals and optimize portfolio"""
        
        signal_data = event.data
        
        # Extract predictions and confidence
        symbols = signal_data.get('symbols', [])
        predictions = signal_data.get('predictions', {})
        confidence_scores = signal_data.get('confidence', {})
        
        if not symbols:
            return None
        
        # Get market data from feature store
        returns_data = self._get_expected_returns(symbols, predictions)
        covariance_data = self._get_covariance_matrix(symbols)
        
        if returns_data is None or covariance_data is None:
            logger.warning("Missing data for optimization")
            return None
        
        # Prepare optimization inputs
        expected_returns = np.array([returns_data[s] for s in symbols])
        covariance = covariance_data
        confidence = np.array([confidence_scores.get(s, 0.5) for s in symbols])
        
        # Get current weights
        current_weights = None
        if self.current_state:
            current_weights = self.current_state.get_weights_array(symbols)
        
        # Apply policy constraints
        constraints = self._get_constraints_from_policy()
        
        # Optimize portfolio
        try:
            result = self.optimizer.optimize(
                expected_returns=expected_returns,
                covariance=covariance,
                constraints=constraints,
                current_weights=current_weights,
                confidence_scores=confidence
            )
            
            if result.success:
                # Store optimization result
                self.optimization_history.append(result)
                
                # Convert to target weights
                self.target_weights = result.to_dict(symbols)
                
                # Store in feature store
                self.feature_store.put(
                    f"portfolio_weights_{self.agent_id}",
                    {
                        'weights': self.target_weights,
                        'expected_return': result.expected_return,
                        'expected_risk': result.expected_risk,
                        'sharpe_ratio': result.sharpe_ratio,
                        'timestamp': datetime.now().isoformat()
                    },
                    ttl_seconds=86400  # 24 hours
                )
                
                # Check if rebalancing is needed
                if self._should_rebalance():
                    return self._create_rebalance_event()
                
                logger.info(
                    f"Portfolio optimized - Sharpe: {result.sharpe_ratio:.3f}, "
                    f"Return: {result.expected_return:.2%}, Risk: {result.expected_risk:.2%}"
                )
            else:
                logger.warning(f"Optimization failed: {result.message}")
        
        except Exception as e:
            logger.error(f"Optimization error: {e}")
        
        return None
    
    async def _handle_market_data(self, event: Event) -> Optional[Event]:
        """Update market data in feature store"""
        
        market_data = event.data
        
        # Store price data
        for symbol, price_data in market_data.get('prices', {}).items():
            self.feature_store.put(
                f"price_{symbol}",
                price_data,
                ttl_seconds=3600  # 1 hour
            )
        
        # Store volume data
        for symbol, volume_data in market_data.get('volumes', {}).items():
            self.feature_store.put(
                f"volume_{symbol}",
                volume_data,
                ttl_seconds=3600
            )
        
        return None
    
    async def _handle_position_update(self, event: Event) -> Optional[Event]:
        """Update current portfolio state"""
        
        position_data = event.data
        
        # Update state
        self.current_state = PortfolioState(
            positions=position_data.get('positions', {}),
            weights=position_data.get('weights', {}),
            cash=position_data.get('cash', 0.0),
            total_value=position_data.get('total_value', 0.0),
            timestamp=position_data.get('timestamp', datetime.now().timestamp())
        )
        
        # Store in feature store
        self.feature_store.put(
            f"portfolio_state_{self.agent_id}",
            {
                'positions': self.current_state.positions,
                'weights': self.current_state.weights,
                'cash': self.current_state.cash,
                'total_value': self.current_state.total_value
            },
            ttl_seconds=86400
        )
        
        return None
    
    async def _handle_risk_update(self, event: Event) -> Optional[Event]:
        """Handle risk updates and adjust constraints"""
        
        risk_data = event.data
        
        # Update constraints based on risk
        if risk_data.get('high_risk', False):
            # Reduce position sizes in high risk
            self.constraints.max_weight = min(
                self.constraints.max_weight * 0.8,
                0.2
            )
            self.constraints.max_leverage = 1.0
            
            logger.info("Risk constraints tightened due to high risk")
        
        # Store risk metrics
        self.feature_store.put(
            f"risk_metrics_{self.agent_id}",
            risk_data,
            ttl_seconds=3600
        )
        
        return None
    
    def _get_expected_returns(
        self,
        symbols: List[str],
        predictions: Dict[str, float]
    ) -> Optional[Dict[str, float]]:
        """Get expected returns from predictions and historical data"""
        
        returns = {}
        
        for symbol in symbols:
            # Use prediction if available
            if symbol in predictions:
                returns[symbol] = predictions[symbol]
            else:
                # Fall back to historical returns from feature store
                hist_returns = self.feature_store.get(f"returns_{symbol}")
                if hist_returns:
                    returns[symbol] = hist_returns.data.get('expected', 0.08)
                else:
                    returns[symbol] = 0.08  # Default
        
        return returns if len(returns) == len(symbols) else None
    
    def _get_covariance_matrix(
        self,
        symbols: List[str]
    ) -> Optional[np.ndarray]:
        """Get covariance matrix from feature store"""
        
        # Try to get pre-computed covariance
        cov_feature = self.feature_store.get("covariance_matrix")
        
        if cov_feature and 'matrix' in cov_feature.data:
            matrix_data = cov_feature.data['matrix']
            symbol_order = cov_feature.data.get('symbols', [])
            
            # Reorder for requested symbols
            if set(symbols) == set(symbol_order):
                indices = [symbol_order.index(s) for s in symbols]
                cov_matrix = np.array(matrix_data)
                return cov_matrix[np.ix_(indices, indices)]
        
        # Fall back to simple correlation estimate
        n = len(symbols)
        correlation = 0.3  # Default correlation
        volatility = 0.2  # Default volatility
        
        # Create correlation matrix
        cov = np.full((n, n), correlation * volatility ** 2)
        np.fill_diagonal(cov, volatility ** 2)
        
        return cov
    
    def _get_constraints_from_policy(self) -> OptimizationConstraints:
        """Get optimization constraints from policy"""
        
        constraints = self.constraints
        
        if self.policy_manager:
            # Try to apply portfolio policy
            policy_applied = self.policy_manager.apply_policy(
                'portfolio_optimization',
                constraints
            )
            
            if not policy_applied:
                # Fall back to risk policy
                self.policy_manager.apply_policy(
                    'risk_management',
                    constraints
                )
        
        return constraints
    
    def _should_rebalance(self) -> bool:
        """Check if portfolio should be rebalanced"""
        
        if not self.current_state or not self.target_weights:
            return False
        
        # Check time since last rebalance
        time_since_rebalance = datetime.now().timestamp() - self.last_rebalance_time
        if time_since_rebalance < self.min_rebalance_interval:
            return False
        
        # Check weight drift
        max_drift = 0.0
        for symbol, target_weight in self.target_weights.items():
            current_weight = self.current_state.weights.get(symbol, 0.0)
            drift = abs(current_weight - target_weight)
            max_drift = max(max_drift, drift)
        
        return max_drift > self.rebalance_threshold
    
    def _create_rebalance_event(self) -> Event:
        """Create rebalancing event"""
        
        self.last_rebalance_time = datetime.now().timestamp()
        
        return Event(
            type=EventType.REBALANCE,
            source=self.agent_id,
            data={
                'target_weights': self.target_weights,
                'current_weights': self.current_state.weights if self.current_state else {},
                'reason': 'portfolio_optimization',
                'timestamp': datetime.now().isoformat()
            }
        )
    
    def _on_policy_change(self, policy_name: str, policy: Any) -> None:
        """Handle policy changes"""
        
        if policy_name in ['portfolio_optimization', 'risk_management']:
            logger.info(f"Policy {policy_name} updated, refreshing constraints")
            self.constraints = self._get_constraints_from_policy()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get portfolio metrics"""
        
        metrics = {
            'agent_id': self.agent_id,
            'has_target_weights': bool(self.target_weights),
            'optimization_count': len(self.optimization_history)
        }
        
        if self.current_state:
            metrics.update({
                'position_count': len(self.current_state.positions),
                'total_value': self.current_state.total_value,
                'cash': self.current_state.cash
            })
        
        if self.optimization_history:
            latest = self.optimization_history[-1]
            metrics.update({
                'latest_sharpe': latest.sharpe_ratio,
                'latest_return': latest.expected_return,
                'latest_risk': latest.expected_risk,
                'latest_turnover': latest.turnover
            })
        
        return metrics