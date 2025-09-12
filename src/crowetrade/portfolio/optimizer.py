"""Mock Portfolio Optimizer for Testing

Provides mock implementations of portfolio optimization functionality
when the actual components are not available.
"""

from enum import Enum
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Portfolio optimization methods"""
    MEAN_VARIANCE = "mean_variance"
    RISK_PARITY = "risk_parity"
    EQUAL_WEIGHT = "equal_weight"
    MINIMUM_VARIANCE = "minimum_variance"
    MAXIMUM_DIVERSIFICATION = "maximum_diversification"


class PortfolioOptimizer:
    """Mock Portfolio Optimizer for testing"""
    
    def __init__(self):
        logger.info("Mock PortfolioOptimizer initialized")
    
    def optimize_portfolio(self,
                          expected_returns,
                          returns_data, 
                          method: OptimizationMethod = OptimizationMethod.MEAN_VARIANCE,
                          risk_aversion: float = 1.0,
                          constraints: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Mock portfolio optimization"""
        
        # Simple mock: equal weight allocation
        try:
            if hasattr(returns_data, 'columns'):
                assets = list(returns_data.columns)
            elif hasattr(expected_returns, 'index'):
                assets = list(expected_returns.index)
            else:
                return {}
            
            if not assets:
                return {}
            
            # Equal weight allocation (modified by constraints)
            base_weight = 1.0 / len(assets)
            
            # Apply constraints if provided
            max_weight = constraints.get('max_weight', 1.0) if constraints else 1.0
            min_weight = constraints.get('min_weight', 0.0) if constraints else 0.0
            
            # Cap individual weights
            final_weight = max(min_weight, min(base_weight, max_weight))
            
            # Create portfolio
            portfolio = {asset: final_weight for asset in assets}
            
            # Normalize to ensure sum <= 1.0
            total_weight = sum(portfolio.values())
            if total_weight > 1.0:
                scale_factor = 0.99 / total_weight
                portfolio = {k: v * scale_factor for k, v in portfolio.items()}
            
            return portfolio
            
        except Exception as e:
            logger.error(f"Mock optimization failed: {e}")
            return {}
