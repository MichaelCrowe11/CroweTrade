"""Portfolio Optimizer Module - Stateless Optimization

Enhancements:
    - Injectible RNG for deterministic testing (max_sharpe / min_variance random search)
    - Public helper to compute risk contributions for external validation
    - Minor resiliency (guard against empty assets / missing returns)
"""

import math
import random
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Dict, List


class OptimizationMethod(Enum):
    MEAN_VARIANCE = "mean_variance"
    RISK_BUDGETING = "risk_budgeting"
    MAX_SHARPE = "max_sharpe"
    MIN_VARIANCE = "min_variance"


@dataclass
class OptimizationConfig:
    method: OptimizationMethod
    constraints: dict[str, Any]
    risk_aversion: float
    rebalance_threshold: float
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method.value,
            "constraints": self.constraints,
            "risk_aversion": self.risk_aversion,
            "rebalance_threshold": self.rebalance_threshold
        }


@dataclass
class OptimalPortfolio:
    weights: dict[str, float]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    metadata: dict[str, Any] | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": self.weights,
            "expected_return": self.expected_return,
            "expected_risk": self.expected_risk,
            "sharpe_ratio": self.sharpe_ratio,
            "metadata": self.metadata
        }


class PortfolioOptimizer:
    def __init__(self, config: OptimizationConfig, rng: Optional[random.Random] = None):
        self.config = config
        self.rng = rng or random.Random()
        self.optimization_methods = {
            OptimizationMethod.MEAN_VARIANCE: self._mean_variance_optimization,
            OptimizationMethod.RISK_BUDGETING: self._risk_budgeting_optimization,
            OptimizationMethod.MAX_SHARPE: self._max_sharpe_optimization,
            OptimizationMethod.MIN_VARIANCE: self._min_variance_optimization
        }
        
    def optimize(self, assets: List[str], expected_returns: Dict[str, float],
                covariance_matrix: Dict[tuple[str, str], float], 
                current_weights: Optional[Dict[str, float]] = None) -> OptimalPortfolio:
        if not assets:
            raise ValueError("assets list must not be empty")
        
        optimization_fn = self.optimization_methods.get(
            self.config.method, 
            self._mean_variance_optimization
        )
        
        raw_weights = optimization_fn(assets, expected_returns, covariance_matrix)
        
        weights = self._apply_constraints(raw_weights)
        
        if current_weights and self._should_rebalance(current_weights, weights):
            weights = self._smooth_rebalance(current_weights, weights)
        
        portfolio_return = self._calculate_portfolio_return(weights, expected_returns)
        portfolio_risk = self._calculate_portfolio_risk(weights, covariance_matrix)
        sharpe = self._calculate_sharpe_ratio(portfolio_return, portfolio_risk)
        
        return OptimalPortfolio(
            weights=weights,
            expected_return=portfolio_return,
            expected_risk=portfolio_risk,
            sharpe_ratio=sharpe,
            metadata={
                "optimization_method": self.config.method.value,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    def _mean_variance_optimization(self, assets: List[str], expected_returns: Dict[str, float],
                                   covariance_matrix: Dict[tuple[str, str], float]) -> Dict[str, float]:
        n = len(assets)
        equal_weight = 1.0 / n
        
        weights = dict.fromkeys(assets, equal_weight)
        
        for _ in range(10):
            gradient = self._calculate_gradient(weights, expected_returns, covariance_matrix)
            for asset in assets:
                weights[asset] -= 0.01 * gradient.get(asset, 0.0)
            
            weights = self._normalize_weights(weights)
        
        return weights
    
    def _risk_budgeting_optimization(self, assets: List[str], expected_returns: Dict[str, float],
                                    covariance_matrix: Dict[tuple[str, str], float]) -> Dict[str, float]:
        n = len(assets)
        risk_budget = 1.0 / n
        
        weights = dict.fromkeys(assets, risk_budget)
        
        for _ in range(10):
            risk_contributions = self._calculate_risk_contributions(weights, covariance_matrix)
            
            for asset in assets:
                target_contribution = risk_budget
                actual_contribution = risk_contributions.get(asset, 0.0)
                adjustment = (target_contribution - actual_contribution) * 0.1
                weights[asset] = max(0.0, weights[asset] + adjustment)
            
            weights = self._normalize_weights(weights)
        
        return weights
    
    def _max_sharpe_optimization(self, assets: List[str], expected_returns: Dict[str, float],
                                covariance_matrix: Dict[tuple[str, str], float]) -> Dict[str, float]:
        best_weights = None
        best_sharpe = -float('inf')
        
        for _ in range(100):
            test_weights = self._generate_random_weights(assets)
            portfolio_return = self._calculate_portfolio_return(test_weights, expected_returns)
            portfolio_risk = self._calculate_portfolio_risk(test_weights, covariance_matrix)
            sharpe = self._calculate_sharpe_ratio(portfolio_return, portfolio_risk)
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights = test_weights
        
        return best_weights or {asset: 1.0/len(assets) for asset in assets}
    
    def _min_variance_optimization(self, assets: List[str], expected_returns: Dict[str, float],
                                  covariance_matrix: Dict[tuple[str, str], float]) -> Dict[str, float]:
        best_weights = None
        min_risk = float('inf')
        
        for _ in range(100):
            test_weights = self._generate_random_weights(assets)
            portfolio_risk = self._calculate_portfolio_risk(test_weights, covariance_matrix)
            
            if portfolio_risk < min_risk:
                min_risk = portfolio_risk
                best_weights = test_weights
        
        return best_weights or {asset: 1.0/len(assets) for asset in assets}
    
    def _apply_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        min_weight = self.config.constraints.get("min_weight", 0.0)
        max_weight = self.config.constraints.get("max_weight", 1.0)
        
        constrained = {}
        for asset, weight in weights.items():
            constrained[asset] = max(min_weight, min(max_weight, weight))
        
        return self._normalize_weights(constrained)
    
    def _should_rebalance(self, current: Dict[str, float], target: Dict[str, float]) -> bool:
        total_deviation = sum(
            abs(current.get(asset, 0.0) - target.get(asset, 0.0)) 
            for asset in set(current) | set(target)
        )
        return total_deviation > self.config.rebalance_threshold
    
    def _smooth_rebalance(self, current: Dict[str, float], target: Dict[str, float]) -> Dict[str, float]:
        smoothing_factor = 0.3
        smoothed = {}
        
        for asset in set(current) | set(target):
            current_weight = current.get(asset, 0.0)
            target_weight = target.get(asset, 0.0)
            smoothed[asset] = current_weight + smoothing_factor * (target_weight - current_weight)
        
        return self._normalize_weights(smoothed)
    
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        total = sum(weights.values())
        if total == 0:
            return {asset: 1.0/len(weights) for asset in weights}
        return {asset: weight/total for asset, weight in weights.items()}
    
    def _generate_random_weights(self, assets: List[str]) -> Dict[str, float]:
        weights = {asset: self.rng.random() for asset in assets}
        return self._normalize_weights(weights)
    
    def _calculate_portfolio_return(self, weights: Dict[str, float], 
                                   expected_returns: Dict[str, float]) -> float:
        return sum(weights.get(asset, 0.0) * expected_returns.get(asset, 0.0) 
                  for asset in weights)
    
    def _calculate_portfolio_risk(self, weights: Dict[str, float], 
                                 covariance_matrix: Dict[tuple[str, str], float]) -> float:
        variance = 0.0
        for asset1 in weights:
            for asset2 in weights:
                w1 = weights[asset1]
                w2 = weights[asset2]
                cov = covariance_matrix.get((asset1, asset2), 0.0)
                variance += w1 * w2 * cov
        return math.sqrt(max(0.0, variance))
    
    def _calculate_sharpe_ratio(self, portfolio_return: float, portfolio_risk: float, 
                               risk_free_rate: float = 0.0) -> float:
        if portfolio_risk == 0:
            return 0.0
        return (portfolio_return - risk_free_rate) / portfolio_risk
    
    def _calculate_gradient(self, weights: Dict[str, float], expected_returns: Dict[str, float],
                          covariance_matrix: Dict[tuple[str, str], float]) -> Dict[str, float]:
        gradient = {}
        for asset in weights:
            return_component = -expected_returns.get(asset, 0.0)
            risk_component = 0.0
            for other_asset in weights:
                cov = covariance_matrix.get((asset, other_asset), 0.0)
                risk_component += 2 * self.config.risk_aversion * weights[other_asset] * cov
            gradient[asset] = return_component + risk_component
        return gradient
    
    def _calculate_risk_contributions(self, weights: Dict[str, float],
                                     covariance_matrix: Dict[tuple[str, str], float]) -> Dict[str, float]:
        portfolio_risk = self._calculate_portfolio_risk(weights, covariance_matrix)
        if portfolio_risk == 0:
            return dict.fromkeys(weights, 0.0)
        
        contributions = {}
        for asset in weights:
            marginal_contribution = 0.0
            for other_asset in weights:
                cov = covariance_matrix.get((asset, other_asset), 0.0)
                marginal_contribution += weights[other_asset] * cov
            
            contributions[asset] = (weights[asset] * marginal_contribution) / portfolio_risk
        
        return contributions

    # Public helpers -----------------------------------------------------
    def risk_contributions(self, weights: Dict[str, float],
                           covariance_matrix: Dict[tuple[str, str], float]) -> Dict[str, float]:
        """Public wrapper for validating risk budgeting in tests."""
        return self._calculate_risk_contributions(weights, covariance_matrix)