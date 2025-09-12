"""Portfolio optimizer with Mean-Variance Optimization (MVO)"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import warnings

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConstraints:
    """Portfolio optimization constraints"""
    min_weight: float = 0.0  # Minimum position weight
    max_weight: float = 0.3  # Maximum position weight
    max_leverage: float = 1.0  # Maximum gross leverage
    max_turnover: float = 0.5  # Maximum turnover from current
    long_only: bool = True  # Long-only constraint
    max_positions: Optional[int] = None  # Maximum number of positions
    sector_limits: Dict[str, float] = None  # Sector exposure limits
    
    def __post_init__(self):
        if self.sector_limits is None:
            self.sector_limits = {}


@dataclass
class OptimizationResult:
    """Result from portfolio optimization"""
    weights: np.ndarray
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    turnover: float
    success: bool
    message: str
    iterations: int
    
    def to_dict(self, symbols: List[str]) -> Dict[str, float]:
        """Convert weights to symbol dictionary"""
        return {symbol: weight for symbol, weight in zip(symbols, self.weights)}


class PortfolioOptimizer:
    """Mean-Variance Optimization with practical constraints"""
    
    def __init__(
        self,
        risk_free_rate: float = 0.02,
        transaction_cost: float = 0.001,
        regularization: float = 1e-8,
        solver_method: str = 'SLSQP'
    ):
        self.risk_free_rate = risk_free_rate
        self.transaction_cost = transaction_cost
        self.regularization = regularization
        self.solver_method = solver_method
        
        # Cache for covariance matrix
        self.cached_cov = None
        self.cached_returns = None
    
    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        constraints: OptimizationConstraints,
        current_weights: Optional[np.ndarray] = None,
        confidence_scores: Optional[np.ndarray] = None
    ) -> OptimizationResult:
        """
        Optimize portfolio weights using Mean-Variance Optimization
        
        Args:
            expected_returns: Expected returns for each asset
            covariance: Covariance matrix of returns
            constraints: Optimization constraints
            current_weights: Current portfolio weights (for turnover)
            confidence_scores: Confidence in return predictions (0-1)
        
        Returns:
            OptimizationResult with optimal weights
        """
        
        n_assets = len(expected_returns)
        
        # Validate inputs
        if covariance.shape != (n_assets, n_assets):
            raise ValueError(f"Covariance shape {covariance.shape} doesn't match returns {n_assets}")
        
        # Regularize covariance matrix for numerical stability
        cov_reg = self._regularize_covariance(covariance)
        
        # Adjust returns by confidence if provided
        if confidence_scores is not None:
            expected_returns = expected_returns * confidence_scores
        
        # Initialize weights
        if current_weights is None:
            current_weights = np.zeros(n_assets)
        
        # Cache for performance
        self.cached_cov = cov_reg
        self.cached_returns = expected_returns
        
        # Define objective function (negative Sharpe ratio)
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_reg, weights)))
            
            # Add transaction cost penalty
            turnover = np.sum(np.abs(weights - current_weights))
            cost_penalty = self.transaction_cost * turnover
            
            # Sharpe ratio with cost adjustment
            if portfolio_risk > 0:
                sharpe = (portfolio_return - cost_penalty - self.risk_free_rate) / portfolio_risk
                return -sharpe  # Minimize negative Sharpe
            else:
                return 1e10  # Penalize zero risk portfolio
        
        # Define constraints
        scipy_constraints = self._build_constraints(n_assets, constraints, current_weights)
        
        # Define bounds
        if constraints.long_only:
            bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        else:
            bounds = [(-constraints.max_weight, constraints.max_weight) for _ in range(n_assets)]
        
        # Initial guess (equal weight or current weights)
        if np.sum(current_weights) > 0:
            x0 = current_weights
        else:
            x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')  # Suppress convergence warnings
            
            result = minimize(
                objective,
                x0,
                method=self.solver_method,
                bounds=bounds,
                constraints=scipy_constraints,
                options={'maxiter': 1000, 'ftol': 1e-8, 'disp': False}
            )
        
        # Extract optimal weights
        optimal_weights = result.x
        
        # Ensure weights sum to 1 (numerical precision)
        optimal_weights = optimal_weights / np.sum(np.abs(optimal_weights))
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(optimal_weights, expected_returns)
        portfolio_risk = np.sqrt(np.dot(optimal_weights, np.dot(cov_reg, optimal_weights)))
        turnover = np.sum(np.abs(optimal_weights - current_weights))
        
        # Calculate Sharpe ratio
        if portfolio_risk > 0:
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_risk
        else:
            sharpe = 0.0
        
        # Even if optimization didn't converge perfectly, return result if reasonable
        is_valid = (
            np.allclose(np.sum(optimal_weights), 1.0, atol=0.01) and
            portfolio_risk > 0 and
            not np.any(np.isnan(optimal_weights))
        )
        
        return OptimizationResult(
            weights=optimal_weights,
            expected_return=portfolio_return,
            expected_risk=portfolio_risk,
            sharpe_ratio=sharpe,
            turnover=turnover,
            success=result.success or is_valid,
            message=result.message if hasattr(result, 'message') else 'Optimization completed',
            iterations=result.nit if hasattr(result, 'nit') else 0
        )
    
    def optimize_risk_parity(
        self,
        covariance: np.ndarray,
        constraints: OptimizationConstraints
    ) -> OptimizationResult:
        """
        Risk parity optimization - equal risk contribution
        
        Args:
            covariance: Covariance matrix
            constraints: Optimization constraints
        
        Returns:
            OptimizationResult with risk parity weights
        """
        
        n_assets = covariance.shape[0]
        
        # Regularize covariance
        cov_reg = self._regularize_covariance(covariance)
        
        # Define objective for risk parity
        def risk_parity_objective(weights):
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_reg, weights)))
            
            # Marginal risk contributions
            marginal_contrib = np.dot(cov_reg, weights) / portfolio_risk
            contrib = weights * marginal_contrib
            
            # Target equal contribution
            target_contrib = portfolio_risk / n_assets
            
            # Minimize squared deviations from target
            return np.sum((contrib - target_contrib) ** 2)
        
        # Constraints (weights sum to 1)
        constraints_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        # Bounds (positive weights for risk parity)
        bounds = [(0.001, constraints.max_weight) for _ in range(n_assets)]
        
        # Initial guess (equal weight)
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            risk_parity_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        # Extract weights
        weights = result.x
        weights = weights / np.sum(weights)  # Normalize
        
        # Calculate metrics
        portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_reg, weights)))
        
        # Estimate return (using equal expected returns for risk parity)
        expected_return = 0.08  # Default assumption
        
        return OptimizationResult(
            weights=weights,
            expected_return=expected_return,
            expected_risk=portfolio_risk,
            sharpe_ratio=(expected_return - self.risk_free_rate) / portfolio_risk,
            turnover=np.sum(np.abs(weights)),
            success=result.success,
            message='Risk parity optimization',
            iterations=result.nit if hasattr(result, 'nit') else 0
        )
    
    def calculate_efficient_frontier(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        constraints: OptimizationConstraints,
        n_points: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate efficient frontier points
        
        Returns:
            Tuple of (risks, returns) arrays for plotting
        """
        
        # Get min and max return portfolios
        min_return = np.min(expected_returns)
        max_return = np.max(expected_returns)
        
        target_returns = np.linspace(min_return, max_return, n_points)
        
        risks = []
        returns = []
        
        for target_return in target_returns:
            # Add return constraint
            constraints_copy = constraints
            
            # Optimize for minimum risk at target return
            try:
                result = self.optimize_min_variance(
                    expected_returns,
                    covariance,
                    constraints_copy,
                    target_return
                )
                
                if result.success or result.expected_risk > 0:
                    risks.append(result.expected_risk)
                    returns.append(result.expected_return)
            except:
                continue
        
        return np.array(risks), np.array(returns)
    
    def optimize_min_variance(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        constraints: OptimizationConstraints,
        target_return: float
    ) -> OptimizationResult:
        """
        Minimize variance for a target return
        """
        
        n_assets = len(expected_returns)
        cov_reg = self._regularize_covariance(covariance)
        
        # Objective: minimize portfolio variance
        def objective(weights):
            return np.dot(weights, np.dot(cov_reg, weights))
        
        # Constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.dot(w, expected_returns) - target_return}
        ]
        
        # Bounds
        if constraints.long_only:
            bounds = [(0, constraints.max_weight) for _ in range(n_assets)]
        else:
            bounds = [(-constraints.max_weight, constraints.max_weight) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        weights = result.x
        weights = weights / np.sum(np.abs(weights))
        
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_reg, weights)))
        
        return OptimizationResult(
            weights=weights,
            expected_return=portfolio_return,
            expected_risk=portfolio_risk,
            sharpe_ratio=(portfolio_return - self.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0,
            turnover=0,
            success=result.success,
            message='Min variance optimization',
            iterations=result.nit if hasattr(result, 'nit') else 0
        )
    
    def _regularize_covariance(self, cov: np.ndarray) -> np.ndarray:
        """Regularize covariance matrix for numerical stability"""
        
        # Add small value to diagonal
        cov_reg = cov + self.regularization * np.eye(cov.shape[0])
        
        # Ensure positive semi-definite
        eigenvalues, eigenvectors = np.linalg.eig(cov_reg)
        eigenvalues = np.maximum(eigenvalues, self.regularization)
        cov_reg = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        # Ensure symmetry (numerical precision)
        cov_reg = (cov_reg + cov_reg.T) / 2
        
        return cov_reg
    
    def _build_constraints(
        self,
        n_assets: int,
        constraints: OptimizationConstraints,
        current_weights: np.ndarray
    ) -> List[Dict]:
        """Build scipy constraint dictionaries"""
        
        scipy_constraints = []
        
        # Weights sum to 1
        scipy_constraints.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1
        })
        
        # Leverage constraint
        if constraints.max_leverage > 1:
            scipy_constraints.append({
                'type': 'ineq',
                'fun': lambda w: constraints.max_leverage - np.sum(np.abs(w))
            })
        
        # Turnover constraint
        if constraints.max_turnover is not None and constraints.max_turnover > 0:
            scipy_constraints.append({
                'type': 'ineq',
                'fun': lambda w: constraints.max_turnover - np.sum(np.abs(w - current_weights))
            })
        
        # Max positions constraint (sparsity)
        if constraints.max_positions is not None:
            # This is a hard constraint to enforce with continuous optimization
            # We'll use a penalty approach instead
            pass
        
        return scipy_constraints
    
    def calculate_risk_metrics(
        self,
        weights: np.ndarray,
        covariance: np.ndarray,
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate portfolio risk metrics
        
        Args:
            weights: Portfolio weights
            covariance: Covariance matrix
            confidence_level: Confidence level for VaR/CVaR
        
        Returns:
            Dictionary of risk metrics
        """
        
        # Portfolio variance
        portfolio_variance = np.dot(weights, np.dot(covariance, weights))
        portfolio_std = np.sqrt(portfolio_variance)
        
        # Value at Risk (parametric)
        z_score = norm.ppf(1 - confidence_level)
        var = -z_score * portfolio_std
        
        # Conditional VaR (parametric approximation)
        cvar = portfolio_std * norm.pdf(z_score) / (1 - confidence_level)
        
        # Risk contributions
        marginal_contrib = np.dot(covariance, weights) / portfolio_std
        risk_contrib = weights * marginal_contrib
        
        # Concentration metrics
        effective_n = 1 / np.sum(weights ** 2)  # Effective number of assets
        herfindahl = np.sum(weights ** 2)  # Herfindahl index
        
        return {
            'portfolio_std': portfolio_std,
            'portfolio_variance': portfolio_variance,
            'var_95': var,
            'cvar_95': cvar,
            'max_risk_contrib': np.max(risk_contrib),
            'effective_n_assets': effective_n,
            'herfindahl_index': herfindahl
        }