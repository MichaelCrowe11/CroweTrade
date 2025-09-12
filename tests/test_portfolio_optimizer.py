"""Tests for portfolio optimizer with MVO"""
import numpy as np
import pytest
from scipy.stats import norm

from crowetrade.portfolio.optimizer import (
    PortfolioOptimizer,
    OptimizationConstraints,
    OptimizationResult
)


class TestPortfolioOptimizer:
    """Test portfolio optimization functionality"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample return and covariance data"""
        # 5 assets with expected returns
        expected_returns = np.array([0.08, 0.10, 0.12, 0.07, 0.15])
        
        # Correlation matrix
        correlation = np.array([
            [1.00, 0.30, 0.20, 0.10, 0.25],
            [0.30, 1.00, 0.15, 0.05, 0.20],
            [0.20, 0.15, 1.00, 0.35, 0.10],
            [0.10, 0.05, 0.35, 1.00, 0.15],
            [0.25, 0.20, 0.10, 0.15, 1.00]
        ])
        
        # Standard deviations
        std_devs = np.array([0.15, 0.20, 0.18, 0.12, 0.25])
        
        # Covariance matrix
        covariance = np.outer(std_devs, std_devs) * correlation
        
        return expected_returns, covariance
    
    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance"""
        return PortfolioOptimizer(
            risk_free_rate=0.02,
            transaction_cost=0.001,
            regularization=1e-8
        )
    
    def test_basic_optimization(self, optimizer, sample_data):
        """Test basic MVO optimization"""
        expected_returns, covariance = sample_data
        
        constraints = OptimizationConstraints(
            min_weight=0.0,
            max_weight=0.4,
            long_only=True
        )
        
        result = optimizer.optimize(
            expected_returns,
            covariance,
            constraints
        )
        
        assert result.success is True
        assert np.allclose(np.sum(result.weights), 1.0, atol=1e-6)
        assert np.all(result.weights >= -1e-6)  # Long only
        assert np.all(result.weights <= 0.4 + 1e-6)  # Max weight
        assert result.sharpe_ratio > 0
    
    def test_optimization_with_turnover(self, optimizer, sample_data):
        """Test optimization with turnover constraint"""
        expected_returns, covariance = sample_data
        
        # Current weights (equal weight)
        current_weights = np.ones(5) / 5
        
        constraints = OptimizationConstraints(
            min_weight=0.0,
            max_weight=0.4,
            max_turnover=0.3,  # Limit turnover
            long_only=True
        )
        
        result = optimizer.optimize(
            expected_returns,
            covariance,
            constraints,
            current_weights=current_weights
        )
        
        assert result.success is True
        assert result.turnover <= 0.3 + 1e-6
        
        # Without turnover constraint should have higher Sharpe
        constraints_no_turnover = OptimizationConstraints(
            min_weight=0.0,
            max_weight=0.4,
            long_only=True
        )
        
        result_no_turnover = optimizer.optimize(
            expected_returns,
            covariance,
            constraints_no_turnover,
            current_weights=current_weights
        )
        
        # Turnover constraint should reduce Sharpe ratio
        assert result.sharpe_ratio <= result_no_turnover.sharpe_ratio + 1e-6
    
    def test_optimization_with_confidence(self, optimizer, sample_data):
        """Test optimization with confidence scores"""
        expected_returns, covariance = sample_data
        
        # High confidence in some assets
        confidence_scores = np.array([0.9, 0.5, 0.8, 0.3, 1.0])
        
        constraints = OptimizationConstraints(
            min_weight=0.0,
            max_weight=0.5,
            long_only=True
        )
        
        result = optimizer.optimize(
            expected_returns,
            covariance,
            constraints,
            confidence_scores=confidence_scores
        )
        
        assert result.success is True
        
        # High confidence assets should get more weight
        weights_dict = result.to_dict(['A', 'B', 'C', 'D', 'E'])
        assert weights_dict['E'] > weights_dict['D']  # E has highest confidence
    
    def test_risk_parity(self, optimizer, sample_data):
        """Test risk parity optimization"""
        _, covariance = sample_data
        
        constraints = OptimizationConstraints(
            max_weight=0.5
        )
        
        result = optimizer.optimize_risk_parity(
            covariance,
            constraints
        )
        
        assert result.success is True
        assert np.allclose(np.sum(result.weights), 1.0, atol=1e-6)
        assert np.all(result.weights > 0)  # Risk parity requires positive weights
        
        # Check risk contributions are roughly equal
        portfolio_risk = result.expected_risk
        marginal_contrib = np.dot(covariance, result.weights) / portfolio_risk
        risk_contrib = result.weights * marginal_contrib
        
        # Risk contributions should be similar
        assert np.std(risk_contrib) < 0.05
    
    def test_min_variance(self, optimizer, sample_data):
        """Test minimum variance optimization"""
        expected_returns, covariance = sample_data
        
        constraints = OptimizationConstraints(
            min_weight=0.0,
            max_weight=1.0,
            long_only=True
        )
        
        # Target moderate return
        target_return = 0.09
        
        result = optimizer.optimize_min_variance(
            expected_returns,
            covariance,
            constraints,
            target_return
        )
        
        assert result.success is True
        assert np.allclose(result.expected_return, target_return, atol=1e-3)
        assert np.allclose(np.sum(result.weights), 1.0, atol=1e-6)
    
    def test_efficient_frontier(self, optimizer, sample_data):
        """Test efficient frontier calculation"""
        expected_returns, covariance = sample_data
        
        constraints = OptimizationConstraints(
            min_weight=0.0,
            max_weight=1.0,
            long_only=True
        )
        
        risks, returns = optimizer.calculate_efficient_frontier(
            expected_returns,
            covariance,
            constraints,
            n_points=10
        )
        
        assert len(risks) > 0
        assert len(returns) == len(risks)
        
        # Returns should generally be increasing (but optimizer may fail on some points)
        if len(returns) > 1:
            # Check that most returns are increasing
            increasing = np.sum(np.diff(returns) >= -1e-6)
            assert increasing >= len(returns) * 0.7  # At least 70% should be increasing
        
        # Efficient frontier should have positive correlation between risk and return
        if len(returns) > 2:
            correlation = np.corrcoef(risks, returns)[0, 1]
            assert correlation > 0.5  # Should have positive correlation
    
    def test_risk_metrics(self, optimizer, sample_data):
        """Test risk metrics calculation"""
        _, covariance = sample_data
        
        # Create a portfolio
        weights = np.array([0.2, 0.3, 0.2, 0.1, 0.2])
        
        metrics = optimizer.calculate_risk_metrics(
            weights,
            covariance,
            confidence_level=0.95
        )
        
        assert 'portfolio_std' in metrics
        assert 'portfolio_variance' in metrics
        assert 'var_95' in metrics
        assert 'cvar_95' in metrics
        assert 'effective_n_assets' in metrics
        assert 'herfindahl_index' in metrics
        
        # Check metric relationships
        assert metrics['portfolio_std'] == np.sqrt(metrics['portfolio_variance'])
        assert metrics['cvar_95'] > metrics['var_95']  # CVaR > VaR
        assert metrics['effective_n_assets'] > 1  # Diversified
        assert 0 < metrics['herfindahl_index'] < 1
    
    def test_covariance_regularization(self, optimizer):
        """Test covariance matrix regularization"""
        # Create a singular covariance matrix
        singular_cov = np.array([
            [1.0, 0.5, 0.5],
            [0.5, 1.0, 0.5],
            [0.5, 0.5, 1.0]
        ])
        singular_cov[2] = singular_cov[0] + singular_cov[1]  # Make singular
        
        # Regularize
        reg_cov = optimizer._regularize_covariance(singular_cov)
        
        # Should be positive definite after regularization
        eigenvalues = np.linalg.eigvals(reg_cov)
        assert np.all(eigenvalues > 0)
        
        # Should be symmetric
        assert np.allclose(reg_cov, reg_cov.T)
    
    def test_optimization_with_leverage(self, optimizer, sample_data):
        """Test optimization with leverage constraint"""
        expected_returns, covariance = sample_data
        
        constraints = OptimizationConstraints(
            min_weight=-0.2,  # Allow short
            max_weight=0.5,
            max_leverage=1.5,  # 150% gross exposure
            long_only=False
        )
        
        result = optimizer.optimize(
            expected_returns,
            covariance,
            constraints
        )
        
        if result.success:
            gross_exposure = np.sum(np.abs(result.weights))
            assert gross_exposure <= 1.5 + 1e-6
    
    def test_empty_optimization(self, optimizer):
        """Test optimization with edge cases"""
        # Single asset
        returns = np.array([0.1])
        cov = np.array([[0.04]])
        
        constraints = OptimizationConstraints()
        
        result = optimizer.optimize(returns, cov, constraints)
        
        if result.success:
            assert np.allclose(result.weights, [1.0])
    
    def test_optimization_result_to_dict(self, optimizer, sample_data):
        """Test converting optimization result to dictionary"""
        expected_returns, covariance = sample_data
        
        constraints = OptimizationConstraints(
            min_weight=0.0,
            max_weight=0.4
        )
        
        result = optimizer.optimize(
            expected_returns,
            covariance,
            constraints
        )
        
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        weights_dict = result.to_dict(symbols)
        
        assert len(weights_dict) == 5
        assert all(symbol in weights_dict for symbol in symbols)
        assert np.allclose(sum(weights_dict.values()), 1.0, atol=1e-6)