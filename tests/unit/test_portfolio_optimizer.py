import math
import random
import pytest

from crowetrade.services.decision_service.optimizer import (
    PortfolioOptimizer,
    OptimizationConfig,
    OptimizationMethod,
)


@pytest.fixture
def assets():
    return ["AAPL", "MSFT", "GOOGL"]


@pytest.fixture
def expected_returns():
    return {"AAPL": 0.02, "MSFT": 0.015, "GOOGL": 0.01}


@pytest.fixture
def covariance_matrix(assets):
    # Simple positive semi-def matrix represented in dict form
    cov = {}
    base = {
        ("AAPL", "AAPL"): 0.04,
        ("MSFT", "MSFT"): 0.05,
        ("GOOGL", "GOOGL"): 0.045,
    }
    # low correlations
    for i in assets:
        for j in assets:
            if i == j:
                cov[(i, j)] = base[(i, j)]
            else:
                cov[(i, j)] = 0.01
    return cov


def build_optimizer(method: OptimizationMethod, **constraints):
    cfg = OptimizationConfig(
        method=method,
        constraints={"min_weight": constraints.get("min_weight", 0.0),
                     "max_weight": constraints.get("max_weight", 0.9)},
        risk_aversion=constraints.get("risk_aversion", 1.0),
        rebalance_threshold=constraints.get("rebalance_threshold", 0.10),
    )
    return PortfolioOptimizer(cfg, rng=random.Random(42))


@pytest.mark.unit
@pytest.mark.parametrize("method", [
    OptimizationMethod.MEAN_VARIANCE,
    OptimizationMethod.RISK_BUDGETING,
    OptimizationMethod.MAX_SHARPE,
    OptimizationMethod.MIN_VARIANCE,
])
def test_basic_optimization_runs(method, assets, expected_returns, covariance_matrix):
    opt = build_optimizer(method)
    result = opt.optimize(assets, expected_returns, covariance_matrix)
    # weights sum to 1
    assert pytest.approx(sum(result.weights.values()), rel=1e-6, abs=1e-6) == 1.0
    # weights within constraints
    for w in result.weights.values():
        assert 0.0 <= w <= 0.9
    # sharpe ratio finite
    assert math.isfinite(result.sharpe_ratio)


@pytest.mark.unit
def test_constraints_enforced(assets, expected_returns, covariance_matrix):
    opt = build_optimizer(OptimizationMethod.MEAN_VARIANCE, min_weight=0.05, max_weight=0.4)
    res = opt.optimize(assets, expected_returns, covariance_matrix)
    for w in res.weights.values():
        assert 0.05 - 1e-8 <= w <= 0.4 + 1e-8


@pytest.mark.unit
def test_rebalance_threshold(assets, expected_returns, covariance_matrix):
    opt = build_optimizer(OptimizationMethod.MEAN_VARIANCE, rebalance_threshold=0.0)  # always rebalance smoothing
    first = opt.optimize(assets, expected_returns, covariance_matrix)
    # Create a large deviation current weight forcing smoothing
    current = {k: (0.0 if i == 0 else 1.0/(len(assets)-1)) for i, k in enumerate(assets)}
    second = opt.optimize(assets, expected_returns, covariance_matrix, current_weights=current)
    # smoothing should pull weights closer to current vs first raw output
    dev_direct = sum(abs(first.weights[a] - current.get(a, 0.0)) for a in assets)
    dev_smoothed = sum(abs(second.weights[a] - current.get(a, 0.0)) for a in assets)
    assert dev_smoothed < dev_direct


@pytest.mark.unit
def test_deterministic_random_methods(assets, expected_returns, covariance_matrix):
    opt1 = build_optimizer(OptimizationMethod.MAX_SHARPE)
    opt2 = build_optimizer(OptimizationMethod.MAX_SHARPE)
    res1 = opt1.optimize(assets, expected_returns, covariance_matrix)
    res2 = opt2.optimize(assets, expected_returns, covariance_matrix)
    assert res1.weights == res2.weights

    opt3 = build_optimizer(OptimizationMethod.MIN_VARIANCE)
    opt4 = build_optimizer(OptimizationMethod.MIN_VARIANCE)
    res3 = opt3.optimize(assets, expected_returns, covariance_matrix)
    res4 = opt4.optimize(assets, expected_returns, covariance_matrix)
    assert res3.weights == res4.weights


@pytest.mark.unit
def test_risk_contributions_sum_to_risk(assets, expected_returns, covariance_matrix):
    opt = build_optimizer(OptimizationMethod.RISK_BUDGETING)
    res = opt.optimize(assets, expected_returns, covariance_matrix)
    rc = opt.risk_contributions(res.weights, covariance_matrix)
    total_contrib = sum(rc.values())
    assert total_contrib == pytest.approx(res.expected_risk, rel=1e-6)


@pytest.mark.unit
def test_empty_assets_error(expected_returns, covariance_matrix):
    opt = build_optimizer(OptimizationMethod.MEAN_VARIANCE)
    with pytest.raises(ValueError):
        opt.optimize([], expected_returns, covariance_matrix)
