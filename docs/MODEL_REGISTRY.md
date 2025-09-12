# Model Registry and A/B Testing System Documentation

## Overview

The Model Registry and A/B Testing system provides comprehensive model lifecycle management and statistical comparison capabilities for the CroweTrade platform.

## Model Registry

### Features
- **Centralized Model Storage**: Version-controlled artifact storage with checksums
- **Model Metadata Management**: Rich metadata including performance metrics and governance info
- **Lifecycle Management**: Status progression from development to production
- **Integrity Verification**: SHA-256 checksums and optional signing
- **Policy Configuration**: Trading policy storage alongside models
- **Performance Tracking**: Backtest metrics and validation scores

### Core Components

#### ModelRegistry Class
```python
from crowetrade.models import ModelRegistry, create_model_metadata, ModelType

# Initialize registry
registry = ModelRegistry("/path/to/registry")

# Create model metadata
metadata = create_model_metadata(
    model_id="signal_v1",
    name="Signal Generation Model v1", 
    version="1.0.0",
    model_type=ModelType.SIGNAL,
    created_by="data_scientist",
    description="LSTM-based signal generation",
    input_features=["returns", "volume", "volatility"],
    output_schema={"signal": "float", "confidence": "float"},
    framework="tensorflow",
    backtest_sharpe=1.8,
    backtest_returns=0.25
)

# Register model
registry_id = registry.register_model(
    model_object=trained_model,
    metadata=metadata,
    policy_config={"position_size": 0.02, "max_leverage": 3.0}
)

# Load model for inference
model, metadata = registry.get_model("signal_v1")  # Latest version
model, metadata = registry.get_model("signal_v1", "1.0.0")  # Specific version
```

#### Model Status Lifecycle
```
DEVELOPMENT → TESTING → STAGING → PRODUCTION
     ↓           ↓         ↓           ↓
  DEPRECATED ← DEPRECATED ← DEPRECATED ← DEPRECATED → ARCHIVED
```

### Model Promotion
```python
# Promote model through lifecycle
registry.promote_model("signal_v1:1.0.0", ModelStatus.TESTING)
registry.promote_model("signal_v1:1.0.0", ModelStatus.STAGING) 
registry.promote_model("signal_v1:1.0.0", ModelStatus.PRODUCTION, approved_by="lead_quant")
```

## A/B Testing System

### Features
- **Multi-Armed Bandit Allocation**: Thompson Sampling, UCB, Epsilon-Greedy
- **Statistical Significance Testing**: Automated hypothesis testing
- **Performance Monitoring**: Real-time metrics and confidence intervals
- **Early Stopping**: Automatic termination on significant results
- **Risk Controls**: Stop-loss thresholds and allocation limits

### Core Components

#### ABTestEngine Class
```python
from crowetrade.models import ABTestEngine, create_ab_test, AllocationStrategy

# Initialize A/B testing engine
ab_engine = ABTestEngine(model_registry, "/path/to/ab_tests")

# Create A/B test
test_config = create_ab_test(
    test_id="signal_comparison_001",
    name="LSTM vs Transformer Signal Models",
    description="Comparing LSTM and Transformer architectures",
    model_registry_ids=["signal_lstm:1.0.0", "signal_transformer:1.0.0"],
    created_by="research_team",
    allocation_strategy=AllocationStrategy.THOMPSON_SAMPLING,
    minimum_sample_size=1000,
    max_duration_days=30
)

test_id = ab_engine.create_test(test_config)
```

#### Traffic Allocation
```python
# Get model allocation for request
model_id = ab_engine.get_model_allocation(test_id)

# Record result
ab_engine.record_result(
    test_id=test_id,
    model_registry_id=model_id, 
    return_value=0.025,  # 2.5% return
    additional_metrics={"latency_ms": 15, "confidence": 0.85}
)
```

#### Results Analysis
```python
# Get comprehensive test results
results = ab_engine.get_test_results(test_id)
print(f"Test Status: {results['status']}")
print(f"Total Requests: {results['total_requests']}")

for arm in results['arms']:
    print(f"Arm {arm['arm_id']}: {arm['mean_return']:.4f} ± {arm['std_return']:.4f}")
    print(f"  Sharpe Ratio: {arm['sharpe_ratio']:.2f}")
    print(f"  Win Rate: {arm['win_rate']:.2%}")

# Statistical significance (for 2-arm tests)
if 'significance_test' in results:
    sig = results['significance_test']
    print(f"Significant: {sig['significant']}")
    print(f"P-value: {sig['p_value']:.4f}")
    print(f"Winner: {sig['winner']}")
```

## Allocation Strategies

### Thompson Sampling (Recommended)
- Bayesian approach using Beta distributions
- Naturally balances exploration vs exploitation
- Converges to optimal allocation over time

### Upper Confidence Bound (UCB)
- Optimistic approach based on confidence intervals
- Good for scenarios requiring rapid exploration
- Theoretical optimality guarantees

### Epsilon-Greedy
- Simple exploration with fixed exploration rate
- Predictable behavior but potentially suboptimal
- Good baseline for comparison

### Fixed Allocation
- Static percentage allocation
- No learning or adaptation
- Useful for controlled experiments

## Integration Examples

### Portfolio Optimization Comparison
```python
# Compare different portfolio optimization strategies
test_config = create_ab_test(
    test_id="portfolio_optimization_comparison",
    name="Mean-Variance vs Risk Parity",
    description="Comparing portfolio optimization methods",
    model_registry_ids=[
        "portfolio_mvo:2.0.0",
        "portfolio_risk_parity:1.5.0"
    ],
    created_by="portfolio_team",
    allocation_strategy=AllocationStrategy.UCB,
    minimum_sample_size=500
)
```

### Regime Detection Evaluation
```python
# Test regime detection models
test_config = create_ab_test(
    test_id="regime_detection_hmm_vs_lstm", 
    name="HMM vs LSTM Regime Detection",
    description="Statistical vs deep learning regime detection",
    model_registry_ids=[
        "regime_hmm:3.1.0",
        "regime_lstm:1.0.0"
    ],
    created_by="research_team",
    allocation_strategy=AllocationStrategy.THOMPSON_SAMPLING
)
```

## Governance and Compliance

### Model Approval Workflow
1. **Development**: Initial model development and testing
2. **Testing**: Validation against historical data
3. **Staging**: Limited production testing
4. **Production**: Full production deployment with approval

### Audit Trail
- All model operations are logged
- Checksums verify model integrity
- Approval records track governance
- Performance metrics enable monitoring

### Risk Controls
- Stop-loss thresholds prevent runaway losses
- Allocation limits ensure diversification
- Early stopping protects against prolonged poor performance
- Maximum test duration prevents indefinite experiments

## Performance Monitoring

### Model Metrics
- Backtest performance (Sharpe, returns, drawdown)
- Live performance tracking
- Latency and resource usage
- Model drift detection

### A/B Test Metrics
- Statistical significance over time
- Confidence intervals
- Effect sizes
- Power analysis

## Best Practices

### Model Registry
1. Use semantic versioning (e.g., 1.2.3)
2. Include comprehensive metadata
3. Store policy configurations with models
4. Regularly clean up old versions
5. Use checksums for integrity verification

### A/B Testing
1. Define clear success metrics before starting
2. Ensure sufficient sample sizes for statistical power
3. Monitor for early stopping opportunities
4. Use appropriate allocation strategies for your use case
5. Document test hypotheses and results

### Integration
1. Automate model deployment pipelines
2. Implement gradual rollouts for new models
3. Monitor model performance continuously
4. Establish clear escalation procedures for issues
5. Regular model retraining and validation

## Technical Architecture

### Storage Structure
```
registry/
├── models/
│   ├── signal_v1/
│   │   ├── 1.0.0/
│   │   │   ├── model.pkl
│   │   │   ├── policy.yaml
│   │   │   └── features.json
│   │   └── 1.1.0/
│   └── portfolio_mvo/
├── metadata/
│   ├── signal_v1:1.0.0.json
│   └── portfolio_mvo:2.0.0.json
└── policies/
    └── trading_policies.yaml

ab_tests/
├── tests/
│   ├── test_001.json
│   └── test_002.json
└── results/
    ├── test_001_final.json
    └── test_002_final.json
```

### Dependencies
- **Core**: numpy, pathlib, json, pickle
- **Statistics**: scipy.stats (for significance testing)
- **Optional**: yaml (for policy configuration)
- **Testing**: pytest, tempfile, unittest.mock

This system provides a production-ready foundation for model lifecycle management and statistical model comparison in algorithmic trading environments.
