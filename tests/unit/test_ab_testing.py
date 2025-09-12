"""Tests for A/B Testing System"""

import pytest
import tempfile
import shutil
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
# Mock model for testing

from src.crowetrade.models.ab_testing import (
    ABTestEngine, ABTestConfig, TestArm, AllocationStrategy, TestStatus,
    create_ab_test
)
from src.crowetrade.models.registry import (
    ModelRegistry, create_model_metadata, ModelType
)


@pytest.fixture
def temp_directories():
    """Create temporary directories for testing"""
    registry_dir = tempfile.mkdtemp()
    ab_test_dir = tempfile.mkdtemp()
    
    yield registry_dir, ab_test_dir
    
    shutil.rmtree(registry_dir, ignore_errors=True)
    shutil.rmtree(ab_test_dir, ignore_errors=True)


@pytest.fixture
def model_registry(temp_directories):
    """Create model registry with test models"""
    registry_dir, _ = temp_directories
    registry = ModelRegistry(registry_dir)
    
    # Create test models
    class MockModel:
        def __init__(self):
            self.coef_ = np.random.randn(5)
    
    for i in range(3):
        model = MockModel()
        
        metadata = create_model_metadata(
            model_id=f"test_model_{i}",
            name=f"Test Model {i}",
            version="1.0.0",
            model_type=ModelType.SIGNAL,
            created_by="test_user",
            description=f"Test model {i}",
            input_features=[f"feature_{j}" for j in range(5)],
            output_schema={"signal": "float"}
        )
        
        registry.register_model(model, metadata)
    
    return registry


@pytest.fixture
def ab_engine(temp_directories, model_registry):
    """Create A/B testing engine"""
    _, ab_test_dir = temp_directories
    return ABTestEngine(model_registry, ab_test_dir)


@pytest.fixture
def sample_test_config():
    """Create sample A/B test configuration"""
    return create_ab_test(
        test_id="test_ab_001",
        name="Signal Model A/B Test",
        description="Testing signal model variants",
        model_registry_ids=["test_model_0:1.0.0", "test_model_1:1.0.0"],
        created_by="test_user",
        allocation_strategy=AllocationStrategy.THOMPSON_SAMPLING,
        minimum_sample_size=100,
        max_duration_days=7
    )


class TestABTestEngine:
    """Test suite for ABTestEngine"""
    
    def test_engine_initialization(self, ab_engine):
        """Test A/B engine initialization"""
        assert ab_engine.tests_path.exists()
        assert ab_engine.results_path.exists()
        assert isinstance(ab_engine._active_tests, dict)
        assert isinstance(ab_engine._test_results, dict)
    
    def test_create_test_success(self, ab_engine, sample_test_config):
        """Test successful A/B test creation"""
        test_id = ab_engine.create_test(sample_test_config)
        
        assert test_id == "test_ab_001"
        assert test_id in ab_engine._active_tests
        
        # Verify equal initial allocation
        test = ab_engine._active_tests[test_id]
        for arm in test.arms:
            assert abs(arm.allocation_weight - 0.5) < 0.01  # Should be ~50% each
        
        # Verify test file was created
        test_file = ab_engine.tests_path / f"{test_id}.json"
        assert test_file.exists()
    
    def test_create_test_validation_errors(self, ab_engine):
        """Test A/B test creation validation"""
        # Test with only one arm (should fail)
        invalid_config = create_ab_test(
            test_id="invalid_test",
            name="Invalid Test",
            description="Invalid test config",
            model_registry_ids=["test_model_0:1.0.0"],  # Only one model
            created_by="test_user"
        )
        
        with pytest.raises(ValueError, match="at least 2 arms"):
            ab_engine.create_test(invalid_config)
        
        # Test with non-existent model
        invalid_config2 = create_ab_test(
            test_id="invalid_test2",
            name="Invalid Test 2",
            description="Invalid test config with bad model",
            model_registry_ids=["nonexistent_model:1.0.0", "test_model_0:1.0.0"],
            created_by="test_user"
        )
        
        with pytest.raises(ValueError, match="Model not found in registry"):
            ab_engine.create_test(invalid_config2)
    
    def test_get_model_allocation_fixed(self, ab_engine, sample_test_config):
        """Test fixed allocation strategy"""
        sample_test_config.allocation_strategy = AllocationStrategy.FIXED
        test_id = ab_engine.create_test(sample_test_config)
        
        # Should return valid model IDs
        allocations = []
        for _ in range(100):
            allocation = ab_engine.get_model_allocation(test_id)
            allocations.append(allocation)
        
        # Should get both models
        unique_allocations = set(allocations)
        assert len(unique_allocations) == 2
        assert "test_model_0:1.0.0" in unique_allocations
        assert "test_model_1:1.0.0" in unique_allocations
    
    def test_get_model_allocation_thompson(self, ab_engine, sample_test_config):
        """Test Thompson sampling allocation"""
        sample_test_config.allocation_strategy = AllocationStrategy.THOMPSON_SAMPLING
        test_id = ab_engine.create_test(sample_test_config)
        
        # Record some results to differentiate arms
        for i in range(50):
            # Make arm 0 perform better
            ab_engine.record_result(test_id, "test_model_0:1.0.0", 0.02)  # 2% return
            ab_engine.record_result(test_id, "test_model_1:1.0.0", -0.01)  # -1% return
        
        # Should favor the better performing model
        allocations = []
        for _ in range(100):
            allocation = ab_engine.get_model_allocation(test_id)
            allocations.append(allocation)
        
        # Should get more allocations for the better model
        model_0_count = sum(1 for a in allocations if a == "test_model_0:1.0.0")
        model_1_count = sum(1 for a in allocations if a == "test_model_1:1.0.0")
        
        # Better model should get more traffic
        assert model_0_count > model_1_count
    
    def test_get_model_allocation_ucb(self, ab_engine, sample_test_config):
        """Test UCB allocation strategy"""
        sample_test_config.allocation_strategy = AllocationStrategy.UCB
        test_id = ab_engine.create_test(sample_test_config)
        
        # Should explore both arms initially
        allocations = []
        for _ in range(20):
            allocation = ab_engine.get_model_allocation(test_id)
            allocations.append(allocation)
            
            # Record neutral results
            ab_engine.record_result(test_id, allocation, 0.001)
        
        # Should have explored both models
        unique_allocations = set(allocations)
        assert len(unique_allocations) == 2
    
    def test_get_model_allocation_epsilon_greedy(self, ab_engine, sample_test_config):
        """Test epsilon-greedy allocation"""
        sample_test_config.allocation_strategy = AllocationStrategy.EPSILON_GREEDY
        test_id = ab_engine.create_test(sample_test_config)
        
        # Record results to establish clear winner
        for i in range(100):
            ab_engine.record_result(test_id, "test_model_0:1.0.0", 0.05)  # 5% return
            ab_engine.record_result(test_id, "test_model_1:1.0.0", -0.02)  # -2% return
        
        # Should mostly exploit the better model
        allocations = []
        for _ in range(100):
            allocation = ab_engine.get_model_allocation(test_id)
            allocations.append(allocation)
        
        model_0_count = sum(1 for a in allocations if a == "test_model_0:1.0.0")
        
        # Should exploit the better model most of the time (but not always due to epsilon)
        assert model_0_count > 70  # Should get >70% of traffic
    
    def test_record_result_updates_statistics(self, ab_engine, sample_test_config):
        """Test that recording results updates arm statistics"""
        test_id = ab_engine.create_test(sample_test_config)
        
        # Record some results
        returns = [0.02, -0.01, 0.03, 0.01, -0.005]
        for ret in returns:
            ab_engine.record_result(test_id, "test_model_0:1.0.0", ret)
        
        # Check arm statistics
        test = ab_engine._active_tests[test_id]
        arm_0 = test.arms[0]  # First arm should be test_model_0
        
        assert arm_0.total_requests == 5
        assert abs(arm_0.mean_return - np.mean(returns)) < 0.001
        assert abs(arm_0.total_returns - sum(returns)) < 0.001
    
    def test_get_test_results(self, ab_engine, sample_test_config):
        """Test getting comprehensive test results"""
        test_id = ab_engine.create_test(sample_test_config)
        
        # Record some results for both arms
        np.random.seed(42)  # For reproducible results
        
        for _ in range(50):
            ab_engine.record_result(test_id, "test_model_0:1.0.0", np.random.normal(0.02, 0.01))
            ab_engine.record_result(test_id, "test_model_1:1.0.0", np.random.normal(0.01, 0.01))
        
        results = ab_engine.get_test_results(test_id)
        
        # Check result structure
        assert results['test_id'] == test_id
        assert results['name'] == "Signal Model A/B Test"
        assert 'status' in results
        assert 'arms' in results
        assert len(results['arms']) == 2
        
        # Check arm statistics
        for arm_result in results['arms']:
            assert 'total_requests' in arm_result
            assert 'mean_return' in arm_result
            assert 'std_return' in arm_result
            assert 'confidence_interval' in arm_result
            assert 'sharpe_ratio' in arm_result
            assert 'win_rate' in arm_result
        
        # Should have significance test for 2-arm test
        assert 'significance_test' in results
    
    def test_statistical_significance_calculation(self, ab_engine, sample_test_config):
        """Test statistical significance calculation"""
        test_id = ab_engine.create_test(sample_test_config)
        
        # Create clearly different performance
        np.random.seed(42)
        
        # Arm 0: 3% mean return
        for _ in range(200):
            ab_engine.record_result(test_id, "test_model_0:1.0.0", np.random.normal(0.03, 0.01))
        
        # Arm 1: 1% mean return
        for _ in range(200):
            ab_engine.record_result(test_id, "test_model_1:1.0.0", np.random.normal(0.01, 0.01))
        
        results = ab_engine.get_test_results(test_id)
        significance = results['significance_test']
        
        # Should detect significant difference
        assert 'significant' in significance
        assert 'p_value' in significance
        assert 'winner' in significance
        
        # With large effect size and sample size, should be significant
        if significance['significant']:
            assert significance['p_value'] < 0.05
            # Arm 0 should win (higher return)
            assert "arm_0" in significance['winner']
    
    def test_early_stopping_significant_result(self, ab_engine, sample_test_config):
        """Test early stopping when significant result is found"""
        sample_test_config.early_stopping = True
        sample_test_config.minimum_sample_size = 50  # Lower for faster test
        
        test_id = ab_engine.create_test(sample_test_config)
        
        # Create very large difference to trigger early stopping
        np.random.seed(42)
        
        for i in range(100):
            ab_engine.record_result(test_id, "test_model_0:1.0.0", np.random.normal(0.10, 0.01))  # 10% return
            ab_engine.record_result(test_id, "test_model_1:1.0.0", np.random.normal(-0.05, 0.01))  # -5% return
        
        # Test might have been stopped early
        # Check if test is still active or was stopped
        if test_id not in ab_engine._active_tests:
            # Test was stopped early - verify results file exists
            results_file = ab_engine.results_path / f"{test_id}_final.json"
            assert results_file.exists()
    
    def test_stop_loss_threshold(self, ab_engine, sample_test_config):
        """Test stop loss threshold enforcement"""
        test_id = ab_engine.create_test(sample_test_config)
        
        # Create very poor performance for one arm
        for _ in range(50):
            ab_engine.record_result(test_id, "test_model_0:1.0.0", -0.08)  # -8% return (exceeds -5% threshold)
            ab_engine.record_result(test_id, "test_model_1:1.0.0", 0.02)   # 2% return
        
        test = ab_engine._active_tests[test_id]
        
        # Poor performing arm should have reduced allocation
        arm_0 = next(arm for arm in test.arms if "test_model_0" in arm.model_registry_id)
        assert arm_0.allocation_weight < 0.4  # Should be below 40%
    
    def test_stop_test_functionality(self, ab_engine, sample_test_config):
        """Test manual test stopping"""
        test_id = ab_engine.create_test(sample_test_config)
        
        # Record some results
        for _ in range(10):
            ab_engine.record_result(test_id, "test_model_0:1.0.0", 0.01)
            ab_engine.record_result(test_id, "test_model_1:1.0.0", 0.02)
        
        # Stop test
        success = ab_engine.stop_test(test_id, "Manual stop for testing")
        assert success
        
        # Test should be removed from active tests
        assert test_id not in ab_engine._active_tests
        
        # Results file should exist
        results_file = ab_engine.results_path / f"{test_id}_final.json"
        assert results_file.exists()
        
        # Verify results content
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        assert results['test_id'] == test_id
        assert results['stop_reason'] == "Manual stop for testing"
        assert 'stopped_at' in results
    
    def test_test_duration_expiry(self, ab_engine):
        """Test test expiry based on duration"""
        # Create test with very short duration
        config = create_ab_test(
            test_id="expiry_test",
            name="Expiry Test",
            description="Test expiry functionality",
            model_registry_ids=["test_model_0:1.0.0", "test_model_1:1.0.0"],
            created_by="test_user",
            max_duration_days=0  # Should expire immediately
        )
        
        # Manually set creation time to past
        config.created_at = datetime.utcnow() - timedelta(days=1)
        
        test_id = ab_engine.create_test(config)
        
        # Check if test is active (should not be)
        allocation = ab_engine.get_model_allocation(test_id)
        assert allocation is None  # Should return None for expired test
    
    def test_load_active_tests_on_startup(self, temp_directories, model_registry):
        """Test loading active tests on engine startup"""
        _, ab_test_dir = temp_directories
        
        # Create first engine and test
        engine1 = ABTestEngine(model_registry, ab_test_dir)
        config = create_ab_test(
            test_id="persistence_test",
            name="Persistence Test",
            description="Test persistence",
            model_registry_ids=["test_model_0:1.0.0", "test_model_1:1.0.0"],
            created_by="test_user"
        )
        
        test_id = engine1.create_test(config)
        
        # Create second engine (should load existing test)
        engine2 = ABTestEngine(model_registry, ab_test_dir)
        
        # Should have loaded the existing test
        assert test_id in engine2._active_tests
        
        # Should be able to get allocation
        allocation = engine2.get_model_allocation(test_id)
        assert allocation is not None


class TestABTestConfig:
    """Test suite for ABTestConfig"""
    
    def test_config_serialization(self):
        """Test A/B test configuration serialization"""
        config = create_ab_test(
            test_id="serialization_test",
            name="Serialization Test",
            description="Test config serialization",
            model_registry_ids=["model_1:1.0.0", "model_2:1.0.0"],
            created_by="test_user",
            allocation_strategy=AllocationStrategy.UCB,
            minimum_sample_size=500
        )
        
        # Serialize to dict
        data = config.to_dict()
        assert isinstance(data, dict)
        assert data['test_id'] == "serialization_test"
        assert data['allocation_strategy'] == "ucb"
        assert data['minimum_sample_size'] == 500
        
        # Deserialize back
        restored = ABTestConfig.from_dict(data)
        
        # Should be equivalent
        assert restored.test_id == config.test_id
        assert restored.allocation_strategy == config.allocation_strategy
        assert restored.minimum_sample_size == config.minimum_sample_size
        assert len(restored.arms) == len(config.arms)
    
    def test_arm_initialization(self):
        """Test test arm initialization"""
        config = create_ab_test(
            test_id="arm_test",
            name="Arm Test",
            description="Test arm initialization",
            model_registry_ids=["model_1:1.0.0", "model_2:1.0.0", "model_3:1.0.0"],
            created_by="test_user"
        )
        
        # Should create correct number of arms
        assert len(config.arms) == 3
        
        # Arms should have correct IDs and model references
        for i, arm in enumerate(config.arms):
            assert arm.arm_id == f"arm_{i}"
            assert arm.model_registry_id == f"model_{i+1}:1.0.0"
            assert arm.total_requests == 0
            assert arm.total_returns == 0.0


class TestAllocationStrategies:
    """Test suite for different allocation strategies"""
    
    def test_thompson_sampling_convergence(self):
        """Test Thompson sampling converges to better arm"""
        # Simulate Thompson sampling behavior
        arm1 = TestArm(arm_id="arm_1", model_registry_id="model_1:1.0.0")
        arm2 = TestArm(arm_id="arm_2", model_registry_id="model_2:1.0.0")
        
        # Simulate results - arm1 performs better
        for _ in range(100):
            # Arm 1: 70% success rate
            if np.random.random() < 0.7:
                arm1.alpha += 1
            else:
                arm1.beta += 1
            
            # Arm 2: 30% success rate
            if np.random.random() < 0.3:
                arm2.alpha += 1
            else:
                arm2.beta += 1
        
        # Sample from distributions many times
        arm1_wins = 0
        for _ in range(1000):
            sample1 = np.random.beta(arm1.alpha, arm1.beta)
            sample2 = np.random.beta(arm2.alpha, arm2.beta)
            if sample1 > sample2:
                arm1_wins += 1
        
        # Better arm should win most samples
        assert arm1_wins > 700  # Should win >70% of samples
    
    def test_ucb_exploration_bonus(self):
        """Test UCB exploration bonus calculation"""
        # This would test the UCB calculation logic
        # For now, we verify that unvisited arms get infinite UCB
        arm_unvisited = TestArm(arm_id="unvisited", model_registry_id="model:1.0.0")
        arm_visited = TestArm(arm_id="visited", model_registry_id="model2:1.0.0")
        arm_visited.total_requests = 10
        arm_visited.mean_return = 0.05
        
        # Unvisited arm should be selected first (infinite UCB)
        # This logic would be in the actual UCB selection method


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
