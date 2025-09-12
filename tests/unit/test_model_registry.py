"""Tests for Model Registry System"""

import pytest
import tempfile
import shutil
import pickle
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from src.crowetrade.models.registry import (
    ModelRegistry, ModelMetadata, ModelArtifacts, ModelStatus, ModelType,
    create_model_metadata
)


@pytest.fixture
def temp_registry():
    """Create temporary registry for testing"""
    temp_dir = tempfile.mkdtemp()
    registry = ModelRegistry(temp_dir)
    yield registry
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_model():
    """Create sample mock model"""
    # Use mock model to avoid sklearn dependency
    class MockModel:
        def __init__(self):
            self.coef_ = np.random.randn(5)
            self.intercept_ = np.random.randn()
        
        def fit(self, X, y):
            pass
        
        def predict(self, X):
            return np.random.randn(len(X))
    
    return MockModel()


@pytest.fixture
def sample_metadata():
    """Create sample model metadata"""
    return create_model_metadata(
        model_id="test_signal_model",
        name="Test Signal Model",
        version="1.0.0",
        model_type=ModelType.SIGNAL,
        created_by="test_user",
        description="Test signal generation model",
        input_features=["feature1", "feature2", "feature3"],
        output_schema={"signal": "float", "confidence": "float"},
        framework="sklearn",
        backtest_sharpe=1.5,
        backtest_returns=0.15,
        backtest_max_dd=-0.08
    )


class TestModelRegistry:
    """Test suite for ModelRegistry"""
    
    def test_registry_initialization(self, temp_registry):
        """Test registry initialization creates proper structure"""
        assert temp_registry.registry_path.exists()
        assert temp_registry.models_path.exists()
        assert temp_registry.metadata_path.exists()
        assert temp_registry.policies_path.exists()
        assert temp_registry.staging_path.exists()
    
    def test_register_model_success(self, temp_registry, sample_model, sample_metadata):
        """Test successful model registration"""
        # Register model
        registry_id = temp_registry.register_model(
            model_object=sample_model,
            metadata=sample_metadata
        )
        
        # Verify registry ID format
        assert registry_id == "test_signal_model:1.0.0"
        
        # Verify model file exists
        model_path = temp_registry.models_path / "test_signal_model" / "1.0.0" / "model.pkl"
        assert model_path.exists()
        
        # Verify metadata file exists
        metadata_path = temp_registry.metadata_path / f"{registry_id}.json"
        assert metadata_path.exists()
        
        # Verify metadata content
        with open(metadata_path, 'r') as f:
            saved_metadata = json.load(f)
        
        assert saved_metadata['model_id'] == "test_signal_model"
        assert saved_metadata['version'] == "1.0.0"
        assert saved_metadata['model_type'] == "signal"
        assert saved_metadata['backtest_sharpe'] == 1.5
        assert 'model_checksum' in saved_metadata
    
    def test_register_model_with_policy(self, temp_registry, sample_model, sample_metadata):
        """Test model registration with policy configuration"""
        policy_config = {
            "position_size": 0.02,
            "max_leverage": 3.0,
            "stop_loss": -0.05
        }
        
        registry_id = temp_registry.register_model(
            model_object=sample_model,
            metadata=sample_metadata,
            policy_config=policy_config
        )
        
        # Verify policy file exists
        policy_path = temp_registry.models_path / "test_signal_model" / "1.0.0" / "policy.yaml"
        assert policy_path.exists()
        
        # Verify policy checksum in metadata
        metadata = temp_registry.get_metadata(registry_id)
        assert metadata.policy_checksum is not None
    
    def test_get_model_success(self, temp_registry, sample_model, sample_metadata):
        """Test successful model retrieval"""
        # Register model first
        registry_id = temp_registry.register_model(sample_model, sample_metadata)
        
        # Retrieve model
        retrieved_model, retrieved_metadata = temp_registry.get_model("test_signal_model", "1.0.0")
        
        # Verify model object
        assert hasattr(retrieved_model, 'coef_')
        assert hasattr(retrieved_model, 'intercept_')
        
        # Verify metadata
        assert retrieved_metadata.model_id == "test_signal_model"
        assert retrieved_metadata.version == "1.0.0"
        assert retrieved_metadata.model_type == ModelType.SIGNAL
    
    def test_get_model_latest_version(self, temp_registry, sample_model):
        """Test retrieving latest version of model"""
        # Register multiple versions
        for version in ["1.0.0", "1.1.0", "2.0.0"]:
            metadata = create_model_metadata(
                model_id="test_model",
                name="Test Model",
                version=version,
                model_type=ModelType.SIGNAL,
                created_by="test_user",
                description=f"Version {version}",
                input_features=["feature1"],
                output_schema={"signal": "float"}
            )
            temp_registry.register_model(sample_model, metadata)
        
        # Get latest version (no version specified)
        retrieved_model, retrieved_metadata = temp_registry.get_model("test_model")
        
        # Should return version 2.0.0 (latest)
        assert retrieved_metadata.version == "2.0.0"
    
    def test_get_model_nonexistent(self, temp_registry):
        """Test retrieving non-existent model raises error"""
        with pytest.raises(ValueError, match="Model not found"):
            temp_registry.get_model("nonexistent_model", "1.0.0")
    
    def test_model_checksum_validation(self, temp_registry, sample_model, sample_metadata):
        """Test model checksum validation"""
        # Register model
        registry_id = temp_registry.register_model(sample_model, sample_metadata)
        
        # Corrupt model file
        model_path = temp_registry.models_path / "test_signal_model" / "1.0.0" / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump("corrupted", f)
        
        # Should raise checksum mismatch error
        with pytest.raises(ValueError, match="checksum mismatch"):
            temp_registry.get_model("test_signal_model", "1.0.0")
    
    def test_list_models_filtering(self, temp_registry, sample_model):
        """Test model listing with filters"""
        # Register models of different types
        signal_metadata = create_model_metadata(
            model_id="signal_model", name="Signal Model", version="1.0.0",
            model_type=ModelType.SIGNAL, created_by="user1",
            description="Signal model", input_features=["f1"], output_schema={"s": "float"}
        )
        
        risk_metadata = create_model_metadata(
            model_id="risk_model", name="Risk Model", version="1.0.0",
            model_type=ModelType.RISK, created_by="user2",
            description="Risk model", input_features=["f1"], output_schema={"r": "float"}
        )
        
        temp_registry.register_model(sample_model, signal_metadata)
        temp_registry.register_model(sample_model, risk_metadata)
        
        # Test type filtering
        signal_models = temp_registry.list_models(model_type=ModelType.SIGNAL)
        assert len(signal_models) == 1
        assert signal_models[0].model_type == ModelType.SIGNAL
        
        # Test creator filtering
        user1_models = temp_registry.list_models(created_by="user1")
        assert len(user1_models) == 1
        assert user1_models[0].created_by == "user1"
        
        # Test status filtering
        all_models = temp_registry.list_models(status=ModelStatus.DEVELOPMENT)
        assert len(all_models) == 2  # Both models should be in development
    
    def test_promote_model_success(self, temp_registry, sample_model, sample_metadata):
        """Test successful model promotion"""
        # Register model
        registry_id = temp_registry.register_model(sample_model, sample_metadata)
        
        # Promote to testing
        success = temp_registry.promote_model(registry_id, ModelStatus.TESTING)
        assert success
        
        # Verify status change
        metadata = temp_registry.get_metadata(registry_id)
        assert metadata.status == ModelStatus.TESTING
        
        # Promote to production with approval
        success = temp_registry.promote_model(registry_id, ModelStatus.PRODUCTION, "approver")
        assert success
        
        metadata = temp_registry.get_metadata(registry_id)
        assert metadata.status == ModelStatus.PRODUCTION
        assert metadata.approved_by == "approver"
        assert metadata.approved_at is not None
    
    def test_promote_model_invalid_transition(self, temp_registry, sample_model, sample_metadata):
        """Test invalid status transition"""
        registry_id = temp_registry.register_model(sample_model, sample_metadata)
        
        # Try to promote directly from development to production (invalid)
        success = temp_registry.promote_model(registry_id, ModelStatus.PRODUCTION)
        assert not success
        
        # Status should remain unchanged
        metadata = temp_registry.get_metadata(registry_id)
        assert metadata.status == ModelStatus.DEVELOPMENT
    
    def test_delete_model_success(self, temp_registry, sample_model, sample_metadata):
        """Test successful model deletion"""
        registry_id = temp_registry.register_model(sample_model, sample_metadata)
        
        # Delete model
        success = temp_registry.delete_model(registry_id)
        assert success
        
        # Verify model is deleted
        metadata = temp_registry.get_metadata(registry_id)
        assert metadata is None
        
        # Verify files are deleted
        model_path = temp_registry.models_path / "test_signal_model" / "1.0.0"
        metadata_file = temp_registry.metadata_path / f"{registry_id}.json"
        assert not model_path.exists()
        assert not metadata_file.exists()
    
    def test_delete_production_model_protection(self, temp_registry, sample_model, sample_metadata):
        """Test protection against deleting production models"""
        registry_id = temp_registry.register_model(sample_model, sample_metadata)
        
        # Promote to production
        temp_registry.promote_model(registry_id, ModelStatus.TESTING)
        temp_registry.promote_model(registry_id, ModelStatus.STAGING)
        temp_registry.promote_model(registry_id, ModelStatus.PRODUCTION)
        
        # Try to delete without force
        success = temp_registry.delete_model(registry_id)
        assert not success
        
        # Model should still exist
        metadata = temp_registry.get_metadata(registry_id)
        assert metadata is not None
        assert metadata.status == ModelStatus.PRODUCTION
        
        # Delete with force should work
        success = temp_registry.delete_model(registry_id, force=True)
        assert success
    
    def test_version_cleanup(self, temp_registry, sample_model):
        """Test automatic cleanup of old versions"""
        # Set low retention limit
        temp_registry.max_versions_per_model = 3
        
        # Register multiple versions
        for i in range(5):
            metadata = create_model_metadata(
                model_id="cleanup_test",
                name="Cleanup Test",
                version=f"1.{i}.0",
                model_type=ModelType.SIGNAL,
                created_by="test_user",
                description=f"Version 1.{i}.0",
                input_features=["f1"],
                output_schema={"s": "float"}
            )
            temp_registry.register_model(sample_model, metadata)
        
        # Should have cleaned up old versions
        models = temp_registry.list_models()
        cleanup_models = [m for m in models if m.model_id == "cleanup_test"]
        
        # Should have 3 or fewer versions (depending on cleanup strategy)
        assert len(cleanup_models) <= 3
    
    def test_get_model_performance(self, temp_registry, sample_model, sample_metadata):
        """Test model performance retrieval"""
        registry_id = temp_registry.register_model(sample_model, sample_metadata)
        
        performance = temp_registry.get_model_performance("test_signal_model")
        
        assert performance['model_id'] == "test_signal_model"
        assert performance['latest_version'] == "1.0.0"
        assert performance['backtest_sharpe'] == 1.5
        assert performance['backtest_returns'] == 0.15
        assert performance['backtest_max_dd'] == -0.08
        assert performance['status'] == "development"
    
    def test_metadata_cache_functionality(self, temp_registry, sample_model, sample_metadata):
        """Test metadata caching works correctly"""
        registry_id = temp_registry.register_model(sample_model, sample_metadata)
        
        # First access should load from disk
        metadata1 = temp_registry.get_metadata(registry_id)
        
        # Second access should use cache
        metadata2 = temp_registry.get_metadata(registry_id)
        
        # Should be the same object (from cache)
        assert metadata1.model_id == metadata2.model_id
        assert metadata1.version == metadata2.version
        
        # Cache should contain the entry
        assert registry_id in temp_registry._metadata_cache
    
    def test_concurrent_registration_safety(self, temp_registry, sample_model):
        """Test thread safety during concurrent registrations"""
        import threading
        import time
        
        results = []
        
        def register_model(model_id_suffix):
            try:
                metadata = create_model_metadata(
                    model_id=f"concurrent_test_{model_id_suffix}",
                    name=f"Concurrent Test {model_id_suffix}",
                    version="1.0.0",
                    model_type=ModelType.SIGNAL,
                    created_by="test_user",
                    description=f"Concurrent test {model_id_suffix}",
                    input_features=["f1"],
                    output_schema={"s": "float"}
                )
                registry_id = temp_registry.register_model(sample_model, metadata)
                results.append(registry_id)
            except Exception as e:
                results.append(f"error_{model_id_suffix}: {e}")
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=register_model, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # All should succeed
        assert len(results) == 5
        assert all(not str(result).startswith("error_") for result in results)


class TestModelMetadata:
    """Test suite for ModelMetadata"""
    
    def test_metadata_serialization(self):
        """Test metadata serialization/deserialization"""
        original = create_model_metadata(
            model_id="test_model",
            name="Test Model",
            version="1.0.0",
            model_type=ModelType.SIGNAL,
            created_by="test_user",
            description="Test model",
            input_features=["f1", "f2"],
            output_schema={"signal": "float"},
            framework="sklearn"
        )
        
        # Serialize to dict
        data = original.to_dict()
        assert isinstance(data, dict)
        assert data['model_type'] == "signal"
        assert data['status'] == "development"
        
        # Deserialize back
        restored = ModelMetadata.from_dict(data)
        
        # Should be equivalent
        assert restored.model_id == original.model_id
        assert restored.version == original.version
        assert restored.model_type == original.model_type
        assert restored.status == original.status
        assert restored.created_by == original.created_by
    
    def test_metadata_validation(self):
        """Test metadata field validation"""
        # Valid metadata should work
        metadata = create_model_metadata(
            model_id="valid_model",
            name="Valid Model",
            version="1.0.0",
            model_type=ModelType.SIGNAL,
            created_by="user",
            description="Valid description",
            input_features=["feature1"],
            output_schema={"output": "float"}
        )
        
        assert metadata.model_id == "valid_model"
        assert isinstance(metadata.created_at, datetime)
        assert metadata.status == ModelStatus.DEVELOPMENT
    
    def test_enum_conversions(self):
        """Test enum conversions in serialization"""
        metadata = create_model_metadata(
            model_id="enum_test",
            name="Enum Test",
            version="1.0.0",
            model_type=ModelType.PORTFOLIO,
            created_by="user",
            description="Testing enum conversions",
            input_features=["f1"],
            output_schema={"output": "float"}
        )
        
        # Serialize
        data = metadata.to_dict()
        assert data['model_type'] == "portfolio"
        assert data['status'] == "development"
        
        # Deserialize
        restored = ModelMetadata.from_dict(data)
        assert restored.model_type == ModelType.PORTFOLIO
        assert restored.status == ModelStatus.DEVELOPMENT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
