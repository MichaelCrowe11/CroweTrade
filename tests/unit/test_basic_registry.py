"""Basic tests for Model Registry functionality"""

import pytest
import tempfile
import shutil
import pickle
import json
from datetime import datetime
from pathlib import Path

from src.crowetrade.models.registry import (
    ModelRegistry, ModelMetadata, ModelStatus, ModelType,
    create_model_metadata
)


class TestBasicModelRegistry:
    """Basic tests that don't require external dependencies"""
    
    def test_create_model_metadata(self):
        """Test creating model metadata"""
        metadata = create_model_metadata(
            model_id="test_model",
            name="Test Model", 
            version="1.0.0",
            model_type=ModelType.SIGNAL,
            created_by="test_user",
            description="Test model description",
            input_features=["feature1", "feature2"],
            output_schema={"signal": "float", "confidence": "float"}
        )
        
        assert metadata.model_id == "test_model"
        assert metadata.name == "Test Model"
        assert metadata.version == "1.0.0"
        assert metadata.model_type == ModelType.SIGNAL
        assert metadata.status == ModelStatus.DEVELOPMENT
        assert metadata.created_by == "test_user"
        assert isinstance(metadata.created_at, datetime)
    
    def test_model_metadata_serialization(self):
        """Test metadata serialization/deserialization"""
        metadata = create_model_metadata(
            model_id="serialization_test",
            name="Serialization Test",
            version="2.1.0",
            model_type=ModelType.PORTFOLIO,
            created_by="test_user",
            description="Testing serialization",
            input_features=["returns", "volatility"],
            output_schema={"allocation": "dict"}
        )
        
        # Serialize to dict
        data = metadata.to_dict()
        assert isinstance(data, dict)
        assert data["model_id"] == "serialization_test"
        assert data["model_type"] == "portfolio"
        assert data["status"] == "development"
        
        # Deserialize back
        restored = ModelMetadata.from_dict(data)
        assert restored.model_id == metadata.model_id
        assert restored.model_type == metadata.model_type
        assert restored.status == metadata.status
    
    def test_registry_directory_creation(self):
        """Test that registry creates required directories"""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)
            
            assert registry.registry_path.exists()
            assert registry.models_path.exists()
            assert registry.metadata_path.exists()
            assert registry.policies_path.exists()
            assert registry.staging_path.exists()
    
    def test_model_types_enum(self):
        """Test ModelType enum values"""
        assert ModelType.SIGNAL.value == "signal"
        assert ModelType.RISK.value == "risk"
        assert ModelType.EXECUTION.value == "execution"
        assert ModelType.REGIME.value == "regime"
        assert ModelType.PORTFOLIO.value == "portfolio"
    
    def test_model_status_enum(self):
        """Test ModelStatus enum values"""
        assert ModelStatus.DEVELOPMENT.value == "development"
        assert ModelStatus.TESTING.value == "testing"
        assert ModelStatus.STAGING.value == "staging"
        assert ModelStatus.PRODUCTION.value == "production"
        assert ModelStatus.DEPRECATED.value == "deprecated"
        assert ModelStatus.ARCHIVED.value == "archived"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
