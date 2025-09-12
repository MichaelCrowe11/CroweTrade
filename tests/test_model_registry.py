"""Tests for ML model registry"""
import pytest
import tempfile
import shutil
import numpy as np
from datetime import datetime
from unittest.mock import Mock

from crowetrade.models.registry import (
    ModelRegistry,
    ModelMetadata,
    ModelVersion,
    ModelStatus,
    ModelMetrics
)


class MockMLModel:
    """Mock ML model for testing"""
    
    def __init__(self, name="test_model"):
        self.name = name
        self.params = {'param1': 1, 'param2': 'value'}
        self.fitted = False
    
    def predict(self, X):
        return np.random.random((len(X),))
    
    def fit(self, X, y):
        self.fitted = True
        return self
    
    def get_params(self):
        return self.params.copy()
    
    def set_params(self, **params):
        self.params.update(params)


class TestModelRegistry:
    """Test model registry functionality"""
    
    @pytest.fixture
    def temp_registry(self):
        """Create temporary registry directory"""
        temp_dir = tempfile.mkdtemp()
        registry = ModelRegistry(
            registry_path=temp_dir,
            enable_validation=False,  # Skip validation for tests
            auto_backup=False
        )
        yield registry
        shutil.rmtree(temp_dir)
    
    def test_registry_initialization(self, temp_registry):
        """Test registry initialization"""
        
        assert temp_registry.registry_path.exists()
        assert (temp_registry.registry_path / "models").exists()
        assert (temp_registry.registry_path / "metadata").exists()
        assert (temp_registry.registry_path / "backups").exists()
    
    def test_model_registration(self, temp_registry):
        """Test model registration"""
        
        model = MockMLModel("test_classifier")
        
        metrics = ModelMetrics(
            accuracy=0.95,
            precision=0.92,
            recall=0.88,
            f1_score=0.90
        )
        
        model_path = temp_registry.register_model(
            name="test_classifier",
            model=model,
            version="1.0.0",
            description="Test classification model",
            model_type="classifier",
            framework="sklearn",
            created_by="test_user",
            metrics=metrics,
            tags=["test", "classifier"],
            dependencies=["scikit-learn>=1.0"]
        )
        
        assert model_path is not None
        assert "test_classifier" in temp_registry._models
        
        # Check metadata
        metadata = temp_registry.get_model_info("test_classifier")
        assert metadata.name == "test_classifier"
        assert metadata.model_type == "classifier"
        assert metadata.framework == "sklearn"
        assert metadata.latest_version == "1.0.0"
        assert "1.0.0" in metadata.versions
        
        # Check version info
        version_info = metadata.versions["1.0.0"]
        assert version_info.description == "Test classification model"
        assert version_info.status == ModelStatus.DEVELOPMENT
        assert version_info.metrics.accuracy == 0.95
        assert "test" in version_info.tags
    
    def test_model_loading(self, temp_registry):
        """Test model loading"""
        
        model = MockMLModel("loadable_model")
        
        temp_registry.register_model(
            name="loadable_model",
            model=model,
            version="1.0.0",
            description="Model for loading test",
            model_type="regressor",
            framework="sklearn",
            created_by="test_user"
        )
        
        # Load by version
        loaded_model = temp_registry.load_model("loadable_model", "1.0.0")
        assert loaded_model is not None
        assert loaded_model.name == "loadable_model"
        
        # Load latest version
        loaded_model_latest = temp_registry.load_model("loadable_model")
        assert loaded_model_latest is not None
        
        # Check usage stats updated
        metadata = temp_registry.get_model_info("loadable_model")
        assert metadata.usage_count > 0
        assert metadata.last_used is not None
    
    def test_model_versioning(self, temp_registry):
        """Test model versioning"""
        
        model_v1 = MockMLModel("versioned_model")
        model_v2 = MockMLModel("versioned_model_v2")
        
        # Register version 1.0.0
        temp_registry.register_model(
            name="versioned_model",
            model=model_v1,
            version="1.0.0",
            description="Version 1",
            model_type="classifier",
            framework="sklearn",
            created_by="test_user"
        )
        
        # Register version 1.1.0
        temp_registry.register_model(
            name="versioned_model",
            model=model_v2,
            version="1.1.0",
            description="Version 1.1 with improvements",
            model_type="classifier",
            framework="sklearn",
            created_by="test_user"
        )
        
        metadata = temp_registry.get_model_info("versioned_model")
        assert len(metadata.versions) == 2
        assert metadata.latest_version == "1.1.0"
        
        versions = temp_registry.get_model_versions("versioned_model")
        assert "1.0.0" in versions
        assert "1.1.0" in versions
        
        # Load specific versions
        model_v1_loaded = temp_registry.load_model("versioned_model", "1.0.0")
        model_v2_loaded = temp_registry.load_model("versioned_model", "1.1.0")
        
        assert model_v1_loaded.name == "versioned_model"
        assert model_v2_loaded.name == "versioned_model_v2"
    
    def test_model_deployment(self, temp_registry):
        """Test model deployment status changes"""
        
        model = MockMLModel("deployment_model")
        
        temp_registry.register_model(
            name="deployment_model",
            model=model,
            version="1.0.0",
            description="Model for deployment test",
            model_type="classifier",
            framework="sklearn",
            created_by="test_user"
        )
        
        # Deploy to staging
        success = temp_registry.deploy_model(
            "deployment_model", "1.0.0", ModelStatus.STAGING
        )
        assert success
        
        metadata = temp_registry.get_model_info("deployment_model")
        assert metadata.versions["1.0.0"].status == ModelStatus.STAGING
        
        # Deploy to production
        success = temp_registry.deploy_model(
            "deployment_model", "1.0.0", ModelStatus.PRODUCTION
        )
        assert success
        
        metadata = temp_registry.get_model_info("deployment_model")
        assert metadata.versions["1.0.0"].status == ModelStatus.PRODUCTION
        assert metadata.production_version == "1.0.0"
        
        # Load production model
        prod_model = temp_registry.load_model(
            "deployment_model", status=ModelStatus.PRODUCTION
        )
        assert prod_model is not None
    
    def test_model_listing(self, temp_registry):
        """Test model listing with filters"""
        
        # Register different types of models
        models = [
            ("classifier_1", "classifier", "sklearn"),
            ("classifier_2", "classifier", "xgboost"),
            ("regressor_1", "regressor", "sklearn"),
            ("regressor_2", "regressor", "lightgbm")
        ]
        
        for name, model_type, framework in models:
            model = MockMLModel(name)
            temp_registry.register_model(
                name=name,
                model=model,
                version="1.0.0",
                description=f"{model_type} model",
                model_type=model_type,
                framework=framework,
                created_by="test_user"
            )
        
        # List all models
        all_models = temp_registry.list_models()
        assert len(all_models) == 4
        
        # List by type
        classifiers = temp_registry.list_models(model_type="classifier")
        assert len(classifiers) == 2
        assert "classifier_1" in classifiers
        assert "classifier_2" in classifiers
        
        # List by framework
        sklearn_models = temp_registry.list_models(framework="sklearn")
        assert len(sklearn_models) == 2
        assert "classifier_1" in sklearn_models
        assert "regressor_1" in sklearn_models
    
    def test_model_validation(self, temp_registry):
        """Test model validation"""
        
        model = MockMLModel("validation_model")
        
        temp_registry.register_model(
            name="validation_model",
            model=model,
            version="1.0.0",
            description="Model for validation test",
            model_type="classifier",
            framework="sklearn",
            created_by="test_user"
        )
        
        # Mock validation function
        def validation_func(model, validation_data):
            # Simulate validation
            predictions = model.predict(validation_data)
            return {
                'accuracy': 0.92,
                'samples_tested': len(validation_data),
                'passed': True
            }
        
        # Mock validation data
        validation_data = np.random.random((100, 10))
        
        success = temp_registry.validate_model(
            "validation_model", "1.0.0", validation_data, validation_func
        )
        
        assert success
        
        metadata = temp_registry.get_model_info("validation_model")
        version_info = metadata.versions["1.0.0"]
        assert version_info.is_validated
        assert version_info.validation_results['accuracy'] == 0.92
        assert version_info.validation_results['passed'] is True
    
    def test_model_comparison(self, temp_registry):
        """Test model comparison"""
        
        # Register two models with different metrics
        model1 = MockMLModel("model_1")
        model2 = MockMLModel("model_2")
        
        metrics1 = ModelMetrics(accuracy=0.90, precision=0.85)
        metrics2 = ModelMetrics(accuracy=0.95, precision=0.90)
        
        temp_registry.register_model(
            name="model_1", model=model1, version="1.0.0",
            description="First model", model_type="classifier",
            framework="sklearn", created_by="test_user", metrics=metrics1
        )
        
        temp_registry.register_model(
            name="model_2", model=model2, version="1.0.0",
            description="Second model", model_type="classifier",
            framework="sklearn", created_by="test_user", metrics=metrics2
        )
        
        comparison = temp_registry.compare_models(
            "model_1", "1.0.0", "model_2", "1.0.0"
        )
        
        assert 'model1' in comparison
        assert 'model2' in comparison
        assert 'differences' in comparison
        
        # Check accuracy difference
        acc_diff = comparison['differences']['accuracy']
        assert acc_diff['model1'] == 0.90
        assert acc_diff['model2'] == 0.95
        assert acc_diff['difference'] == 0.05
    
    def test_model_deletion(self, temp_registry):
        """Test model version deletion"""
        
        model = MockMLModel("deletable_model")
        
        # Register multiple versions
        for version in ["1.0.0", "1.1.0", "1.2.0"]:
            temp_registry.register_model(
                name="deletable_model",
                model=model,
                version=version,
                description=f"Version {version}",
                model_type="classifier",
                framework="sklearn",
                created_by="test_user"
            )
        
        metadata = temp_registry.get_model_info("deletable_model")
        assert len(metadata.versions) == 3
        
        # Delete middle version
        success = temp_registry.delete_model_version("deletable_model", "1.1.0")
        assert success
        
        metadata = temp_registry.get_model_info("deletable_model")
        assert len(metadata.versions) == 2
        assert "1.1.0" not in metadata.versions
        
        # Try to load deleted version
        deleted_model = temp_registry.load_model("deletable_model", "1.1.0")
        assert deleted_model is None
    
    def test_registry_stats(self, temp_registry):
        """Test registry statistics"""
        
        # Register some models
        for i in range(3):
            model = MockMLModel(f"stats_model_{i}")
            temp_registry.register_model(
                name=f"stats_model_{i}",
                model=model,
                version="1.0.0",
                description=f"Stats test model {i}",
                model_type="classifier" if i < 2 else "regressor",
                framework="sklearn",
                created_by="test_user"
            )
        
        # Deploy one to production
        temp_registry.deploy_model("stats_model_0", "1.0.0", ModelStatus.PRODUCTION)
        
        stats = temp_registry.get_registry_stats()
        
        assert stats['total_models'] == 3
        assert stats['total_versions'] == 3
        assert stats['status_distribution']['production'] == 1
        assert stats['status_distribution']['development'] == 2
        assert stats['framework_distribution']['sklearn'] == 3
        assert stats['type_distribution']['classifier'] == 2
        assert stats['type_distribution']['regressor'] == 1
    
    def test_callbacks(self, temp_registry):
        """Test registry callbacks"""
        
        registered_calls = []
        deployed_calls = []
        
        def on_registered(name, version):
            registered_calls.append((name, version))
        
        def on_deployed(name, version, status):
            deployed_calls.append((name, version, status))
        
        temp_registry.add_model_registered_callback(on_registered)
        temp_registry.add_model_deployed_callback(on_deployed)
        
        model = MockMLModel("callback_model")
        
        # Register model
        temp_registry.register_model(
            name="callback_model",
            model=model,
            version="1.0.0",
            description="Callback test model",
            model_type="classifier",
            framework="sklearn",
            created_by="test_user"
        )
        
        assert len(registered_calls) == 1
        assert registered_calls[0] == ("callback_model", "1.0.0")
        
        # Deploy model
        temp_registry.deploy_model("callback_model", "1.0.0", ModelStatus.PRODUCTION)
        
        assert len(deployed_calls) == 1
        assert deployed_calls[0] == ("callback_model", "1.0.0", ModelStatus.PRODUCTION)
    
    def test_model_metrics_class(self):
        """Test ModelMetrics functionality"""
        
        metrics = ModelMetrics(
            accuracy=0.95,
            precision=0.90,
            recall=0.88,
            f1_score=0.89,
            custom_metrics={'auc_score': 0.97}
        )
        
        # Test to_dict
        metrics_dict = metrics.to_dict()
        assert metrics_dict['accuracy'] == 0.95
        assert metrics_dict['custom_metrics']['auc_score'] == 0.97
        
        # Test from_dict
        reconstructed = ModelMetrics.from_dict(metrics_dict)
        assert reconstructed.accuracy == 0.95
        assert reconstructed.custom_metrics['auc_score'] == 0.97
    
    def test_version_cleanup(self, temp_registry):
        """Test automatic version cleanup"""
        
        temp_registry.max_versions_per_model = 3
        model = MockMLModel("cleanup_model")
        
        # Register more versions than the limit
        for i in range(5):
            temp_registry.register_model(
                name="cleanup_model",
                model=model,
                version=f"1.{i}.0",
                description=f"Version 1.{i}.0",
                model_type="classifier",
                framework="sklearn",
                created_by="test_user"
            )
        
        metadata = temp_registry.get_model_info("cleanup_model")
        # Should keep only the latest 3 versions
        assert len(metadata.versions) <= temp_registry.max_versions_per_model