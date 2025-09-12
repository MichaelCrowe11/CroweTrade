"""Tests for model serializers"""
import pytest
import tempfile
import numpy as np
from unittest.mock import Mock, patch

from crowetrade.models.serializers import (
    PickleSerializer,
    JoblibSerializer,
    get_serializer,
    get_serializer_by_extension,
    get_serializer_by_name,
    list_available_serializers,
    _detect_model_type
)


class MockSklearnModel:
    """Mock scikit-learn model"""
    
    def __init__(self):
        self.__module__ = 'sklearn.ensemble.forest'
        self.feature_importances_ = np.array([0.1, 0.2, 0.7])
    
    def predict(self, X):
        return np.random.random((X.shape[0],))
    
    def fit(self, X, y):
        return self
    
    def get_params(self):
        return {'n_estimators': 100}
    
    def set_params(self, **params):
        pass


class MockXGBoostModel:
    """Mock XGBoost model"""
    
    def __init__(self):
        self.__module__ = 'xgboost.sklearn'
        self.__class__.__name__ = 'XGBClassifier'
    
    def predict(self, X):
        return np.random.random((X.shape[0],))
    
    def fit(self, X, y):
        return self
    
    def get_params(self):
        return {'n_estimators': 100, 'learning_rate': 0.1}
    
    def set_params(self, **params):
        pass


class TestPickleSerializer:
    """Test pickle serializer"""
    
    @pytest.fixture
    def serializer(self):
        return PickleSerializer()
    
    @pytest.fixture
    def temp_file(self):
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            yield f.name
    
    def test_save_and_load(self, serializer, temp_file):
        """Test saving and loading with pickle"""
        
        model = MockSklearnModel()
        
        # Save model
        serializer.save(model, temp_file)
        
        # Load model
        loaded_model = serializer.load(temp_file)
        
        assert loaded_model is not None
        assert hasattr(loaded_model, 'predict')
        assert hasattr(loaded_model, 'feature_importances_')
        np.testing.assert_array_equal(
            loaded_model.feature_importances_, 
            model.feature_importances_
        )
    
    def test_supported_extensions(self, serializer):
        """Test supported file extensions"""
        
        extensions = serializer.supported_extensions
        assert '.pkl' in extensions
        assert '.pickle' in extensions
    
    def test_supported_types(self, serializer):
        """Test supported model types"""
        
        types = serializer.supported_types
        assert 'sklearn' in types
        assert 'general' in types


class TestJoblibSerializer:
    """Test joblib serializer"""
    
    @pytest.fixture
    def serializer(self):
        return JoblibSerializer()
    
    @pytest.fixture
    def temp_file(self):
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            yield f.name
    
    def test_save_and_load_with_joblib(self, serializer, temp_file):
        """Test saving and loading with joblib if available"""
        
        model = MockSklearnModel()
        
        # Mock joblib availability
        if hasattr(serializer, 'joblib') and serializer.joblib:
            # Save and load
            serializer.save(model, temp_file)
            loaded_model = serializer.load(temp_file)
            
            assert loaded_model is not None
            assert hasattr(loaded_model, 'predict')
        else:
            # Should fall back to pickle
            serializer.save(model, temp_file)
            loaded_model = serializer.load(temp_file)
            assert loaded_model is not None
    
    def test_fallback_to_pickle(self):
        """Test fallback to pickle when joblib unavailable"""
        
        # Create serializer with no joblib
        serializer = JoblibSerializer()
        serializer.joblib = None
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_file = f.name
        
        model = MockSklearnModel()
        
        # Should use pickle fallback
        serializer.save(model, temp_file)
        loaded_model = serializer.load(temp_file)
        
        assert loaded_model is not None


class TestSerializerDetection:
    """Test automatic serializer detection"""
    
    def test_detect_sklearn_model(self):
        """Test detection of sklearn models"""
        
        model = MockSklearnModel()
        model_type = _detect_model_type(model)
        assert model_type == 'sklearn'
    
    def test_detect_xgboost_model(self):
        """Test detection of XGBoost models"""
        
        model = MockXGBoostModel()
        model_type = _detect_model_type(model)
        assert model_type == 'xgboost'
    
    def test_detect_unknown_model(self):
        """Test detection of unknown model types"""
        
        class UnknownModel:
            pass
        
        model = UnknownModel()
        model_type = _detect_model_type(model)
        assert model_type == 'general'
    
    def test_get_serializer_for_sklearn(self):
        """Test getting appropriate serializer for sklearn model"""
        
        model = MockSklearnModel()
        serializer = get_serializer(model)
        
        # Should prefer joblib for sklearn models
        supported_types = serializer.supported_types
        assert 'sklearn' in supported_types
    
    def test_get_serializer_for_unknown(self):
        """Test getting serializer for unknown model type"""
        
        class UnknownModel:
            def predict(self, X):
                return np.zeros(X.shape[0])
        
        model = UnknownModel()
        serializer = get_serializer(model)
        
        # Should fall back to pickle
        assert isinstance(serializer, PickleSerializer)
    
    def test_get_serializer_by_extension(self):
        """Test getting serializer by file extension"""
        
        pkl_serializer = get_serializer_by_extension('model.pkl')
        assert isinstance(pkl_serializer, PickleSerializer)
        
        joblib_serializer = get_serializer_by_extension('model.joblib')
        # Could be joblib or pickle depending on availability
        assert joblib_serializer is not None
        
        unknown_serializer = get_serializer_by_extension('model.unknown')
        # Should default to pickle
        assert isinstance(unknown_serializer, PickleSerializer)
    
    def test_get_serializer_by_name(self):
        """Test getting serializer by name"""
        
        pickle_serializer = get_serializer_by_name('pickle')
        assert isinstance(pickle_serializer, PickleSerializer)
        
        joblib_serializer = get_serializer_by_name('joblib')
        assert joblib_serializer is not None
        
        unknown_serializer = get_serializer_by_name('unknown')
        assert unknown_serializer is None
    
    def test_list_available_serializers(self):
        """Test listing available serializers"""
        
        serializers = list_available_serializers()
        
        assert 'pickle' in serializers
        assert 'joblib' in serializers
        
        # Pickle should always be available
        assert serializers['pickle']['available'] is True
        
        # Each serializer should have extensions and types
        for name, info in serializers.items():
            if info['available']:
                assert 'extensions' in info
                assert 'types' in info
                assert isinstance(info['extensions'], list)
                assert isinstance(info['types'], list)


class TestSerializerIntegration:
    """Test serializer integration scenarios"""
    
    def test_round_trip_serialization(self):
        """Test complete save/load cycle"""
        
        model = MockSklearnModel()
        model.custom_attr = "test_value"
        
        with tempfile.NamedTemporaryFile(suffix='.pkl') as f:
            # Get appropriate serializer
            serializer = get_serializer(model)
            
            # Save model
            serializer.save(model, f.name)
            
            # Load model
            loaded_model = serializer.load(f.name)
            
            # Verify model functionality
            assert hasattr(loaded_model, 'predict')
            assert hasattr(loaded_model, 'custom_attr')
            assert loaded_model.custom_attr == "test_value"
            
            # Test prediction capability
            X_test = np.random.random((10, 3))
            predictions = loaded_model.predict(X_test)
            assert len(predictions) == 10
    
    def test_serializer_error_handling(self):
        """Test error handling in serializers"""
        
        serializer = PickleSerializer()
        
        # Test save to invalid path
        with pytest.raises(Exception):
            serializer.save(MockSklearnModel(), "/invalid/path/model.pkl")
        
        # Test load from non-existent file
        with pytest.raises(Exception):
            serializer.load("/non/existent/file.pkl")
    
    @patch('crowetrade.models.serializers.pickle')
    def test_serialization_failure_handling(self, mock_pickle):
        """Test handling of serialization failures"""
        
        # Mock pickle to raise an exception
        mock_pickle.dump.side_effect = Exception("Serialization failed")
        mock_pickle.HIGHEST_PROTOCOL = 4
        
        serializer = PickleSerializer()
        model = MockSklearnModel()
        
        with tempfile.NamedTemporaryFile() as f:
            with pytest.raises(Exception):
                serializer.save(model, f.name)
    
    def test_large_model_serialization(self):
        """Test serialization of larger models"""
        
        class LargeModel:
            def __init__(self):
                # Create large arrays to test serialization performance
                self.large_weights = np.random.random((1000, 1000))
                self.large_biases = np.random.random(1000)
            
            def predict(self, X):
                return np.dot(X, self.large_weights[:X.shape[1], :100]).mean(axis=1)
            
            def get_params(self):
                return {'weights_shape': self.large_weights.shape}
            
            def set_params(self, **params):
                pass
        
        model = LargeModel()
        
        with tempfile.NamedTemporaryFile(suffix='.pkl') as f:
            serializer = PickleSerializer()
            
            # Save large model
            serializer.save(model, f.name)
            
            # Load large model
            loaded_model = serializer.load(f.name)
            
            assert loaded_model is not None
            assert hasattr(loaded_model, 'large_weights')
            assert loaded_model.large_weights.shape == (1000, 1000)
            
            # Test functionality
            X_test = np.random.random((5, 10))
            predictions = loaded_model.predict(X_test)
            assert len(predictions) == 5


class TestSpecializedSerializers:
    """Test specialized serializers (ONNX, PyTorch, TensorFlow)"""
    
    def test_onnx_serializer_unavailable(self):
        """Test ONNX serializer when libraries not available"""
        
        from crowetrade.models.serializers import ONNXSerializer
        serializer = ONNXSerializer()
        
        # If ONNX not available, should handle gracefully
        if not serializer.onnx:
            with pytest.raises(RuntimeError):
                serializer.save(Mock(), "test.onnx")
    
    def test_pytorch_serializer_unavailable(self):
        """Test PyTorch serializer when PyTorch not available"""
        
        from crowetrade.models.serializers import PyTorchSerializer
        serializer = PyTorchSerializer()
        
        # If PyTorch not available, should handle gracefully
        if not serializer.torch:
            with pytest.raises(RuntimeError):
                serializer.save(Mock(), "test.pt")
    
    def test_tensorflow_serializer_unavailable(self):
        """Test TensorFlow serializer when TensorFlow not available"""
        
        from crowetrade.models.serializers import TensorFlowSerializer
        serializer = TensorFlowSerializer()
        
        # If TensorFlow not available, should handle gracefully
        if not serializer.tf:
            with pytest.raises(RuntimeError):
                serializer.save(Mock(), "test.h5")
    
    def test_serializer_capabilities(self):
        """Test that all serializers report their capabilities correctly"""
        
        serializers_info = list_available_serializers()
        
        for name, info in serializers_info.items():
            assert 'available' in info
            assert 'extensions' in info
            assert 'types' in info
            
            if info['available']:
                # Available serializers should have non-empty capabilities
                assert len(info['extensions']) > 0
                assert len(info['types']) > 0
            else:
                # Unavailable serializers should have error info
                assert 'error' in info