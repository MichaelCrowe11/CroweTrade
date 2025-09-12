"""Model serialization and deserialization utilities"""
import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Type, Dict, Optional

logger = logging.getLogger(__name__)


class ModelSerializer(ABC):
    """Abstract base class for model serializers"""
    
    @abstractmethod
    def save(self, model: Any, path: str) -> None:
        """Save model to file"""
        pass
    
    @abstractmethod
    def load(self, path: str) -> Any:
        """Load model from file"""
        pass
    
    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """File extensions supported by this serializer"""
        pass
    
    @property
    @abstractmethod
    def supported_types(self) -> list[str]:
        """Model types/frameworks supported"""
        pass


class PickleSerializer(ModelSerializer):
    """Pickle-based serializer for scikit-learn and general Python objects"""
    
    def save(self, model: Any, path: str) -> None:
        """Save model using pickle"""
        try:
            with open(path, 'wb') as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug(f"Saved model to {path} using pickle")
        except Exception as e:
            logger.error(f"Failed to save model with pickle: {e}")
            raise
    
    def load(self, path: str) -> Any:
        """Load model using pickle"""
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
            logger.debug(f"Loaded model from {path} using pickle")
            return model
        except Exception as e:
            logger.error(f"Failed to load model with pickle: {e}")
            raise
    
    @property
    def supported_extensions(self) -> list[str]:
        return ['.pkl', '.pickle']
    
    @property
    def supported_types(self) -> list[str]:
        return ['sklearn', 'xgboost', 'lightgbm', 'catboost', 'general']


class JoblibSerializer(ModelSerializer):
    """Joblib-based serializer for scikit-learn models (more efficient)"""
    
    def __init__(self):
        try:
            import joblib
            self.joblib = joblib
        except ImportError:
            logger.warning("joblib not available, falling back to pickle")
            self.joblib = None
    
    def save(self, model: Any, path: str) -> None:
        """Save model using joblib"""
        if not self.joblib:
            # Fallback to pickle
            PickleSerializer().save(model, path)
            return
        
        try:
            self.joblib.dump(model, path, compress=3)
            logger.debug(f"Saved model to {path} using joblib")
        except Exception as e:
            logger.error(f"Failed to save model with joblib: {e}")
            raise
    
    def load(self, path: str) -> Any:
        """Load model using joblib"""
        if not self.joblib:
            # Fallback to pickle
            return PickleSerializer().load(path)
        
        try:
            model = self.joblib.load(path)
            logger.debug(f"Loaded model from {path} using joblib")
            return model
        except Exception as e:
            logger.error(f"Failed to load model with joblib: {e}")
            raise
    
    @property
    def supported_extensions(self) -> list[str]:
        return ['.joblib', '.pkl']
    
    @property
    def supported_types(self) -> list[str]:
        return ['sklearn', 'xgboost', 'lightgbm']


class ONNXSerializer(ModelSerializer):
    """ONNX serializer for cross-platform model deployment"""
    
    def __init__(self):
        try:
            import onnx
            import onnxruntime
            self.onnx = onnx
            self.onnxruntime = onnxruntime
        except ImportError:
            logger.warning("ONNX not available")
            self.onnx = None
            self.onnxruntime = None
    
    def save(self, model: Any, path: str) -> None:
        """Save model in ONNX format"""
        if not self.onnx:
            raise RuntimeError("ONNX not available")
        
        try:
            # This is a simplified example - actual ONNX conversion
            # would depend on the source framework
            if hasattr(model, 'to_onnx'):
                # sklearn models with skl2onnx
                onnx_model = model.to_onnx()
                self.onnx.save_model(onnx_model, path)
            else:
                raise ValueError("Model doesn't support ONNX conversion")
            
            logger.debug(f"Saved model to {path} using ONNX")
        except Exception as e:
            logger.error(f"Failed to save model with ONNX: {e}")
            raise
    
    def load(self, path: str) -> Any:
        """Load ONNX model"""
        if not self.onnxruntime:
            raise RuntimeError("ONNX Runtime not available")
        
        try:
            # Create ONNX Runtime session
            session = self.onnxruntime.InferenceSession(path)
            
            # Wrap in a simple inference class
            class ONNXModel:
                def __init__(self, session):
                    self.session = session
                    self.input_name = session.get_inputs()[0].name
                    self.output_name = session.get_outputs()[0].name
                
                def predict(self, X):
                    return self.session.run(
                        [self.output_name],
                        {self.input_name: X}
                    )[0]
                
                def get_params(self):
                    return {'onnx_path': path}
                
                def set_params(self, **params):
                    pass
            
            model = ONNXModel(session)
            logger.debug(f"Loaded ONNX model from {path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise
    
    @property
    def supported_extensions(self) -> list[str]:
        return ['.onnx']
    
    @property
    def supported_types(self) -> list[str]:
        return ['onnx', 'sklearn', 'pytorch', 'tensorflow']


class PyTorchSerializer(ModelSerializer):
    """PyTorch model serializer"""
    
    def __init__(self):
        try:
            import torch
            self.torch = torch
        except ImportError:
            logger.warning("PyTorch not available")
            self.torch = None
    
    def save(self, model: Any, path: str) -> None:
        """Save PyTorch model"""
        if not self.torch:
            raise RuntimeError("PyTorch not available")
        
        try:
            # Save state dict (recommended)
            if hasattr(model, 'state_dict'):
                self.torch.save(model.state_dict(), path)
            else:
                # Save entire model (less portable)
                self.torch.save(model, path)
            
            logger.debug(f"Saved PyTorch model to {path}")
        except Exception as e:
            logger.error(f"Failed to save PyTorch model: {e}")
            raise
    
    def load(self, path: str) -> Any:
        """Load PyTorch model"""
        if not self.torch:
            raise RuntimeError("PyTorch not available")
        
        try:
            # Try to load as state dict first, then as full model
            model = self.torch.load(path, map_location='cpu')
            logger.debug(f"Loaded PyTorch model from {path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            raise
    
    @property
    def supported_extensions(self) -> list[str]:
        return ['.pt', '.pth', '.pytorch']
    
    @property
    def supported_types(self) -> list[str]:
        return ['pytorch']


class TensorFlowSerializer(ModelSerializer):
    """TensorFlow/Keras model serializer"""
    
    def __init__(self):
        try:
            import tensorflow as tf
            self.tf = tf
        except ImportError:
            logger.warning("TensorFlow not available")
            self.tf = None
    
    def save(self, model: Any, path: str) -> None:
        """Save TensorFlow model"""
        if not self.tf:
            raise RuntimeError("TensorFlow not available")
        
        try:
            # Use SavedModel format for TensorFlow 2.x
            if hasattr(model, 'save'):
                model.save(path, save_format='tf')
            else:
                self.tf.saved_model.save(model, path)
            
            logger.debug(f"Saved TensorFlow model to {path}")
        except Exception as e:
            logger.error(f"Failed to save TensorFlow model: {e}")
            raise
    
    def load(self, path: str) -> Any:
        """Load TensorFlow model"""
        if not self.tf:
            raise RuntimeError("TensorFlow not available")
        
        try:
            model = self.tf.keras.models.load_model(path)
            logger.debug(f"Loaded TensorFlow model from {path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load TensorFlow model: {e}")
            raise
    
    @property
    def supported_extensions(self) -> list[str]:
        return ['.h5', '.tf', '.pb']
    
    @property
    def supported_types(self) -> list[str]:
        return ['tensorflow', 'keras']


# Registry of available serializers
_SERIALIZERS: Dict[str, Type[ModelSerializer]] = {
    'pickle': PickleSerializer,
    'joblib': JoblibSerializer,
    'onnx': ONNXSerializer,
    'pytorch': PyTorchSerializer,
    'tensorflow': TensorFlowSerializer,
}

# Default serializer priority order
_DEFAULT_PRIORITY = ['joblib', 'pickle', 'onnx', 'pytorch', 'tensorflow']


def get_serializer(model: Any) -> ModelSerializer:
    """
    Get appropriate serializer for a model
    
    Args:
        model: Model instance
    
    Returns:
        ModelSerializer instance
    """
    
    # Try to detect model type
    model_type = _detect_model_type(model)
    
    # Find compatible serializers
    compatible_serializers = []
    
    for serializer_name, serializer_class in _SERIALIZERS.items():
        try:
            serializer = serializer_class()
            if model_type in serializer.supported_types:
                compatible_serializers.append((serializer_name, serializer))
        except Exception:
            # Skip serializers that can't be instantiated
            continue
    
    if not compatible_serializers:
        logger.warning(f"No specific serializer found for {model_type}, using pickle")
        return PickleSerializer()
    
    # Sort by priority
    compatible_serializers.sort(
        key=lambda x: _DEFAULT_PRIORITY.index(x[0]) if x[0] in _DEFAULT_PRIORITY else 999
    )
    
    return compatible_serializers[0][1]


def get_serializer_by_extension(file_path: str) -> ModelSerializer:
    """
    Get serializer based on file extension
    
    Args:
        file_path: Path to model file
    
    Returns:
        ModelSerializer instance
    """
    
    extension = Path(file_path).suffix.lower()
    
    # Find serializer that supports this extension
    for serializer_name, serializer_class in _SERIALIZERS.items():
        try:
            serializer = serializer_class()
            if extension in serializer.supported_extensions:
                return serializer
        except Exception:
            continue
    
    # Default to pickle
    logger.warning(f"No specific serializer found for extension {extension}, using pickle")
    return PickleSerializer()


def get_serializer_by_name(name: str) -> Optional[ModelSerializer]:
    """
    Get serializer by name
    
    Args:
        name: Serializer name
    
    Returns:
        ModelSerializer instance or None
    """
    
    if name not in _SERIALIZERS:
        return None
    
    try:
        return _SERIALIZERS[name]()
    except Exception as e:
        logger.error(f"Failed to create serializer {name}: {e}")
        return None


def _detect_model_type(model: Any) -> str:
    """Detect model framework/type"""
    
    model_class = type(model).__name__
    model_module = getattr(type(model), '__module__', '')
    
    # Scikit-learn
    if 'sklearn' in model_module:
        return 'sklearn'
    
    # XGBoost
    if 'xgboost' in model_module or 'XGB' in model_class:
        return 'xgboost'
    
    # LightGBM
    if 'lightgbm' in model_module or 'LGB' in model_class:
        return 'lightgbm'
    
    # CatBoost
    if 'catboost' in model_module:
        return 'catboost'
    
    # PyTorch
    if 'torch' in model_module:
        return 'pytorch'
    
    # TensorFlow/Keras
    if 'tensorflow' in model_module or 'keras' in model_module:
        return 'tensorflow'
    
    # Default
    return 'general'


def list_available_serializers() -> Dict[str, Dict[str, Any]]:
    """List all available serializers and their capabilities"""
    
    available = {}
    
    for name, serializer_class in _SERIALIZERS.items():
        try:
            serializer = serializer_class()
            available[name] = {
                'extensions': serializer.supported_extensions,
                'types': serializer.supported_types,
                'available': True
            }
        except Exception as e:
            available[name] = {
                'extensions': [],
                'types': [],
                'available': False,
                'error': str(e)
            }
    
    return available