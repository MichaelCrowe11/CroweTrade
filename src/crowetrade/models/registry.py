"""Model registry for managing ML models with versioning and metadata"""
import logging
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Protocol, Callable
import threading

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model deployment status"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


@dataclass
class ModelMetrics:
    """Model performance metrics"""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetrics':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class ModelVersion:
    """Version information for a model"""
    version: str
    created_at: datetime
    created_by: str
    description: str
    status: ModelStatus = ModelStatus.DEVELOPMENT
    metrics: Optional[ModelMetrics] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Model artifacts
    model_path: Optional[str] = None
    config_path: Optional[str] = None
    weights_path: Optional[str] = None
    
    # Validation
    is_validated: bool = False
    validation_results: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['status'] = self.status.value
        if self.metrics:
            data['metrics'] = self.metrics.to_dict()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """Create from dictionary"""
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['status'] = ModelStatus(data['status'])
        if 'metrics' in data and data['metrics']:
            data['metrics'] = ModelMetrics.from_dict(data['metrics'])
        return cls(**data)


@dataclass
class ModelMetadata:
    """Complete metadata for a model"""
    name: str
    description: str
    model_type: str  # 'classifier', 'regressor', 'reinforcement', etc.
    framework: str   # 'sklearn', 'pytorch', 'tensorflow', etc.
    created_at: datetime
    created_by: str
    
    # Current version info
    latest_version: str
    production_version: Optional[str] = None
    
    # All versions
    versions: Dict[str, ModelVersion] = field(default_factory=dict)
    
    # Model schema
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    python_version: Optional[str] = None
    
    # Usage tracking
    usage_count: int = 0
    last_used: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        if self.last_used:
            data['last_used'] = self.last_used.isoformat()
        
        # Convert versions
        data['versions'] = {
            version_str: version.to_dict()
            for version_str, version in self.versions.items()
        }
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary"""
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'last_used' in data and data['last_used']:
            data['last_used'] = datetime.fromisoformat(data['last_used'])
        
        # Convert versions
        if 'versions' in data:
            data['versions'] = {
                version_str: ModelVersion.from_dict(version_data)
                for version_str, version_data in data['versions'].items()
            }
        
        return cls(**data)


@dataclass
class ModelArtifact:
    """Model artifact (file) information"""
    path: str
    size_bytes: int
    checksum: str
    created_at: datetime
    artifact_type: str  # 'model', 'config', 'weights', 'metadata'


class MLModel(Protocol):
    """Protocol for ML model interface"""
    
    def predict(self, X: Any) -> Any:
        """Make predictions on input data"""
        ...
    
    def fit(self, X: Any, y: Any) -> None:
        """Train the model on data"""
        ...
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        ...
    
    def set_params(self, **params) -> None:
        """Set model parameters"""
        ...


class ModelRegistry:
    """
    Centralized registry for ML models with versioning, metadata, and lifecycle management
    """
    
    def __init__(
        self,
        registry_path: str = "./model_registry",
        enable_validation: bool = True,
        auto_backup: bool = True,
        max_versions_per_model: int = 50
    ):
        self.registry_path = Path(registry_path)
        self.enable_validation = enable_validation
        self.auto_backup = auto_backup
        self.max_versions_per_model = max_versions_per_model
        
        # Thread safety
        self._lock = threading.RLock()
        
        # In-memory cache
        self._models: Dict[str, ModelMetadata] = {}
        self._loaded_models: Dict[str, MLModel] = {}
        
        # Callbacks
        self.model_registered_callbacks: List[Callable[[str, str], None]] = []
        self.model_deployed_callbacks: List[Callable[[str, str, ModelStatus], None]] = []
        
        # Initialize registry
        self._initialize_registry()
        self._load_models_metadata()
    
    def _initialize_registry(self) -> None:
        """Initialize registry directory structure"""
        
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.registry_path / "models").mkdir(exist_ok=True)
        (self.registry_path / "metadata").mkdir(exist_ok=True)
        (self.registry_path / "backups").mkdir(exist_ok=True)
        
        logger.info(f"Initialized model registry at {self.registry_path}")
    
    def _load_models_metadata(self) -> None:
        """Load all model metadata from disk"""
        
        metadata_dir = self.registry_path / "metadata"
        
        with self._lock:
            for metadata_file in metadata_dir.glob("*.json"):
                try:
                    with open(metadata_file, 'r') as f:
                        data = json.load(f)
                    
                    metadata = ModelMetadata.from_dict(data)
                    self._models[metadata.name] = metadata
                    
                except Exception as e:
                    logger.error(f"Failed to load metadata from {metadata_file}: {e}")
        
        logger.info(f"Loaded {len(self._models)} models from registry")
    
    def register_model(
        self,
        name: str,
        model: MLModel,
        version: str,
        description: str,
        model_type: str,
        framework: str,
        created_by: str,
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        metrics: Optional[ModelMetrics] = None,
        tags: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register a new model or version
        
        Returns:
            Model path for the registered version
        """
        
        with self._lock:
            # Create or update model metadata
            if name not in self._models:
                model_metadata = ModelMetadata(
                    name=name,
                    description=description,
                    model_type=model_type,
                    framework=framework,
                    created_at=datetime.now(),
                    created_by=created_by,
                    latest_version=version,
                    input_schema=input_schema,
                    output_schema=output_schema,
                    dependencies=dependencies or [],
                    python_version=self._get_python_version()
                )
                self._models[name] = model_metadata
            else:
                model_metadata = self._models[name]
                model_metadata.latest_version = version
            
            # Create version info
            version_info = ModelVersion(
                version=version,
                created_at=datetime.now(),
                created_by=created_by,
                description=description,
                metrics=metrics,
                tags=tags or [],
                metadata=metadata or {}
            )
            
            # Save model artifacts
            model_path = self._save_model_artifacts(name, version, model, version_info)
            version_info.model_path = str(model_path)
            
            # Add version to metadata
            model_metadata.versions[version] = version_info
            
            # Cleanup old versions if needed
            self._cleanup_old_versions(name)
            
            # Save metadata
            self._save_model_metadata(model_metadata)
            
            # Cache loaded model
            self._loaded_models[f"{name}:{version}"] = model
            
            # Trigger callbacks
            for callback in self.model_registered_callbacks:
                try:
                    callback(name, version)
                except Exception as e:
                    logger.error(f"Error in model registered callback: {e}")
            
            logger.info(f"Registered model {name}:{version}")
            
            return str(model_path)
    
    def load_model(
        self,
        name: str,
        version: Optional[str] = None,
        status: Optional[ModelStatus] = None
    ) -> Optional[MLModel]:
        """
        Load a model from the registry
        
        Args:
            name: Model name
            version: Specific version (defaults to latest)
            status: Load model with specific status (e.g., production)
        
        Returns:
            Loaded model instance
        """
        
        with self._lock:
            if name not in self._models:
                logger.error(f"Model {name} not found in registry")
                return None
            
            model_metadata = self._models[name]
            
            # Determine version to load
            target_version = self._resolve_version(model_metadata, version, status)
            if not target_version:
                return None
            
            # Check cache first
            cache_key = f"{name}:{target_version}"
            if cache_key in self._loaded_models:
                self._update_usage_stats(name)
                return self._loaded_models[cache_key]
            
            # Load from disk
            version_info = model_metadata.versions[target_version]
            if not version_info.model_path or not os.path.exists(version_info.model_path):
                logger.error(f"Model artifacts not found for {name}:{target_version}")
                return None
            
            try:
                model = self._load_model_from_path(version_info.model_path)
                self._loaded_models[cache_key] = model
                self._update_usage_stats(name)
                
                logger.info(f"Loaded model {name}:{target_version}")
                return model
                
            except Exception as e:
                logger.error(f"Failed to load model {name}:{target_version}: {e}")
                return None
    
    def deploy_model(
        self,
        name: str,
        version: str,
        status: ModelStatus
    ) -> bool:
        """Deploy model to a specific status (staging/production)"""
        
        with self._lock:
            if name not in self._models:
                logger.error(f"Model {name} not found")
                return False
            
            model_metadata = self._models[name]
            
            if version not in model_metadata.versions:
                logger.error(f"Version {version} not found for model {name}")
                return False
            
            version_info = model_metadata.versions[version]
            
            # Update version status
            old_status = version_info.status
            version_info.status = status
            
            # Update production version if deploying to production
            if status == ModelStatus.PRODUCTION:
                model_metadata.production_version = version
            
            # Save metadata
            self._save_model_metadata(model_metadata)
            
            # Trigger callbacks
            for callback in self.model_deployed_callbacks:
                try:
                    callback(name, version, status)
                except Exception as e:
                    logger.error(f"Error in model deployed callback: {e}")
            
            logger.info(f"Deployed model {name}:{version} to {status.value}")
            return True
    
    def get_model_info(self, name: str) -> Optional[ModelMetadata]:
        """Get complete model information"""
        return self._models.get(name)
    
    def list_models(
        self,
        model_type: Optional[str] = None,
        framework: Optional[str] = None,
        status: Optional[ModelStatus] = None
    ) -> List[str]:
        """List models matching criteria"""
        
        models = []
        
        for name, metadata in self._models.items():
            # Filter by model type
            if model_type and metadata.model_type != model_type:
                continue
            
            # Filter by framework
            if framework and metadata.framework != framework:
                continue
            
            # Filter by status (check any version has this status)
            if status:
                has_status = any(
                    version.status == status
                    for version in metadata.versions.values()
                )
                if not has_status:
                    continue
            
            models.append(name)
        
        return sorted(models)
    
    def get_model_versions(self, name: str) -> List[str]:
        """Get all versions of a model"""
        
        if name not in self._models:
            return []
        
        return list(self._models[name].versions.keys())
    
    def delete_model_version(
        self,
        name: str,
        version: str,
        force: bool = False
    ) -> bool:
        """Delete a specific model version"""
        
        with self._lock:
            if name not in self._models:
                logger.error(f"Model {name} not found")
                return False
            
            model_metadata = self._models[name]
            
            if version not in model_metadata.versions:
                logger.error(f"Version {version} not found for model {name}")
                return False
            
            version_info = model_metadata.versions[version]
            
            # Check if it's production version
            if version_info.status == ModelStatus.PRODUCTION and not force:
                logger.error(f"Cannot delete production version {version} without force=True")
                return False
            
            # Delete artifacts
            if version_info.model_path and os.path.exists(version_info.model_path):
                try:
                    if os.path.isdir(version_info.model_path):
                        shutil.rmtree(version_info.model_path)
                    else:
                        os.remove(version_info.model_path)
                except Exception as e:
                    logger.warning(f"Failed to delete model artifacts: {e}")
            
            # Remove from metadata
            del model_metadata.versions[version]
            
            # Update latest version if needed
            if model_metadata.latest_version == version:
                remaining_versions = list(model_metadata.versions.keys())
                if remaining_versions:
                    # Get most recent version
                    latest = max(
                        remaining_versions,
                        key=lambda v: model_metadata.versions[v].created_at
                    )
                    model_metadata.latest_version = latest
                else:
                    # No versions left, delete model
                    del self._models[name]
                    self._delete_model_metadata(name)
                    logger.info(f"Deleted model {name} (no versions remaining)")
                    return True
            
            # Remove from cache
            cache_key = f"{name}:{version}"
            if cache_key in self._loaded_models:
                del self._loaded_models[cache_key]
            
            # Save metadata
            self._save_model_metadata(model_metadata)
            
            logger.info(f"Deleted model version {name}:{version}")
            return True
    
    def validate_model(
        self,
        name: str,
        version: str,
        validation_data: Any,
        validation_func: Callable[[MLModel, Any], Dict[str, Any]]
    ) -> bool:
        """Validate a model version"""
        
        model = self.load_model(name, version)
        if not model:
            return False
        
        try:
            validation_results = validation_func(model, validation_data)
            
            # Update version info
            with self._lock:
                if name in self._models and version in self._models[name].versions:
                    version_info = self._models[name].versions[version]
                    version_info.is_validated = True
                    version_info.validation_results = validation_results
                    
                    self._save_model_metadata(self._models[name])
                    
                    logger.info(f"Validated model {name}:{version}")
                    return True
            
        except Exception as e:
            logger.error(f"Model validation failed for {name}:{version}: {e}")
        
        return False
    
    def compare_models(
        self,
        model1_name: str,
        model1_version: str,
        model2_name: str,
        model2_version: str
    ) -> Dict[str, Any]:
        """Compare metrics between two model versions"""
        
        model1_info = self._models.get(model1_name, {}).versions.get(model1_version)
        model2_info = self._models.get(model2_name, {}).versions.get(model2_version)
        
        if not model1_info or not model2_info:
            return {'error': 'Model version not found'}
        
        comparison = {
            'model1': {
                'name': model1_name,
                'version': model1_version,
                'metrics': model1_info.metrics.to_dict() if model1_info.metrics else {},
                'status': model1_info.status.value,
                'created_at': model1_info.created_at.isoformat()
            },
            'model2': {
                'name': model2_name,
                'version': model2_version,
                'metrics': model2_info.metrics.to_dict() if model2_info.metrics else {},
                'status': model2_info.status.value,
                'created_at': model2_info.created_at.isoformat()
            },
            'differences': {}
        }
        
        # Compare metrics
        if model1_info.metrics and model2_info.metrics:
            metrics1 = model1_info.metrics.to_dict()
            metrics2 = model2_info.metrics.to_dict()
            
            for metric_name in set(metrics1.keys()) | set(metrics2.keys()):
                val1 = metrics1.get(metric_name)
                val2 = metrics2.get(metric_name)
                
                if val1 is not None and val2 is not None:
                    comparison['differences'][metric_name] = {
                        'model1': val1,
                        'model2': val2,
                        'difference': val2 - val1,
                        'percent_change': ((val2 - val1) / val1 * 100) if val1 != 0 else float('inf')
                    }
        
        return comparison
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        
        total_models = len(self._models)
        total_versions = sum(len(m.versions) for m in self._models.values())
        
        status_counts = {status.value: 0 for status in ModelStatus}
        framework_counts = {}
        type_counts = {}
        
        for metadata in self._models.values():
            # Count by framework
            framework = metadata.framework
            framework_counts[framework] = framework_counts.get(framework, 0) + 1
            
            # Count by type
            model_type = metadata.model_type
            type_counts[model_type] = type_counts.get(model_type, 0) + 1
            
            # Count by status
            for version in metadata.versions.values():
                status_counts[version.status.value] += 1
        
        return {
            'total_models': total_models,
            'total_versions': total_versions,
            'status_distribution': status_counts,
            'framework_distribution': framework_counts,
            'type_distribution': type_counts,
            'registry_path': str(self.registry_path),
            'registry_size_mb': self._get_registry_size_mb()
        }
    
    def add_model_registered_callback(self, callback: Callable[[str, str], None]) -> None:
        """Add callback for model registration events"""
        self.model_registered_callbacks.append(callback)
    
    def add_model_deployed_callback(self, callback: Callable[[str, str, ModelStatus], None]) -> None:
        """Add callback for model deployment events"""
        self.model_deployed_callbacks.append(callback)
    
    # Helper methods
    
    def _resolve_version(
        self,
        metadata: ModelMetadata,
        version: Optional[str],
        status: Optional[ModelStatus]
    ) -> Optional[str]:
        """Resolve version based on criteria"""
        
        if version:
            if version in metadata.versions:
                return version
            else:
                logger.error(f"Version {version} not found")
                return None
        
        if status == ModelStatus.PRODUCTION:
            return metadata.production_version
        
        if status:
            # Find latest version with specified status
            matching_versions = [
                (v, info.created_at) for v, info in metadata.versions.items()
                if info.status == status
            ]
            if matching_versions:
                return max(matching_versions, key=lambda x: x[1])[0]
        
        # Default to latest version
        return metadata.latest_version
    
    def _save_model_artifacts(
        self,
        name: str,
        version: str,
        model: MLModel,
        version_info: ModelVersion
    ) -> Path:
        """Save model artifacts to disk"""
        
        model_dir = self.registry_path / "models" / name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model using appropriate serializer
        from .serializers import get_serializer
        serializer = get_serializer(model)
        
        model_path = model_dir / "model.pkl"
        serializer.save(model, str(model_path))
        
        # Save additional metadata
        metadata_path = model_dir / "version_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(version_info.to_dict(), f, indent=2)
        
        return model_path
    
    def _load_model_from_path(self, model_path: str) -> MLModel:
        """Load model from file path"""
        
        from .serializers import get_serializer_by_extension
        
        serializer = get_serializer_by_extension(model_path)
        return serializer.load(model_path)
    
    def _save_model_metadata(self, metadata: ModelMetadata) -> None:
        """Save model metadata to disk"""
        
        metadata_file = self.registry_path / "metadata" / f"{metadata.name}.json"
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
    
    def _delete_model_metadata(self, name: str) -> None:
        """Delete model metadata file"""
        
        metadata_file = self.registry_path / "metadata" / f"{name}.json"
        if metadata_file.exists():
            metadata_file.unlink()
    
    def _cleanup_old_versions(self, name: str) -> None:
        """Remove old versions if exceeding limit"""
        
        metadata = self._models[name]
        versions = list(metadata.versions.items())
        
        if len(versions) > self.max_versions_per_model:
            # Sort by creation date, keep newest
            versions.sort(key=lambda x: x[1].created_at)
            
            versions_to_remove = versions[:-self.max_versions_per_model]
            
            for version_str, version_info in versions_to_remove:
                # Don't delete production versions
                if version_info.status == ModelStatus.PRODUCTION:
                    continue
                
                logger.info(f"Cleaning up old version {name}:{version_str}")
                self.delete_model_version(name, version_str, force=True)
    
    def _update_usage_stats(self, name: str) -> None:
        """Update model usage statistics"""
        
        if name in self._models:
            self._models[name].usage_count += 1
            self._models[name].last_used = datetime.now()
            
            # Save updated metadata
            self._save_model_metadata(self._models[name])
    
    def _get_python_version(self) -> str:
        """Get current Python version"""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    def _get_registry_size_mb(self) -> float:
        """Calculate registry size in MB"""
        
        total_size = 0
        
        for root, dirs, files in os.walk(self.registry_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    total_size += os.path.getsize(file_path)
                except (OSError, IOError):
                    pass
        
        return total_size / (1024 * 1024)  # Convert to MB