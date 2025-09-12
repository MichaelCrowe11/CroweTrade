"""Model Registry System

Centralized registry for managing trading models, versions, and deployment:
- Model artifact storage with checksums and signatures
- Version control and rollback capabilities  
- Performance tracking and model health monitoring
- Policy enforcement and approval workflows
- A/B testing and canary deployment support
"""

import hashlib
import json
import logging
import os
import pickle
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import uuid

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model lifecycle status"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ModelType(Enum):
    """Model type classifications"""
    SIGNAL = "signal"                # Alpha/signal generation models
    RISK = "risk"                    # Risk estimation models
    EXECUTION = "execution"          # Execution optimization models
    REGIME = "regime"                # Regime detection models
    PORTFOLIO = "portfolio"          # Portfolio optimization models


@dataclass
class ModelMetadata:
    """Model metadata and configuration"""
    model_id: str
    name: str
    version: str
    model_type: ModelType
    status: ModelStatus
    created_at: datetime
    created_by: str
    description: str
    
    # Model specifics
    framework: str                   # sklearn, pytorch, xgboost, etc.
    input_features: List[str]        # Required input features
    output_schema: Dict[str, str]    # Output format specification
    
    # Performance metrics
    backtest_sharpe: Optional[float] = None
    backtest_returns: Optional[float] = None
    backtest_max_dd: Optional[float] = None
    validation_score: Optional[float] = None
    
    # Deployment info
    resource_requirements: Optional[Dict[str, Any]] = None
    latency_requirements: Optional[float] = None  # Max inference time (ms)
    
    # Governance
    approval_status: str = "pending"
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    
    # Checksums and integrity
    model_checksum: Optional[str] = None
    policy_checksum: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        # Convert enums and datetime objects
        data['model_type'] = self.model_type.value
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        if self.approved_at:
            data['approved_at'] = self.approved_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary"""
        # Convert strings back to enums and datetime
        data['model_type'] = ModelType(data['model_type'])
        data['status'] = ModelStatus(data['status'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('approved_at'):
            data['approved_at'] = datetime.fromisoformat(data['approved_at'])
        return cls(**data)


@dataclass
class ModelArtifacts:
    """Model artifact file paths"""
    model_file: str                  # Serialized model (pickle, joblib, etc.)
    policy_file: Optional[str] = None       # YAML policy configuration
    feature_spec_file: Optional[str] = None # Feature specification
    validation_report: Optional[str] = None # Validation/backtest results
    documentation: Optional[str] = None     # Model documentation


class ModelRegistry:
    """Centralized model registry with versioning and governance"""
    
    def __init__(self, 
                 registry_path: Union[str, Path],
                 enable_signing: bool = True,
                 max_versions_per_model: int = 10):
        
        self.registry_path = Path(registry_path)
        self.enable_signing = enable_signing
        self.max_versions_per_model = max_versions_per_model
        
        # Create registry structure
        self.models_path = self.registry_path / "models"
        self.metadata_path = self.registry_path / "metadata"
        self.policies_path = self.registry_path / "policies"
        self.staging_path = self.registry_path / "staging"
        
        for path in [self.models_path, self.metadata_path, self.policies_path, self.staging_path]:
            path.mkdir(parents=True, exist_ok=True)
            
        # In-memory cache for frequently accessed metadata
        self._metadata_cache: Dict[str, ModelMetadata] = {}
        self._load_metadata_cache()
        
        logger.info(f"ModelRegistry initialized at {self.registry_path}")
    
    def register_model(self,
                      model_object: Any,
                      metadata: ModelMetadata,
                      artifacts: Optional[ModelArtifacts] = None,
                      policy_config: Optional[Dict[str, Any]] = None) -> str:
        """Register a new model version
        
        Args:
            model_object: The trained model object
            metadata: Model metadata
            artifacts: Optional additional artifacts
            policy_config: Trading policy configuration
            
        Returns:
            Model registry ID (model_id:version)
        """
        registry_id = f"{metadata.model_id}:{metadata.version}"
        
        try:
            # Create version-specific directory
            version_path = self.models_path / metadata.model_id / metadata.version
            version_path.mkdir(parents=True, exist_ok=True)
            
            # Serialize and save model
            model_file = version_path / "model.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model_object, f)
            
            # Calculate model checksum
            model_checksum = self._calculate_checksum(model_file)
            metadata.model_checksum = model_checksum
            
            # Save policy configuration if provided
            policy_checksum = None
            if policy_config:
                policy_file = version_path / "policy.yaml"
                with open(policy_file, 'w') as f:
                    try:
                        import yaml
                        yaml.dump(policy_config, f, default_flow_style=False)
                    except ImportError:
                        # Fallback to JSON if yaml not available
                        import json
                        json.dump(policy_config, f, indent=2)
                policy_checksum = self._calculate_checksum(policy_file)
                metadata.policy_checksum = policy_checksum
            
            # Copy additional artifacts
            if artifacts:
                self._copy_artifacts(artifacts, version_path)
            
            # Save metadata
            metadata_file = self.metadata_path / f"{registry_id}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            # Update cache
            self._metadata_cache[registry_id] = metadata
            
            # Cleanup old versions if needed
            self._cleanup_old_versions(metadata.model_id)
            
            logger.info(f"Model registered: {registry_id} (checksum: {model_checksum[:8]})")
            return registry_id
            
        except Exception as e:
            logger.error(f"Error registering model {registry_id}: {e}")
            # Cleanup partial registration
            if 'version_path' in locals() and version_path.exists():
                shutil.rmtree(version_path, ignore_errors=True)
            raise
    
    def get_model(self, model_id: str, version: Optional[str] = None) -> Tuple[Any, ModelMetadata]:
        """Load model and metadata
        
        Args:
            model_id: Model identifier
            version: Specific version (default: latest approved)
            
        Returns:
            Tuple of (model_object, metadata)
        """
        if version is None:
            version = self.get_latest_version(model_id, status=ModelStatus.PRODUCTION)
            if not version:
                version = self.get_latest_version(model_id)  # Fallback to any latest
        
        registry_id = f"{model_id}:{version}"
        
        # Get metadata
        metadata = self.get_metadata(registry_id)
        if not metadata:
            raise ValueError(f"Model not found: {registry_id}")
        
        # Load model object
        model_file = self.models_path / model_id / version / "model.pkl"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        # Verify checksum if available
        if metadata.model_checksum:
            current_checksum = self._calculate_checksum(model_file)
            if current_checksum != metadata.model_checksum:
                raise ValueError(f"Model checksum mismatch for {registry_id}")
        
        with open(model_file, 'rb') as f:
            model_object = pickle.load(f)
        
        logger.info(f"Model loaded: {registry_id}")
        return model_object, metadata
    
    def get_metadata(self, registry_id: str) -> Optional[ModelMetadata]:
        """Get model metadata"""
        # Check cache first
        if registry_id in self._metadata_cache:
            return self._metadata_cache[registry_id]
        
        # Load from disk
        metadata_file = self.metadata_path / f"{registry_id}.json"
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
            metadata = ModelMetadata.from_dict(data)
            self._metadata_cache[registry_id] = metadata
            return metadata
        except Exception as e:
            logger.error(f"Error loading metadata for {registry_id}: {e}")
            return None
    
    def list_models(self, 
                   model_type: Optional[ModelType] = None,
                   status: Optional[ModelStatus] = None,
                   created_by: Optional[str] = None) -> List[ModelMetadata]:
        """List models with optional filtering"""
        models = []
        
        for metadata_file in self.metadata_path.glob("*.json"):
            try:
                metadata = self.get_metadata(metadata_file.stem)
                if not metadata:
                    continue
                
                # Apply filters
                if model_type and metadata.model_type != model_type:
                    continue
                if status and metadata.status != status:
                    continue
                if created_by and metadata.created_by != created_by:
                    continue
                
                models.append(metadata)
                
            except Exception as e:
                logger.warning(f"Error loading metadata from {metadata_file}: {e}")
        
        # Sort by creation date (newest first)
        models.sort(key=lambda m: m.created_at, reverse=True)
        return models
    
    def get_latest_version(self, 
                          model_id: str, 
                          status: Optional[ModelStatus] = None) -> Optional[str]:
        """Get latest version of a model"""
        models = self.list_models()
        
        # Filter by model_id and status
        candidates = [
            m for m in models 
            if m.model_id == model_id and (status is None or m.status == status)
        ]
        
        if not candidates:
            return None
        
        # Sort by version (assuming semantic versioning)
        candidates.sort(key=lambda m: self._version_sort_key(m.version), reverse=True)
        return candidates[0].version
    
    def promote_model(self, 
                     registry_id: str, 
                     new_status: ModelStatus,
                     approved_by: Optional[str] = None) -> bool:
        """Promote model to new status"""
        metadata = self.get_metadata(registry_id)
        if not metadata:
            logger.error(f"Cannot promote unknown model: {registry_id}")
            return False
        
        # Status transition validation
        valid_transitions = {
            ModelStatus.DEVELOPMENT: [ModelStatus.TESTING, ModelStatus.DEPRECATED],
            ModelStatus.TESTING: [ModelStatus.STAGING, ModelStatus.DEVELOPMENT, ModelStatus.DEPRECATED],
            ModelStatus.STAGING: [ModelStatus.PRODUCTION, ModelStatus.TESTING, ModelStatus.DEPRECATED],
            ModelStatus.PRODUCTION: [ModelStatus.DEPRECATED, ModelStatus.ARCHIVED],
            ModelStatus.DEPRECATED: [ModelStatus.ARCHIVED]
        }
        
        if new_status not in valid_transitions.get(metadata.status, []):
            logger.error(f"Invalid status transition: {metadata.status.value} -> {new_status.value}")
            return False
        
        # Update metadata
        metadata.status = new_status
        if new_status == ModelStatus.PRODUCTION and approved_by:
            metadata.approved_by = approved_by
            metadata.approved_at = datetime.utcnow()
            metadata.approval_status = "approved"
        
        # Save updated metadata
        metadata_file = self.metadata_path / f"{registry_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # Update cache
        self._metadata_cache[registry_id] = metadata
        
        logger.info(f"Model promoted: {registry_id} -> {new_status.value}")
        return True
    
    def delete_model(self, registry_id: str, force: bool = False) -> bool:
        """Delete model version"""
        metadata = self.get_metadata(registry_id)
        if not metadata:
            return False
        
        # Protect production models
        if metadata.status == ModelStatus.PRODUCTION and not force:
            logger.error(f"Cannot delete production model without force=True: {registry_id}")
            return False
        
        try:
            model_id, version = registry_id.split(":")
            
            # Remove files
            version_path = self.models_path / model_id / version
            if version_path.exists():
                shutil.rmtree(version_path)
            
            # Remove metadata
            metadata_file = self.metadata_path / f"{registry_id}.json"
            if metadata_file.exists():
                metadata_file.unlink()
            
            # Remove from cache
            if registry_id in self._metadata_cache:
                del self._metadata_cache[registry_id]
            
            logger.info(f"Model deleted: {registry_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model {registry_id}: {e}")
            return False
    
    def get_model_performance(self, model_id: str, days: int = 30) -> Dict[str, Any]:
        """Get model performance metrics over time"""
        # This would integrate with the monitoring system
        # For now, return cached backtest metrics
        models = [m for m in self.list_models() if m.model_id == model_id]
        
        if not models:
            return {}
        
        latest = models[0]  # Already sorted by creation date
        
        return {
            "model_id": model_id,
            "latest_version": latest.version,
            "backtest_sharpe": latest.backtest_sharpe,
            "backtest_returns": latest.backtest_returns,
            "backtest_max_dd": latest.backtest_max_dd,
            "validation_score": latest.validation_score,
            "status": latest.status.value,
            "created_at": latest.created_at.isoformat()
        }
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _copy_artifacts(self, artifacts: ModelArtifacts, destination: Path) -> None:
        """Copy model artifacts to registry"""
        for attr_name, file_path in asdict(artifacts).items():
            if file_path and os.path.exists(file_path):
                dest_file = destination / f"{attr_name}.file"
                shutil.copy2(file_path, dest_file)
    
    def _cleanup_old_versions(self, model_id: str) -> None:
        """Remove old versions beyond the retention limit"""
        models = [m for m in self.list_models() if m.model_id == model_id]
        
        if len(models) <= self.max_versions_per_model:
            return
        
        # Keep production and staging models, remove oldest development/testing
        to_remove = []
        non_production = [
            m for m in models 
            if m.status in [ModelStatus.DEVELOPMENT, ModelStatus.TESTING]
        ]
        
        if len(non_production) > self.max_versions_per_model // 2:
            # Remove oldest non-production versions
            non_production.sort(key=lambda m: m.created_at)
            excess = len(non_production) - (self.max_versions_per_model // 2)
            to_remove = non_production[:excess]
        
        for metadata in to_remove:
            registry_id = f"{metadata.model_id}:{metadata.version}"
            self.delete_model(registry_id, force=True)
            logger.info(f"Cleaned up old version: {registry_id}")
    
    def _version_sort_key(self, version: str) -> Tuple[int, ...]:
        """Convert version string to sortable tuple"""
        try:
            return tuple(int(x) for x in version.split('.'))
        except ValueError:
            # Fallback for non-semantic versions
            return (0, 0, hash(version) % 1000)
    
    def _load_metadata_cache(self) -> None:
        """Load metadata cache from disk"""
        try:
            for metadata_file in self.metadata_path.glob("*.json"):
                registry_id = metadata_file.stem
                metadata = self.get_metadata(registry_id)
                if metadata:
                    self._metadata_cache[registry_id] = metadata
            
            logger.info(f"Loaded {len(self._metadata_cache)} models into cache")
            
        except Exception as e:
            logger.error(f"Error loading metadata cache: {e}")


def create_model_metadata(model_id: str,
                         name: str,
                         version: str,
                         model_type: ModelType,
                         created_by: str,
                         description: str,
                         input_features: List[str],
                         output_schema: Dict[str, str],
                         **kwargs) -> ModelMetadata:
    """Convenience function to create ModelMetadata"""
    
    return ModelMetadata(
        model_id=model_id,
        name=name,
        version=version,
        model_type=model_type,
        status=ModelStatus.DEVELOPMENT,
        created_at=datetime.utcnow(),
        created_by=created_by,
        description=description,
        framework=kwargs.get("framework", "unknown"),
        input_features=input_features,
        output_schema=output_schema,
        backtest_sharpe=kwargs.get("backtest_sharpe"),
        backtest_returns=kwargs.get("backtest_returns"),
        backtest_max_dd=kwargs.get("backtest_max_dd"),
        validation_score=kwargs.get("validation_score"),
        resource_requirements=kwargs.get("resource_requirements"),
        latency_requirements=kwargs.get("latency_requirements")
    )
