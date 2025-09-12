"""Persistent versioned feature store with Parquet backend"""
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import logging

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class FeatureVersion:
    """Versioned feature with lineage tracking"""
    feature_id: str
    version: int
    timestamp: float
    data: Dict[str, Any]
    prev_version: Optional[int] = None
    feature_spec_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    ttl_seconds: Optional[int] = None
    
    def is_expired(self) -> bool:
        """Check if feature has expired based on TTL"""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.timestamp > self.ttl_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)


class PersistentFeatureStore:
    """Feature store with Parquet persistence and versioning"""
    
    def __init__(
        self,
        storage_path: str = "./feature_store",
        max_versions: int = 100,
        auto_persist: bool = True,
        ttl_seconds: Optional[int] = 86400  # 24 hours default
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.max_versions = max_versions
        self.auto_persist = auto_persist
        self.default_ttl = ttl_seconds
        
        # In-memory cache
        self.features: Dict[str, List[FeatureVersion]] = {}
        self.version_counter: Dict[str, int] = {}
        
        # Load existing data
        self._load_from_disk()
    
    def put(
        self,
        feature_id: str,
        data: Dict[str, Any],
        feature_spec: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FeatureVersion:
        """Store a new version of a feature"""
        
        # Calculate feature spec hash if provided
        spec_hash = None
        if feature_spec:
            spec_hash = hashlib.sha256(
                json.dumps(feature_spec, sort_keys=True).encode()
            ).hexdigest()[:16]
        
        # Get next version number
        prev_version = self.version_counter.get(feature_id, 0)
        new_version = prev_version + 1
        self.version_counter[feature_id] = new_version
        
        # Create versioned feature
        feature = FeatureVersion(
            feature_id=feature_id,
            version=new_version,
            timestamp=time.time(),
            data=data,
            prev_version=prev_version if prev_version > 0 else None,
            feature_spec_hash=spec_hash,
            metadata=metadata or {},
            ttl_seconds=ttl_seconds or self.default_ttl
        )
        
        # Store in memory
        if feature_id not in self.features:
            self.features[feature_id] = []
        self.features[feature_id].append(feature)
        
        # Trim old versions
        if len(self.features[feature_id]) > self.max_versions:
            self.features[feature_id] = self.features[feature_id][-self.max_versions:]
        
        # Persist if enabled
        if self.auto_persist:
            self._persist_feature(feature)
        
        logger.debug(f"Stored feature {feature_id} v{new_version} with hash {spec_hash}")
        return feature
    
    def get(
        self,
        feature_id: str,
        version: Optional[int] = None,
        include_expired: bool = False
    ) -> Optional[FeatureVersion]:
        """Get a specific version or latest version of a feature"""
        
        if feature_id not in self.features:
            return None
        
        versions = self.features[feature_id]
        if not versions:
            return None
        
        # Get specific version or latest
        if version is not None:
            for feat in versions:
                if feat.version == version:
                    if not include_expired and feat.is_expired():
                        return None
                    return feat
            return None
        else:
            # Return latest non-expired version
            for feat in reversed(versions):
                if include_expired or not feat.is_expired():
                    return feat
            return None
    
    def get_history(
        self,
        feature_id: str,
        limit: int = 10,
        include_expired: bool = False
    ) -> List[FeatureVersion]:
        """Get version history for a feature"""
        
        if feature_id not in self.features:
            return []
        
        versions = self.features[feature_id]
        if not include_expired:
            versions = [v for v in versions if not v.is_expired()]
        
        return versions[-limit:] if limit else versions
    
    def query(
        self,
        feature_pattern: Optional[str] = None,
        spec_hash: Optional[str] = None,
        min_version: Optional[int] = None,
        since_timestamp: Optional[float] = None
    ) -> List[FeatureVersion]:
        """Query features by various criteria"""
        
        results = []
        
        for feature_id, versions in self.features.items():
            # Filter by pattern
            if feature_pattern and feature_pattern not in feature_id:
                continue
            
            for feat in versions:
                # Filter by spec hash
                if spec_hash and feat.feature_spec_hash != spec_hash:
                    continue
                
                # Filter by version
                if min_version and feat.version < min_version:
                    continue
                
                # Filter by timestamp
                if since_timestamp and feat.timestamp < since_timestamp:
                    continue
                
                # Skip expired
                if feat.is_expired():
                    continue
                
                results.append(feat)
        
        return results
    
    def get_lineage(self, feature_id: str, version: int) -> List[FeatureVersion]:
        """Get the lineage chain for a specific version"""
        
        lineage = []
        current = self.get(feature_id, version, include_expired=True)
        
        while current:
            lineage.append(current)
            if current.prev_version:
                current = self.get(feature_id, current.prev_version, include_expired=True)
            else:
                break
        
        return list(reversed(lineage))
    
    def _persist_feature(self, feature: FeatureVersion) -> None:
        """Persist a feature to Parquet"""
        
        if not PARQUET_AVAILABLE:
            logger.warning("PyArrow not available, skipping persistence")
            return
        
        # Create filename with date partition
        date_str = datetime.fromtimestamp(feature.timestamp).strftime("%Y%m%d")
        filename = self.storage_path / f"features_{date_str}.parquet"
        
        # Convert to DataFrame-like structure with consistent types
        data = {
            'feature_id': [feature.feature_id],
            'version': [feature.version],
            'timestamp': [feature.timestamp],
            'data': [json.dumps(feature.data)],
            'prev_version': [feature.prev_version if feature.prev_version is not None else -1],
            'feature_spec_hash': [feature.feature_spec_hash if feature.feature_spec_hash else ''],
            'metadata': [json.dumps(feature.metadata)],
            'ttl_seconds': [feature.ttl_seconds if feature.ttl_seconds is not None else -1]
        }
        
        # Define schema for consistency
        schema = pa.schema([
            ('feature_id', pa.string()),
            ('version', pa.int64()),
            ('timestamp', pa.float64()),
            ('data', pa.string()),
            ('prev_version', pa.int64()),
            ('feature_spec_hash', pa.string()),
            ('metadata', pa.string()),
            ('ttl_seconds', pa.int64())
        ])
        
        table = pa.table(data, schema=schema)
        
        # Append to existing file or create new
        if filename.exists():
            existing = pq.read_table(filename)
            combined = pa.concat_tables([existing, table])
            pq.write_table(combined, filename)
        else:
            pq.write_table(table, filename)
    
    def _load_from_disk(self) -> None:
        """Load features from Parquet files"""
        
        if not PARQUET_AVAILABLE:
            logger.info("PyArrow not available, skipping disk load")
            return
        
        parquet_files = list(self.storage_path.glob("features_*.parquet"))
        
        for file in parquet_files:
            try:
                table = pq.read_table(file)
                df = table.to_pandas()
                
                for _, row in df.iterrows():
                    # Convert sentinel values back to None
                    prev_version = row['prev_version'] if row['prev_version'] != -1 else None
                    spec_hash = row['feature_spec_hash'] if row['feature_spec_hash'] else None
                    ttl = row['ttl_seconds'] if row['ttl_seconds'] != -1 else None
                    
                    feature = FeatureVersion(
                        feature_id=row['feature_id'],
                        version=row['version'],
                        timestamp=row['timestamp'],
                        data=json.loads(row['data']),
                        prev_version=prev_version,
                        feature_spec_hash=spec_hash,
                        metadata=json.loads(row['metadata']) if row['metadata'] else {},
                        ttl_seconds=ttl
                    )
                    
                    # Update in-memory state
                    if feature.feature_id not in self.features:
                        self.features[feature.feature_id] = []
                    self.features[feature.feature_id].append(feature)
                    
                    # Update version counter
                    if feature.feature_id not in self.version_counter:
                        self.version_counter[feature.feature_id] = 0
                    self.version_counter[feature.feature_id] = max(
                        self.version_counter[feature.feature_id],
                        feature.version
                    )
                
                logger.info(f"Loaded {len(df)} features from {file}")
                
            except Exception as e:
                logger.error(f"Failed to load {file}: {e}")
    
    def cleanup_expired(self) -> int:
        """Remove expired features from memory"""
        
        removed = 0
        for feature_id in list(self.features.keys()):
            versions = self.features[feature_id]
            active = [v for v in versions if not v.is_expired()]
            removed += len(versions) - len(active)
            
            if active:
                self.features[feature_id] = active
            else:
                del self.features[feature_id]
        
        logger.info(f"Cleaned up {removed} expired features")
        return removed
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics"""
        
        total_features = sum(len(v) for v in self.features.values())
        expired = sum(
            1 for versions in self.features.values()
            for v in versions if v.is_expired()
        )
        
        return {
            'unique_features': len(self.features),
            'total_versions': total_features,
            'expired_versions': expired,
            'active_versions': total_features - expired,
            'storage_path': str(self.storage_path),
            'parquet_available': PARQUET_AVAILABLE
        }