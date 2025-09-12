"""Tests for persistent feature store with versioning"""
import json
import time
import tempfile
import shutil
from pathlib import Path
import pytest

from crowetrade.data.feature_store import PersistentFeatureStore, FeatureVersion


class TestPersistentFeatureStore:
    """Test persistent feature store functionality"""
    
    @pytest.fixture
    def temp_store(self):
        """Create a temporary feature store"""
        temp_dir = tempfile.mkdtemp()
        store = PersistentFeatureStore(
            storage_path=temp_dir,
            max_versions=10,
            ttl_seconds=3600
        )
        yield store
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_put_and_get(self, temp_store):
        """Test basic put and get operations"""
        # Store a feature
        feature_data = {'value': 100, 'source': 'test'}
        version = temp_store.put('feature1', feature_data)
        
        assert version.version == 1
        assert version.feature_id == 'feature1'
        assert version.data == feature_data
        assert version.prev_version is None
        
        # Retrieve the feature
        retrieved = temp_store.get('feature1')
        assert retrieved is not None
        assert retrieved.version == 1
        assert retrieved.data == feature_data
    
    def test_versioning(self, temp_store):
        """Test version tracking"""
        # Store multiple versions
        v1 = temp_store.put('feature2', {'value': 1})
        v2 = temp_store.put('feature2', {'value': 2})
        v3 = temp_store.put('feature2', {'value': 3})
        
        assert v1.version == 1
        assert v2.version == 2
        assert v2.prev_version == 1
        assert v3.version == 3
        assert v3.prev_version == 2
        
        # Get specific version
        retrieved_v2 = temp_store.get('feature2', version=2)
        assert retrieved_v2.data == {'value': 2}
        
        # Get latest version
        latest = temp_store.get('feature2')
        assert latest.version == 3
        assert latest.data == {'value': 3}
    
    def test_feature_spec_hash(self, temp_store):
        """Test feature specification hashing"""
        spec = {'type': 'numeric', 'range': [0, 100]}
        
        v1 = temp_store.put('feature3', {'value': 50}, feature_spec=spec)
        assert v1.feature_spec_hash is not None
        
        # Same spec should produce same hash
        v2 = temp_store.put('feature3', {'value': 60}, feature_spec=spec)
        assert v2.feature_spec_hash == v1.feature_spec_hash
        
        # Different spec should produce different hash
        spec2 = {'type': 'categorical', 'values': ['A', 'B']}
        v3 = temp_store.put('feature3', {'value': 'A'}, feature_spec=spec2)
        assert v3.feature_spec_hash != v1.feature_spec_hash
    
    def test_ttl_expiration(self, temp_store):
        """Test TTL expiration"""
        # Create feature with short TTL
        temp_store.put('feature4', {'value': 100}, ttl_seconds=0.1)
        
        # Should be available immediately
        assert temp_store.get('feature4') is not None
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Should be expired
        assert temp_store.get('feature4') is None
        
        # Should be available with include_expired
        assert temp_store.get('feature4', include_expired=True) is not None
    
    def test_get_history(self, temp_store):
        """Test version history retrieval"""
        # Create multiple versions
        for i in range(5):
            temp_store.put('feature5', {'value': i})
        
        # Get full history
        history = temp_store.get_history('feature5')
        assert len(history) == 5
        assert [h.data['value'] for h in history] == [0, 1, 2, 3, 4]
        
        # Get limited history
        limited = temp_store.get_history('feature5', limit=3)
        assert len(limited) == 3
        assert [h.data['value'] for h in limited] == [2, 3, 4]
    
    def test_query(self, temp_store):
        """Test query functionality"""
        # Add various features
        spec1 = {'type': 'A'}
        spec2 = {'type': 'B'}
        
        temp_store.put('model_v1_feat1', {'value': 1}, feature_spec=spec1)
        temp_store.put('model_v1_feat2', {'value': 2}, feature_spec=spec1)
        temp_store.put('model_v2_feat1', {'value': 3}, feature_spec=spec2)
        
        # Query by pattern
        model_v1 = temp_store.query(feature_pattern='model_v1')
        assert len(model_v1) == 2
        
        # Query by spec hash
        spec1_hash = model_v1[0].feature_spec_hash
        same_spec = temp_store.query(spec_hash=spec1_hash)
        assert len(same_spec) == 2
    
    def test_lineage(self, temp_store):
        """Test lineage tracking"""
        # Create version chain
        v1 = temp_store.put('feature6', {'value': 1})
        v2 = temp_store.put('feature6', {'value': 2})
        v3 = temp_store.put('feature6', {'value': 3})
        
        # Get lineage for v3
        lineage = temp_store.get_lineage('feature6', 3)
        assert len(lineage) == 3
        assert [l.version for l in lineage] == [1, 2, 3]
        assert [l.data['value'] for l in lineage] == [1, 2, 3]
    
    def test_cleanup_expired(self, temp_store):
        """Test cleanup of expired features"""
        # Create features with short TTL
        temp_store.put('expire1', {'value': 1}, ttl_seconds=0.1)
        temp_store.put('expire2', {'value': 2}, ttl_seconds=0.1)
        temp_store.put('keep', {'value': 3}, ttl_seconds=3600)
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Cleanup
        removed = temp_store.cleanup_expired()
        assert removed == 2
        
        # Check remaining
        stats = temp_store.get_stats()
        assert stats['active_versions'] == 1
        assert temp_store.get('keep') is not None
    
    def test_max_versions_trim(self, temp_store):
        """Test that old versions are trimmed"""
        temp_store.max_versions = 3
        
        # Create more than max versions
        for i in range(5):
            temp_store.put('feature7', {'value': i})
        
        # Should only keep last 3 versions
        history = temp_store.get_history('feature7')
        assert len(history) == 3
        assert [h.data['value'] for h in history] == [2, 3, 4]
        
        # Check that version 1 and 2 are gone
        assert temp_store.get('feature7', version=1) is None
        assert temp_store.get('feature7', version=2) is None
        assert temp_store.get('feature7', version=3) is not None