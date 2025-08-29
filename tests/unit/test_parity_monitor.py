"""Tests for InMemoryFeatureStore and ParityMonitor"""

import pytest
import math
from typing import Dict, List
from crowetrade.services.data_service.parity_monitor import (
    InMemoryFeatureStore,
    ParityMonitor,
    psi,
    _bin_edges
)


class TestInMemoryFeatureStore:
    """Test suite for InMemoryFeatureStore"""
    
    def test_init(self):
        """Test initialization of feature store"""
        store = InMemoryFeatureStore()
        assert store._data == {}
    
    def test_put_single_vector(self):
        """Test storing a single feature vector"""
        store = InMemoryFeatureStore()
        vector = {"f1": 1.0, "f2": 2.0, "f3": 3.0}
        store.put("key1", vector, 1000)
        
        assert "key1" in store._data
        assert store._data["key1"] == (1000, vector)
    
    def test_put_overwrites_existing(self):
        """Test that put overwrites existing entries"""
        store = InMemoryFeatureStore()
        store.put("key1", {"f1": 1.0}, 1000)
        store.put("key1", {"f1": 2.0}, 2000)
        
        assert store._data["key1"] == (2000, {"f1": 2.0})
    
    def test_get_full_vector(self):
        """Test retrieving a full feature vector"""
        store = InMemoryFeatureStore()
        vector = {"f1": 1.0, "f2": 2.0, "f3": 3.0}
        store.put("key1", vector, 1000)
        
        ts, retrieved = store.get("key1")
        assert ts == 1000
        assert retrieved == vector
    
    def test_get_with_field_selection(self):
        """Test retrieving specific fields from a vector"""
        store = InMemoryFeatureStore()
        vector = {"f1": 1.0, "f2": 2.0, "f3": 3.0}
        store.put("key1", vector, 1000)
        
        ts, retrieved = store.get("key1", fields=["f1", "f3"])
        assert ts == 1000
        assert retrieved == {"f1": 1.0, "f3": 3.0}
    
    def test_get_nonexistent_key(self):
        """Test retrieving a nonexistent key returns default values"""
        store = InMemoryFeatureStore()
        ts, retrieved = store.get("nonexistent")
        assert ts == 0
        assert retrieved == {}
    
    def test_get_with_nonexistent_fields(self):
        """Test retrieving nonexistent fields returns empty result"""
        store = InMemoryFeatureStore()
        store.put("key1", {"f1": 1.0}, 1000)
        
        ts, retrieved = store.get("key1", fields=["f2", "f3"])
        assert ts == 1000
        assert retrieved == {}
    
    def test_batch_get_multiple_keys(self):
        """Test batch retrieval of multiple keys"""
        store = InMemoryFeatureStore()
        store.put("key1", {"f1": 1.0}, 1000)
        store.put("key2", {"f2": 2.0}, 2000)
        store.put("key3", {"f3": 3.0}, 3000)
        
        result = store.batch_get(["key1", "key2", "key3"])
        
        assert result["key1"] == (1000, {"f1": 1.0})
        assert result["key2"] == (2000, {"f2": 2.0})
        assert result["key3"] == (3000, {"f3": 3.0})
    
    def test_batch_get_with_nonexistent_keys(self):
        """Test batch retrieval with some nonexistent keys"""
        store = InMemoryFeatureStore()
        store.put("key1", {"f1": 1.0}, 1000)
        
        result = store.batch_get(["key1", "nonexistent"])
        
        assert result["key1"] == (1000, {"f1": 1.0})
        assert result["nonexistent"] == (0, {})
    
    def test_immutability_of_retrieved_data(self):
        """Test that retrieved data doesn't affect stored data"""
        store = InMemoryFeatureStore()
        original = {"f1": 1.0}
        store.put("key1", original, 1000)
        
        ts, retrieved = store.get("key1")
        retrieved["f2"] = 2.0
        
        ts2, retrieved2 = store.get("key1")
        assert "f2" not in retrieved2
        assert retrieved2 == {"f1": 1.0}


class TestBinEdges:
    """Test suite for _bin_edges function"""
    
    def test_empty_values(self):
        """Test bin edges for empty values"""
        edges = _bin_edges([], bins=10)
        assert edges == [0.0]
    
    def test_single_value(self):
        """Test bin edges for single repeated value"""
        edges = _bin_edges([5.0, 5.0, 5.0], bins=3)
        assert len(edges) == 4
        assert edges[0] == 5.0
        assert edges[-1] > 5.0
    
    def test_uniform_distribution(self):
        """Test bin edges for uniform distribution"""
        values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        edges = _bin_edges(values, bins=5)
        
        assert len(edges) == 6
        assert edges[0] == 0.0
        assert edges[-1] == 5.0
        assert all(edges[i] < edges[i+1] for i in range(len(edges)-1))
    
    def test_custom_bin_count(self):
        """Test different bin counts"""
        values = list(range(100))
        
        edges_10 = _bin_edges(values, bins=10)
        edges_20 = _bin_edges(values, bins=20)
        
        assert len(edges_10) == 11
        assert len(edges_20) == 21


class TestPSI:
    """Test suite for PSI calculation"""
    
    def test_identical_distributions(self):
        """Test PSI for identical distributions"""
        ref = [1.0, 2.0, 3.0, 4.0, 5.0]
        cur = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        psi_val = psi(ref, cur, bins=5)
        assert psi_val < 0.01
    
    def test_empty_distributions(self):
        """Test PSI for empty distributions"""
        assert psi([], [], bins=10) == 0.0
        assert psi([1.0], [], bins=10) > 0
        assert psi([], [1.0], bins=10) > 0
    
    def test_shifted_distribution(self):
        """Test PSI for shifted distribution"""
        ref = [1.0, 2.0, 3.0, 4.0, 5.0]
        cur = [3.0, 4.0, 5.0, 6.0, 7.0]
        
        psi_val = psi(ref, cur, bins=5)
        assert psi_val > 0.1
    
    def test_scaled_distribution(self):
        """Test PSI for scaled distribution"""
        ref = [1.0, 2.0, 3.0, 4.0, 5.0]
        cur = [10.0, 20.0, 30.0, 40.0, 50.0]
        
        psi_val = psi(ref, cur, bins=5)
        assert psi_val > 0.2
    
    def test_normal_vs_uniform(self):
        """Test PSI between normal and uniform distributions"""
        import random
        random.seed(42)
        
        ref = [random.gauss(0, 1) for _ in range(1000)]
        cur = [random.uniform(-3, 3) for _ in range(1000)]
        
        psi_val = psi(ref, cur, bins=10)
        assert psi_val > 0.05
    
    def test_large_sample_stability(self):
        """Test PSI stability with large samples"""
        import random
        random.seed(42)
        
        ref = [random.gauss(0, 1) for _ in range(10000)]
        cur = [random.gauss(0, 1.1) for _ in range(10000)]
        
        psi_val = psi(ref, cur, bins=20)
        assert 0.01 < psi_val < 0.1


class TestParityMonitor:
    """Test suite for ParityMonitor"""
    
    def test_init_default_threshold(self):
        """Test initialization with default threshold"""
        monitor = ParityMonitor()
        assert monitor.threshold == 0.1
    
    def test_init_custom_threshold(self):
        """Test initialization with custom threshold"""
        monitor = ParityMonitor(threshold=0.2)
        assert monitor.threshold == 0.2
    
    def test_check_identical_distributions(self):
        """Test check with identical feature distributions"""
        monitor = ParityMonitor()
        
        ref_vecs = [
            {"f1": 1.0, "f2": 2.0},
            {"f1": 2.0, "f2": 3.0},
            {"f1": 3.0, "f2": 4.0}
        ]
        cur_vecs = ref_vecs.copy()
        
        psis = monitor.check(ref_vecs, cur_vecs)
        
        assert "f1" in psis
        assert "f2" in psis
        assert all(v < 0.01 for v in psis.values())
    
    def test_check_drifted_distributions(self):
        """Test check with drifted feature distributions"""
        monitor = ParityMonitor()
        
        ref_vecs = [
            {"f1": 1.0, "f2": 2.0},
            {"f1": 2.0, "f2": 3.0},
            {"f1": 3.0, "f2": 4.0}
        ]
        cur_vecs = [
            {"f1": 5.0, "f2": 2.5},
            {"f1": 6.0, "f2": 3.5},
            {"f1": 7.0, "f2": 4.5}
        ]
        
        psis = monitor.check(ref_vecs, cur_vecs)
        
        assert psis["f1"] > psis["f2"]
        assert psis["f1"] > 0.1
    
    def test_check_missing_features(self):
        """Test check with missing features in some vectors"""
        monitor = ParityMonitor()
        
        ref_vecs = [
            {"f1": 1.0, "f2": 2.0},
            {"f1": 2.0},
            {"f1": 3.0, "f2": 4.0}
        ]
        cur_vecs = [
            {"f1": 1.0},
            {"f1": 2.0, "f2": 3.0},
            {"f2": 4.0}
        ]
        
        psis = monitor.check(ref_vecs, cur_vecs)
        
        assert "f1" in psis
        assert "f2" in psis
        assert all(v >= 0 for v in psis.values())
    
    def test_breached_no_drift(self):
        """Test breached with no drift"""
        monitor = ParityMonitor(threshold=0.1)
        
        psis = {"f1": 0.05, "f2": 0.08, "f3": 0.02}
        breached = monitor.breached(psis)
        
        assert breached == []
    
    def test_breached_with_drift(self):
        """Test breached with some features drifted"""
        monitor = ParityMonitor(threshold=0.1)
        
        psis = {"f1": 0.15, "f2": 0.08, "f3": 0.25}
        breached = monitor.breached(psis)
        
        assert set(breached) == {"f1", "f3"}
    
    def test_breached_all_drifted(self):
        """Test breached with all features drifted"""
        monitor = ParityMonitor(threshold=0.1)
        
        psis = {"f1": 0.15, "f2": 0.18, "f3": 0.25}
        breached = monitor.breached(psis)
        
        assert set(breached) == {"f1", "f2", "f3"}
    
    def test_integration_with_feature_store(self):
        """Test integration between FeatureStore and ParityMonitor"""
        store = InMemoryFeatureStore()
        monitor = ParityMonitor(threshold=0.15)
        
        for i in range(10):
            store.put(f"ref_{i}", {"f1": i * 1.0, "f2": i * 2.0}, 1000 + i)
        
        for i in range(10):
            store.put(f"cur_{i}", {"f1": (i + 5) * 1.0, "f2": i * 2.1}, 2000 + i)
        
        ref_keys = [f"ref_{i}" for i in range(10)]
        cur_keys = [f"cur_{i}" for i in range(10)]
        
        ref_data = store.batch_get(ref_keys)
        cur_data = store.batch_get(cur_keys)
        
        ref_vecs = [v for _, v in ref_data.values()]
        cur_vecs = [v for _, v in cur_data.values()]
        
        psis = monitor.check(ref_vecs, cur_vecs)
        breached = monitor.breached(psis)
        
        assert "f1" in psis
        assert "f2" in psis
        assert psis["f1"] > psis["f2"]
        assert "f1" in breached


@pytest.mark.parametrize("ref_size,cur_size,expected_drift", [
    (100, 100, False),
    (100, 500, True),
    (500, 100, True),
])
def test_psi_sample_size_sensitivity(ref_size, cur_size, expected_drift):
    """Test PSI sensitivity to sample size differences"""
    import random
    random.seed(42)
    
    ref = [random.gauss(0, 1) for _ in range(ref_size)]
    cur = [random.gauss(0, 1) for _ in range(cur_size)]
    
    psi_val = psi(ref, cur, bins=10)
    
    if expected_drift:
        assert psi_val > 0.05
    else:
        assert psi_val < 0.1