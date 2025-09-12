import time
from pathlib import Path
from crowetrade.feature_store.store import InMemoryFeatureStore, DiskFeatureStore
from crowetrade.core.policy import PolicyRegistry


def test_in_memory_feature_store_versioning():
    store = InMemoryFeatureStore()
    r1 = store.put("AAPL", {"feat": 1.0})
    r2 = store.put("AAPL", {"feat": 2.0})
    assert r2.version == r1.version + 1
    last = store.get("AAPL")
    assert last.values["feat"] == 2.0
    hist = store.history("AAPL")
    assert len(hist) == 2
    assert hist[0].version == 1


def test_disk_feature_store_persistence(tmp_path: Path):
    store = DiskFeatureStore(tmp_path)
    r1 = store.put("BTC", {"x": 10.0})
    r2 = store.put("BTC", {"x": 15.0})
    assert r2.version == r1.version + 1
    # Recreate store to ensure persistence
    store2 = DiskFeatureStore(tmp_path)
    last = store2.get("BTC")
    assert last and last.values["x"] == 15.0
    hist = store2.history("BTC")
    assert len(hist) == 2


def test_policy_hot_reload(tmp_path: Path):
    policies = tmp_path / "pol"
    policies.mkdir()
    pol_file = policies / "test.yaml"
    pol_file.write_text("id: test_policy\nentry_gate:\n  prob_edge_min: 0.5\n", encoding="utf-8")
    reg = PolicyRegistry(policies)
    p1 = reg.load("test_policy")
    h1 = reg.policy_hash("test_policy")
    assert h1 is not None
    # Modify policy
    time.sleep(0.01)  # ensure mtime difference
    pol_file.write_text("id: test_policy\nentry_gate:\n  prob_edge_min: 0.6\n", encoding="utf-8")
    changed = reg.maybe_reload("test_policy")
    assert changed is True
    h2 = reg.policy_hash("test_policy")
    assert h2 != h1
    p2 = reg.load("test_policy")
    assert p2.gates["prob_edge_min"] == 0.6