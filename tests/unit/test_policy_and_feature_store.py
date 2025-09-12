import tempfile
from pathlib import Path
import yaml
import pytest

from crowetrade.core.policy import PolicyRegistry
from crowetrade.feature_store.store import InMemoryFeatureStore


@pytest.mark.unit
def test_policy_registry_load():
    with tempfile.TemporaryDirectory() as d:
        policy_path = Path(d) / "p1.yaml"
        policy_path.write_text(
            yaml.safe_dump(
                {
                    "id": "p1",
                    "entry_gate": {"prob_edge_min": 0.6, "sigma_max": 0.02},
                    "sizing": {"lambda": 0.2},
                }
            ),
            encoding="utf-8",
        )
        reg = PolicyRegistry(d)
        pol = reg.load("p1")
        assert pol.id == "p1"
        assert pol.gates["prob_edge_min"] == 0.6


@pytest.mark.unit
def test_in_memory_feature_store_basic():
    fs = InMemoryFeatureStore()
    fs.put("AAPL", {"f1": 1.0, "f2": 2.0}, version=3)
    rec = fs.get("AAPL")
    assert rec is not None
    assert rec.version == 3
    assert rec.values["f1"] == 1.0
    batch = fs.batch_get(["AAPL", "MSFT"])  # missing key returns None
    assert batch["AAPL"].version == 3
    assert batch["MSFT"] is None
