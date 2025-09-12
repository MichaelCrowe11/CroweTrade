import sys
from pathlib import Path
from datetime import datetime

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def sample_signal():
    """Sample signal for testing (updated to new Signal contract)."""
    from crowetrade.core.contracts import Signal
    return Signal(
        instrument="AAPL",
        horizon="1d",
        mu=0.05,
        sigma=0.20,
        prob_edge_pos=0.7,
        policy_id="test_policy",
    )


@pytest.fixture
def sample_feature_vector():
    """Sample FeatureVector matching production contract."""
    from crowetrade.core.contracts import FeatureVector
    return FeatureVector(
        instrument="AAPL",
        asof=datetime.utcnow(),
        horizon="1d",
        values={
            "mom": 0.02,
            "vol": 0.15,
            "volume": 1_000_000,
            "rsi": 65,
            "ma_ratio": 1.02,
        },
        quality={"lag_ms": 50, "coverage": 1.0},
    )
