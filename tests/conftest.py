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
    """Sample signal for testing"""
    from crowetrade.core.contracts import Signal
    return Signal(
        instrument="TEST",
        signal=0.5,
        policy_id="test_policy",
        timestamp=datetime.utcnow()
    )


@pytest.fixture
def sample_feature_vector():
    """Sample feature vector for testing"""
    return {
        "momentum": 0.02,
        "volatility": 0.15,
        "volume": 1000000,
        "rsi": 65,
        "ma_ratio": 1.02
    }
