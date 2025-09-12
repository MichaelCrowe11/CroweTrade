import tempfile
import json
import numpy as np
import pytest

from crowetrade.monitoring.audit import AuditLogger
from crowetrade.optim.prox_mvo import prox_mvo, MVOConfig


@pytest.mark.unit
def test_audit_logger_appends():
    with tempfile.TemporaryDirectory() as d:
        p = f"{d}/audit.jsonl"
        logger = AuditLogger(p)
        logger.log("test.event", {"x": 1})
        logger.log("test.event", {"x": 2})
        lines = open(p, "r", encoding="utf-8").read().strip().splitlines()
        assert len(lines) == 2
        rec1 = json.loads(lines[0])
        rec2 = json.loads(lines[1])
        assert rec1["event"] == "test.event"
        assert rec1["x"] == 1
        assert rec2["x"] == 2


@pytest.mark.unit
def test_prox_mvo_converges_simple():
    rng = np.random.default_rng(0)
    n = 5
    Sigma = rng.normal(size=(n, n))
    Sigma = Sigma @ Sigma.T + 0.1 * np.eye(n)
    mu = rng.normal(size=n)
    w = prox_mvo(mu, Sigma, MVOConfig(lam=0.5, step=0.05, iters=500))
    # Basic sanity: finite, small gradient
    grad = Sigma @ w - 0.5 * mu
    assert np.linalg.norm(grad) < 1e-2
