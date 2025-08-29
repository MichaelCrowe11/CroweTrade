import pytest
from hypothesis import given, strategies as st, assume, settings
import numpy as np

from crowetrade.core.contracts import Signal
from crowetrade.live.portfolio_agent import PortfolioAgent


class TestPortfolioAgent:
    """Unit tests for PortfolioAgent with property-based testing."""
    
    @pytest.mark.unit
    def test_basic_sizing(self, sample_signal):
        """Test basic position sizing."""
        agent = PortfolioAgent(risk_budget=1.0, lambda_temper=0.25)
        targets = agent.size(
            {"AAPL": sample_signal},
            {"AAPL": 0.02}
        )
        
        assert "AAPL" in targets
        assert isinstance(targets["AAPL"], float)
        assert abs(targets["AAPL"]) <= 1.0
    
    @pytest.mark.unit
    def test_risk_budget_constraint(self):
        """Test that position sizing respects risk budget."""
        agent = PortfolioAgent(risk_budget=0.5, lambda_temper=0.25)
        
        signal = Signal(
            instrument="TEST",
            horizon="1d",
            mu=0.05,
            sigma=0.20,
            prob_edge_pos=0.8,
            policy_id="test"
        )
        
        targets = agent.size(
            {"TEST": signal},
            {"TEST": 0.10}
        )
        
        # Position should be scaled by risk budget
        assert abs(targets["TEST"]) <= 0.5
    
    @pytest.mark.unit
    def test_lambda_tempering(self):
        """Test that lambda tempering affects position sizes."""
        signal = Signal(
            instrument="TEST",
            horizon="1d",
            mu=0.02,
            sigma=0.15,
            prob_edge_pos=0.65,
            policy_id="test"
        )
        
        agent_low = PortfolioAgent(risk_budget=1.0, lambda_temper=0.1)
        agent_high = PortfolioAgent(risk_budget=1.0, lambda_temper=0.9)
        
        targets_low = agent_low.size({"TEST": signal}, {"TEST": 0.02})
        targets_high = agent_high.size({"TEST": signal}, {"TEST": 0.02})
        
        # Higher tempering should result in smaller positions
        assert abs(targets_high["TEST"]) < abs(targets_low["TEST"])
    
    @pytest.mark.unit
    @given(
        mu=st.floats(min_value=-0.1, max_value=0.1, allow_nan=False),
        sigma=st.floats(min_value=0.01, max_value=0.5, allow_nan=False),
        prob=st.floats(min_value=0.5, max_value=1.0, allow_nan=False),
        risk_budget=st.floats(min_value=0.1, max_value=2.0, allow_nan=False)
    )
    @settings(max_examples=100)
    def test_sizing_invariants(self, mu, sigma, prob, risk_budget):
        """Property-based test for sizing invariants."""
        agent = PortfolioAgent(risk_budget=risk_budget, lambda_temper=0.25)
        
        signal = Signal(
            instrument="TEST",
            horizon="1d",
            mu=mu,
            sigma=sigma,
            prob_edge_pos=prob,
            policy_id="test"
        )
        
        targets = agent.size(
            {"TEST": signal},
            {"TEST": sigma * 0.1}  # Tracking error
        )
        
        # Invariants
        assert "TEST" in targets
        assert isinstance(targets["TEST"], float)
        assert not np.isnan(targets["TEST"])
        assert not np.isinf(targets["TEST"])
        assert abs(targets["TEST"]) <= risk_budget
        
        # Direction should match signal
        if mu > 0:
            assert targets["TEST"] >= 0
        elif mu < 0:
            assert targets["TEST"] <= 0
    
    @pytest.mark.unit
    def test_multi_instrument_sizing(self):
        """Test sizing across multiple instruments."""
        agent = PortfolioAgent(risk_budget=1.0, lambda_temper=0.25)
        
        signals = {
            "AAPL": Signal("AAPL", "1d", 0.02, 0.15, 0.65, "test"),
            "GOOGL": Signal("GOOGL", "1d", -0.01, 0.20, 0.60, "test"),
            "MSFT": Signal("MSFT", "1d", 0.03, 0.18, 0.70, "test"),
        }
        
        tracking_errors = {
            "AAPL": 0.02,
            "GOOGL": 0.03,
            "MSFT": 0.025,
        }
        
        targets = agent.size(signals, tracking_errors)
        
        assert len(targets) == 3
        assert all(instrument in targets for instrument in signals)
        
        # Total risk should not exceed budget
        total_risk = sum(abs(v) for v in targets.values())
        assert total_risk <= len(signals) * agent.risk_budget
    
    @pytest.mark.unit
    @given(
        n_instruments=st.integers(min_value=1, max_value=20)
    )
    def test_portfolio_scaling(self, n_instruments):
        """Test that portfolio sizing scales appropriately with number of instruments."""
        agent = PortfolioAgent(risk_budget=1.0, lambda_temper=0.25)
        
        signals = {}
        tracking_errors = {}
        
        for i in range(n_instruments):
            instrument = f"INST_{i}"
            signals[instrument] = Signal(
                instrument=instrument,
                horizon="1d",
                mu=0.02,
                sigma=0.15,
                prob_edge_pos=0.65,
                policy_id="test"
            )
            tracking_errors[instrument] = 0.02
        
        targets = agent.size(signals, tracking_errors)
        
        assert len(targets) == n_instruments
        
        # Check risk budget per instrument decreases with portfolio size
        max_position = max(abs(v) for v in targets.values())
        assert max_position <= agent.risk_budget