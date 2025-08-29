import pytest
from hypothesis import given, strategies as st, assume, settings
from datetime import UTC, datetime

from crowetrade.core.contracts import FeatureVector, Signal
from crowetrade.live.signal_agent import SignalAgent


class TestSignalAgent:
    """Unit tests for SignalAgent with property-based testing."""
    
    @pytest.mark.unit
    def test_signal_generation_with_positive_momentum(self, sample_feature_vector):
        """Test signal generation with positive momentum."""
        def model(values):
            return values.get("mom", 0), 0.02, 0.6
        
        agent = SignalAgent(
            model=model,
            policy_id="test_policy",
            gates={"prob_edge_min": 0.55, "sigma_max": 0.03}
        )
        
        signal = agent.infer(sample_feature_vector)
        assert signal is not None
        assert signal.instrument == sample_feature_vector.instrument
        assert signal.mu == sample_feature_vector.values["mom"]
        assert signal.prob_edge_pos == 0.6
    
    @pytest.mark.unit
    def test_signal_gating_probability_edge(self):
        """Test that signals are gated by probability edge threshold."""
        def model(values):
            return 0.01, 0.02, 0.45  # Below threshold
        
        agent = SignalAgent(
            model=model,
            policy_id="test_policy",
            gates={"prob_edge_min": 0.55, "sigma_max": 0.03}
        )
        
        fv = FeatureVector(
            instrument="TEST",
            asof=datetime.now(UTC),
            horizon="1d",
            values={"mom": 0.01},
            quality={"lag_ms": 50, "coverage": 1.0}
        )
        
        signal = agent.infer(fv)
        assert signal is None
    
    @pytest.mark.unit
    def test_signal_gating_sigma_max(self):
        """Test that signals are gated by sigma threshold."""
        def model(values):
            return 0.01, 0.05, 0.65  # Sigma above threshold
        
        agent = SignalAgent(
            model=model,
            policy_id="test_policy",
            gates={"prob_edge_min": 0.55, "sigma_max": 0.03}
        )
        
        fv = FeatureVector(
            instrument="TEST",
            asof=datetime.now(UTC),
            horizon="1d",
            values={"mom": 0.01},
            quality={"lag_ms": 50, "coverage": 1.0}
        )
        
        signal = agent.infer(fv)
        assert signal is None
    
    @pytest.mark.unit
    @given(
        mu=st.floats(min_value=-0.1, max_value=0.1, allow_nan=False),
        sigma=st.floats(min_value=0.001, max_value=0.1, allow_nan=False),
        prob=st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
    )
    @settings(max_examples=100)
    def test_signal_invariants(self, mu, sigma, prob):
        """Property-based test for signal generation invariants."""
        def model(values):
            return mu, sigma, prob
        
        agent = SignalAgent(
            model=model,
            policy_id="test_policy",
            gates={"prob_edge_min": 0.55, "sigma_max": 0.03}
        )
        
        fv = FeatureVector(
            instrument="TEST",
            asof=datetime.now(UTC),
            horizon="1d",
            values={"test": mu},
            quality={"lag_ms": 50, "coverage": 1.0}
        )
        
        signal = agent.infer(fv)
        
        # Invariant: Signal should only be generated if gates pass
        if prob >= 0.55 and sigma <= 0.03:
            assert signal is not None
            assert signal.mu == mu
            assert signal.sigma == sigma
            assert signal.prob_edge_pos == prob
            assert signal.policy_id == "test_policy"
        else:
            assert signal is None
    
    @pytest.mark.unit
    @given(
        instruments=st.lists(
            st.text(min_size=1, max_size=10, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
            min_size=1,
            max_size=10
        )
    )
    def test_batch_signal_processing(self, instruments):
        """Test processing multiple instruments in batch."""
        def model(values):
            return values.get("mom", 0), 0.02, 0.6
        
        agent = SignalAgent(
            model=model,
            policy_id="test_policy",
            gates={"prob_edge_min": 0.55, "sigma_max": 0.03}
        )
        
        signals = []
        for instrument in instruments:
            fv = FeatureVector(
                instrument=instrument,
                asof=datetime.now(UTC),
                horizon="1d",
                values={"mom": 0.01},
                quality={"lag_ms": 50, "coverage": 1.0}
            )
            signal = agent.infer(fv)
            if signal:
                signals.append(signal)
        
        # All generated signals should have correct instrument mapping
        for signal, instrument in zip(signals, instruments):
            assert signal.instrument == instrument