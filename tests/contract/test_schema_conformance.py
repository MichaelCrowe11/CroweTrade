import json
from pathlib import Path
import pytest
from hypothesis import given, strategies as st, settings
from datetime import UTC, datetime
from dataclasses import asdict

import jsonschema
from jsonschema import validate, ValidationError

from crowetrade.core.contracts import FeatureVector, Signal, TargetPosition, Fill

ROOT = Path(__file__).resolve().parents[2]
SCHEMAS_DIR = ROOT / "specs" / "schemas"
EXAMPLES_DIR = ROOT / "specs" / "examples"


class TestSchemaConformance:
    """Contract tests for schema conformance."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Load schemas for testing."""
        self.schemas = {}
        for schema_file in SCHEMAS_DIR.glob("*.schema.json"):
            schema_name = schema_file.stem.replace(".schema", "")
            with open(schema_file) as f:
                self.schemas[schema_name] = json.load(f)
    
    @pytest.mark.contract
    def test_feature_vector_schema_valid(self):
        """Test that valid feature vectors conform to schema."""
        fv = FeatureVector(
            instrument="AAPL",
            asof=datetime.now(UTC),
            horizon="1d",
            values={"mom": 0.02, "vol": 0.15},
            quality={"lag_ms": 100, "coverage": 0.95}
        )
        
        # Convert to dict for validation
        fv_dict = {
            "instrument": fv.instrument,
            "asof": fv.asof.isoformat(),
            "horizon": fv.horizon,
            "values": fv.values,
            "quality": fv.quality
        }
        
        # Should not raise
        validate(instance=fv_dict, schema=self.schemas["feature_vector"])
    
    @pytest.mark.contract
    def test_signal_schema_valid(self):
        """Test that valid signals conform to schema."""
        signal = Signal(
            instrument="AAPL",
            horizon="1d",
            mu=0.02,
            sigma=0.15,
            prob_edge_pos=0.65,
            policy_id="cs_mom_v1"
        )
        
        signal_dict = {
            "instrument": signal.instrument,
            "horizon": signal.horizon,
            "mu": signal.mu,
            "sigma": signal.sigma,
            "prob_edge_pos": signal.prob_edge_pos,
            "policy_id": signal.policy_id
        }
        
        validate(instance=signal_dict, schema=self.schemas["signal"])
    
    @pytest.mark.contract
    def test_target_position_schema_valid(self):
        """Test that valid target positions conform to schema."""
        target = TargetPosition(
            portfolio="main",
            instrument="AAPL",
            qty_target=100.0,
            max_child_participation=0.1,
            risk_budget=0.05,
            policy_id="cs_mom_v1"
        )
        
        target_dict = {
            "portfolio": target.portfolio,
            "instrument": target.instrument,
            "qty_target": target.qty_target,
            "max_child_participation": target.max_child_participation,
            "risk_budget": target.risk_budget,
            "policy_id": target.policy_id
        }
        
        validate(instance=target_dict, schema=self.schemas["target_position"])
    
    @pytest.mark.contract
    def test_invalid_feature_vector_missing_field(self):
        """Test that missing required fields are caught."""
        invalid_fv = {
            "instrument": "AAPL",
            # Missing asof
            "horizon": "1d",
            "values": {"mom": 0.02},
            "quality": {"lag_ms": 100}
        }
        
        with pytest.raises(ValidationError):
            validate(instance=invalid_fv, schema=self.schemas["feature_vector"])
    
    @pytest.mark.contract
    def test_invalid_signal_bad_probability(self):
        """Test that invalid probability values are caught."""
        invalid_signal = {
            "instrument": "AAPL",
            "horizon": "1d",
            "mu": 0.02,
            "sigma": 0.15,
            "prob_edge_pos": 1.5,  # Invalid: > 1.0
            "policy_id": "test"
        }
        
        with pytest.raises(ValidationError):
            validate(instance=invalid_signal, schema=self.schemas["signal"])
    
    @pytest.mark.contract
    @given(
        instrument=st.text(min_size=1, max_size=10),
        mu=st.floats(allow_nan=False, allow_infinity=False),
        sigma=st.floats(min_value=0, allow_nan=False, allow_infinity=False),
        prob=st.floats(min_value=0, max_value=1, allow_nan=False)
    )
    @settings(max_examples=50)
    def test_signal_schema_property_based(self, instrument, mu, sigma, prob):
        """Property-based test for signal schema validation."""
        signal_dict = {
            "instrument": instrument,
            "horizon": "1d",
            "mu": mu,
            "sigma": sigma,
            "prob_edge_pos": prob,
            "policy_id": "test"
        }
        
        # Should validate without errors
        validate(instance=signal_dict, schema=self.schemas["signal"])
    
    @pytest.mark.contract
    def test_backward_compatibility(self):
        """Test that old message formats are still supported."""
        # V1 format (hypothetical older version)
        v1_signal = {
            "instrument": "AAPL",
            "horizon": "1d",
            "mu": 0.02,
            "sigma": 0.15,
            "prob_edge_pos": 0.65,
            "policy_id": "test"
            # Older version might not have newer optional fields
        }
        
        # Should still validate
        validate(instance=v1_signal, schema=self.schemas["signal"])
    
    @pytest.mark.contract
    def test_examples_conform_to_schemas(self):
        """Test that all provided examples conform to their schemas."""
        for example_file in EXAMPLES_DIR.glob("*.example.json"):
            schema_name = example_file.stem.replace(".example", "")
            
            with open(example_file) as f:
                example = json.load(f)
            
            if schema_name in self.schemas:
                validate(instance=example, schema=self.schemas[schema_name])


class TestIdempotency:
    """Test idempotency of message processing."""
    
    @pytest.mark.contract
    def test_signal_processing_idempotent(self):
        """Test that processing the same signal twice yields same result."""
        from crowetrade.live.signal_agent import SignalAgent
        
        def model(values):
            return values.get("mom", 0), 0.02, 0.6
        
        agent = SignalAgent(
            model=model,
            policy_id="test",
            gates={"prob_edge_min": 0.55, "sigma_max": 0.03}
        )
        
        fv = FeatureVector(
            instrument="TEST",
            asof=datetime.now(UTC),
            horizon="1d",
            values={"mom": 0.02},
            quality={"lag_ms": 100, "coverage": 0.95}
        )
        
        # Process same feature vector multiple times
        results = [agent.infer(fv) for _ in range(5)]
        
        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result.mu == first_result.mu
            assert result.sigma == first_result.sigma
            assert result.prob_edge_pos == first_result.prob_edge_pos
            assert result.policy_id == first_result.policy_id
    
    @pytest.mark.contract
    def test_portfolio_sizing_idempotent(self):
        """Test that portfolio sizing is idempotent."""
        from crowetrade.live.portfolio_agent import PortfolioAgent
        
        agent = PortfolioAgent(risk_budget=1.0, lambda_temper=0.25)
        
        signals = {
            "AAPL": Signal("AAPL", "1d", 0.02, 0.15, 0.65, "test"),
            "GOOGL": Signal("GOOGL", "1d", -0.01, 0.20, 0.60, "test"),
        }
        
        tracking_errors = {
            "AAPL": 0.02,
            "GOOGL": 0.03,
        }
        
        # Size portfolio multiple times with same inputs
        results = [agent.size(signals, tracking_errors) for _ in range(5)]
        
        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result