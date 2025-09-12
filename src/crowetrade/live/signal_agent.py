from __future__ import annotations

import math
from collections.abc import Callable

from crowetrade.core.contracts import FeatureVector, Signal
from crowetrade.core.validation import SIGNAL_SCHEMA, to_dict
from crowetrade.core.policy import PolicyRegistry, Policy


class SignalAgent:
    """Generates risk-aware trading signals from feature vectors using a provided model.

    Gates: {'prob_edge_min': float, 'sigma_max': float}
    Model: Callable[[dict], tuple[float, float, float]] -> (mu, sigma, prob_edge_pos)
    """

    def __init__(
        self,
        model: Callable[[dict], tuple[float, float, float]],
        policy_id: str,
        gates: dict | None = None,
        policy_registry: PolicyRegistry | None = None,
    ):
        self.model = model
        self.policy_id = policy_id
        self._explicit_gates = gates or {}
        self._policy_registry = policy_registry
        self._policy: Policy | None = None

    @property
    def gates(self) -> dict:
        if self._policy_registry:
            # Hot-reload check (non-blocking fast mtime compare)
            try:
                self._policy_registry.maybe_reload(self.policy_id)
                self._policy = self._policy_registry.load(self.policy_id)
            except Exception:  # pragma: no cover - defensive
                self._policy = None
        if self._policy:
            return {**self._policy.gates, **self._explicit_gates}
        return self._explicit_gates

    def infer(self, fv: FeatureVector) -> Signal | None:
        mu, sigma, pep = self.model(fv.values)
        if pep < float(self.gates.get("prob_edge_min", 0.5)):
            return None
        if sigma > float(self.gates.get("sigma_max", math.inf)):
            return None
        policy_hash = None
        if self._policy_registry:
            policy_hash = self._policy_registry.policy_hash(self.policy_id)
        sig = Signal(
            instrument=fv.instrument,
            horizon=fv.horizon,
            mu=float(mu),
            sigma=float(sigma),
            prob_edge_pos=float(pep),
            policy_id=self.policy_id,
            policy_hash=policy_hash,
        )
        # Runtime validation (can raise jsonschema.ValidationError)
        try:  # pragma: no cover - validation path easy to exercise in tests
            SIGNAL_SCHEMA.validate(to_dict(sig))
        except Exception:
            return None
        return sig
