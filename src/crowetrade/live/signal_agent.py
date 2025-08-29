from __future__ import annotations

import math
from collections.abc import Callable

from crowetrade.core.contracts import FeatureVector, Signal


class SignalAgent:
    """Generates risk-aware trading signals from feature vectors using a provided model.

    Gates: {'prob_edge_min': float, 'sigma_max': float}
    Model: Callable[[dict], tuple[float, float, float]] -> (mu, sigma, prob_edge_pos)
    """

    def __init__(
        self, model: Callable[[dict], tuple[float, float, float]], policy_id: str, gates: dict
    ):
        self.model = model
        self.policy_id = policy_id
        self.gates = gates

    def infer(self, fv: FeatureVector) -> Signal | None:
        mu, sigma, pep = self.model(fv.values)
        if pep < float(self.gates.get("prob_edge_min", 0.5)):
            return None
        if sigma > float(self.gates.get("sigma_max", math.inf)):
            return None
        return Signal(
            instrument=fv.instrument,
            horizon=fv.horizon,
            mu=float(mu),
            sigma=float(sigma),
            prob_edge_pos=float(pep),
            policy_id=self.policy_id,
        )
