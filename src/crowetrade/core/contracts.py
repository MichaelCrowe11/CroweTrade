from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class FeatureVector:
    instrument: str
    asof: datetime
    horizon: str
    values: dict[str, float]
    quality: dict[str, float]


@dataclass
class Signal:
    instrument: str
    horizon: str
    mu: float
    sigma: float
    prob_edge_pos: float
    policy_id: str
    policy_hash: str | None = None


@dataclass
class TargetPosition:
    portfolio: str
    instrument: str
    qty_target: float
    max_child_participation: float
    risk_budget: float
    policy_id: str
    policy_hash: str | None = None


@dataclass
class Fill:
    instrument: str
    qty: float
    price: float
    ts: datetime
    venue: str
