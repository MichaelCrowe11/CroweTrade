from __future__ import annotations

# coverage: ignore file

"""Lightweight JSON Schema validation helpers.

Uses jsonschema if available; otherwise no-op (can be upgraded later).
"""

from dataclasses import dataclass
from typing import Any, Mapping

try:  # Optional dependency
    import jsonschema  # type: ignore
except Exception:  # pragma: no cover - fallback
    jsonschema = None  # type: ignore


@dataclass
class SchemaSpec:
    name: str
    schema: Mapping[str, Any]

    def validate(self, payload: Mapping[str, Any]) -> None:
        if jsonschema is None:  # pragma: no cover
            return
        jsonschema.validate(payload, self.schema)  # type: ignore


# Minimal embedded schemas (can later be loaded from specs/schemas)
SIGNAL_SCHEMA = SchemaSpec(
    name="signal_v1",
    schema={
        "type": "object",
        "required": ["instrument", "horizon", "mu", "sigma", "prob_edge_pos", "policy_id"],
        "properties": {
            "instrument": {"type": "string"},
            "horizon": {"type": "string"},
            "mu": {"type": "number"},
            "sigma": {"type": "number"},
            "prob_edge_pos": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "policy_id": {"type": "string"},
            "policy_hash": {"type": ["string", "null"]},
        },
        "additionalProperties": False,
    },
)

TARGET_SCHEMA = SchemaSpec(
    name="target_position_v1",
    schema={
        "type": "object",
        "required": ["portfolio", "instrument", "qty_target", "risk_budget", "policy_id"],
        "properties": {
            "portfolio": {"type": "string"},
            "instrument": {"type": "string"},
            "qty_target": {"type": "number"},
            "max_child_participation": {"type": "number"},
            "risk_budget": {"type": "number"},
            "policy_id": {"type": "string"},
        },
        "additionalProperties": True,
    },
)


def to_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    raise TypeError(f"Unsupported object type for validation: {type(obj)}")
