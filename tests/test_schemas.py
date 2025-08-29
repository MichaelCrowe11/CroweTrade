from __future__ import annotations

import json
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "specs" / "schemas"
EXAMPLES = ROOT / "specs" / "examples"


def _load_json(p: Path):
    return json.loads(p.read_text())


def test_feature_vector_schema():
    schema = _load_json(SCHEMAS / "feature_vector.schema.json")
    example = _load_json(EXAMPLES / "feature_vector.example.json")
    jsonschema.validate(instance=example, schema=schema)


def test_signal_schema():
    schema = _load_json(SCHEMAS / "signal.schema.json")
    example = _load_json(EXAMPLES / "signal.example.json")
    jsonschema.validate(instance=example, schema=schema)


def test_target_position_schema():
    schema = _load_json(SCHEMAS / "target_position.schema.json")
    example = _load_json(EXAMPLES / "target_position.example.json")
    jsonschema.validate(instance=example, schema=schema)
