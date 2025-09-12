from __future__ import annotations

"""Policy loading and validation utilities.

Policies define gating, risk, sizing, and execution parameters. They are YAML
documents stored under `specs/policies/`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import yaml
import json
from hashlib import sha256
from threading import RLock
from time import time


@dataclass
class Policy:
    id: str
    raw: dict[str, Any]

    @property
    def gates(self) -> dict[str, Any]:
        return self.raw.get("entry_gate", {})

    @property
    def sizing(self) -> dict[str, Any]:
        return self.raw.get("sizing", {})

    @property
    def risk(self) -> dict[str, Any]:
        return self.raw.get("risk", {})


class PolicyRegistry:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._cache: dict[str, Policy] = {}
        self._mtimes: dict[str, float] = {}
        self._hashes: dict[str, str] = {}
        self._lock = RLock()

    def _hash_raw(self, raw: dict[str, Any]) -> str:
        return sha256(json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def policy_hash(self, policy_id: str) -> str | None:
        return self._hashes.get(policy_id)

    def load(self, policy_id: str, force: bool = False) -> Policy:
        """Load (or reload) a policy by id.

        Hot-reload semantics: if file mtime changes since last load, cache is invalidated.
        """
        with self._lock:
            path: Path | None = None
            for p in self.root.glob("*.yaml"):
                try:
                    text = p.read_text(encoding="utf-8")
                    data = yaml.safe_load(text) or {}
                except Exception:  # pragma: no cover - defensive
                    continue
                if data.get("id") == policy_id:
                    path = p
                    current_mtime = p.stat().st_mtime
                    if (not force) and policy_id in self._cache and self._mtimes.get(policy_id) == current_mtime:
                        return self._cache[policy_id]
                    pol = Policy(id=policy_id, raw=data)
                    self._cache[policy_id] = pol
                    self._mtimes[policy_id] = current_mtime
                    self._hashes[policy_id] = self._hash_raw(data)
                    return pol
            raise FileNotFoundError(f"Policy '{policy_id}' not found in {self.root}")

    def maybe_reload(self, policy_id: str) -> bool:
        """Check mtime; reload if changed. Returns True if reloaded."""
        with self._lock:
            # Identify file first
            for p in self.root.glob("*.yaml"):
                try:
                    text = p.read_text(encoding="utf-8")
                    data = yaml.safe_load(text) or {}
                except Exception:  # pragma: no cover
                    continue
                if data.get("id") != policy_id:
                    continue
                mtime = p.stat().st_mtime
                if policy_id not in self._mtimes or self._mtimes[policy_id] != mtime:
                    pol = Policy(id=policy_id, raw=data)
                    self._cache[policy_id] = pol
                    self._mtimes[policy_id] = mtime
                    self._hashes[policy_id] = self._hash_raw(data)
                    return True
                return False
            return False
