from __future__ import annotations

"""Feature Store abstraction with simple in-memory + version tracking.

This is a stepping stone toward a persisted online/offline parity store.
"""

from dataclasses import dataclass
from typing import Mapping, Iterable, MutableMapping, Sequence
from time import time
import json
from pathlib import Path
from threading import Lock
from hashlib import sha256


def _stable_hash(obj: dict) -> str:
    """Stable hash for a feature dict (sorted keys)."""
    enc = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return sha256(enc.encode("utf-8")).hexdigest()


@dataclass
class FeatureRecord:
    ts: float
    version: int
    values: dict[str, float]
    prev_version: int | None = None
    value_hash: str | None = None


class FeatureStore:
    def put(self, key: str, values: Mapping[str, float]) -> FeatureRecord: ...  # pragma: no cover
    def get(self, key: str) -> FeatureRecord | None: ...  # pragma: no cover
    def batch_get(self, keys: Iterable[str]) -> dict[str, FeatureRecord | None]: ...  # pragma: no cover
    def get_version(self, key: str, version: int) -> FeatureRecord | None: ...  # pragma: no cover
    def history(self, key: str, limit: int | None = None) -> Sequence[FeatureRecord]: ...  # pragma: no cover


class InMemoryFeatureStore(FeatureStore):
    def __init__(self):
        self._data: MutableMapping[str, list[FeatureRecord]] = {}

    def put(self, key: str, values: Mapping[str, float], version: int | None = None) -> FeatureRecord:
        lst = self._data.setdefault(key, [])
        computed_version = (lst[-1].version + 1) if lst else 1
        # If caller supplies version ensure monotonic; otherwise use computed
        if version is not None and version >= computed_version:
            computed_version = version
        prev_version = lst[-1].version if lst else None
        rec = FeatureRecord(ts=time(), version=computed_version, values=dict(values), prev_version=prev_version, value_hash=_stable_hash(dict(values)))
        lst.append(rec)
        return rec

    def get(self, key: str) -> FeatureRecord | None:
        lst = self._data.get(key)
        if not lst:
            return None
        last = lst[-1]
        return FeatureRecord(ts=last.ts, version=last.version, values=dict(last.values), prev_version=last.prev_version, value_hash=last.value_hash)

    def batch_get(self, keys: Iterable[str]) -> dict[str, FeatureRecord | None]:
        return {k: self.get(k) for k in keys}

    def get_version(self, key: str, version: int) -> FeatureRecord | None:
        lst = self._data.get(key)
        if not lst:
            return None
        for rec in lst:
            if rec.version == version:
                return FeatureRecord(ts=rec.ts, version=rec.version, values=dict(rec.values), prev_version=rec.prev_version, value_hash=rec.value_hash)
        return None

    def history(self, key: str, limit: int | None = None) -> Sequence[FeatureRecord]:
        lst = self._data.get(key, [])
        if limit is None:
            subset = lst
        else:
            subset = lst[-limit:]
        return [FeatureRecord(ts=r.ts, version=r.version, values=dict(r.values), prev_version=r.prev_version, value_hash=r.value_hash) for r in subset]


class DiskFeatureStore(FeatureStore):
    """Simple JSONL append-only per-key history.

    Each key -> <root>/<key>.jsonl with lines: {ts, version, prev_version, values, value_hash}
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def _path(self, key: str) -> Path:
        safe = key.replace("/", "_")
        return self.root / f"{safe}.jsonl"

    def put(self, key: str, values: Mapping[str, float], version: int | None = None) -> FeatureRecord:
        path = self._path(key)
        with self._lock:
            last = self.get(key)
            computed_version = 1 if last is None else last.version + 1
            if version is not None and version >= computed_version:
                computed_version = version
            prev_version = None if last is None else last.version
            rec = FeatureRecord(ts=time(), version=computed_version, values=dict(values), prev_version=prev_version, value_hash=_stable_hash(dict(values)))
            line = json.dumps({
                "ts": rec.ts,
                "version": rec.version,
                "prev_version": rec.prev_version,
                "values": rec.values,
                "value_hash": rec.value_hash,
            }, separators=(",", ":"))
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        return rec

    def _read_all(self, key: str) -> list[FeatureRecord]:
        path = self._path(key)
        if not path.exists():
            return []
        out: list[FeatureRecord] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    out.append(FeatureRecord(ts=float(obj["ts"]), version=int(obj["version"]), prev_version=obj.get("prev_version"), values=dict(obj["values"]), value_hash=obj.get("value_hash")))
                except Exception:  # pragma: no cover - defensive
                    continue
        return out

    def get(self, key: str) -> FeatureRecord | None:
        hist = self._read_all(key)
        if not hist:
            return None
        return hist[-1]

    def batch_get(self, keys: Iterable[str]) -> dict[str, FeatureRecord | None]:
        return {k: self.get(k) for k in keys}

    def get_version(self, key: str, version: int) -> FeatureRecord | None:
        for rec in self._read_all(key):
            if rec.version == version:
                return rec
        return None

    def history(self, key: str, limit: int | None = None) -> Sequence[FeatureRecord]:
        hist = self._read_all(key)
        if limit is None:
            return hist
        return hist[-limit:]
