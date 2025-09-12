from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from typing import Protocol


class FeatureStore(Protocol):
    def put(self, key: str, vector: Mapping[str, float], ts: int) -> None: ...
    def get(self, key: str, fields: Iterable[str] | None = None) -> tuple[int, dict[str, float]]: ...
    def batch_get(self, keys: Iterable[str]) -> dict[str, tuple[int, dict[str, float]]]: ...


@dataclass
class InMemoryFeatureStore(FeatureStore):
    _data: MutableMapping[str, tuple[int, dict[str, float]]]

    def __init__(self):
        self._data = {}

    def put(self, key: str, vector: Mapping[str, float], ts: int) -> None:
        self._data[key] = (ts, dict(vector))

    def get(self, key: str, fields: Iterable[str] | None = None) -> tuple[int, dict[str, float]]:
        ts, vec = self._data.get(key, (0, {}))
        if fields is None:
            return ts, dict(vec)
        f = set(fields)
        return ts, {k: v for k, v in vec.items() if k in f}

    def batch_get(self, keys: Iterable[str]) -> dict[str, tuple[int, dict[str, float]]]:
        return {k: self.get(k) for k in keys}


def _bin_edges(values: list[float], bins: int = 10) -> list[float]:
    if not values:
        return [0.0]
    lo, hi = min(values), max(values)
    if lo == hi:
        hi = lo + 1e-9
    step = (hi - lo) / bins
    return [lo + i * step for i in range(bins + 1)]


def psi(ref: list[float], cur: list[float], bins: int = 10) -> float:
    """
    Population Stability Index between reference (offline) and current (online).
    Returns 0 for identical distributions; >0.2 often indicates drift.
    """
    if not ref and not cur:
        return 0.0
    # Always derive bin edges from reference where possible for stability
    edges = _bin_edges(ref if ref else cur, bins=bins)
    def hist(x: list[float]) -> list[int]:
        h = [0] * (len(edges) - 1)
        for v in x:
            idx = min(max(0, next((i for i, e in enumerate(edges[1:]) if v <= e), len(h) - 1)), len(h) - 1)
            h[idx] += 1
        return h

    hr, hc = hist(ref), hist(cur)
    nr, nc = max(1, sum(hr)), max(1, sum(hc))
    psi_val = 0.0
    for r, c in zip(hr, hc, strict=False):
        pr = max(r / nr, 1e-6)
        pc = max(c / nc, 1e-6)
        psi_val += (pc - pr) * math.log(pc / pr)
    return psi_val


class ParityMonitor:
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold

    def check(self, ref_vecs: list[Mapping[str, float]], cur_vecs: list[Mapping[str, float]]) -> dict[str, float]:
        keys = set().union(*(v.keys() for v in ref_vecs + cur_vecs))
        out: dict[str, float] = {}
        for k in keys:
            ref = [float(v.get(k, 0.0)) for v in ref_vecs]
            cur = [float(v.get(k, 0.0)) for v in cur_vecs]
            out[k] = psi(ref, cur)
        return out

    def breached(self, psis: Mapping[str, float]) -> list[str]:
        return [k for k, v in psis.items() if v > self.threshold]