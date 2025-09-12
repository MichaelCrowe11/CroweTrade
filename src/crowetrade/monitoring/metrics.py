from __future__ import annotations

"""Unified lightweight metrics registry (counters & gauges).

If `prometheus_client` is available it will mirror updates to real Prometheus metrics.
Otherwise it remains in-process only.
"""

from dataclasses import dataclass, field
from threading import Lock
from typing import Dict

try:  # pragma: no cover - optional
    from prometheus_client import Counter as PromCounter, Gauge as PromGauge  # type: ignore
except Exception:  # pragma: no cover
    PromCounter = None  # type: ignore
    PromGauge = None  # type: ignore


class Counter:
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._lock = Lock()
        self._value = 0.0
        self._prom = PromCounter(name, description) if PromCounter else None

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount
            if self._prom:
                self._prom.inc(amount)  # type: ignore

    @property
    def value(self) -> float:  # pragma: no cover - trivial
        return self._value


class Gauge:
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._lock = Lock()
        self._value = 0.0
        self._prom = PromGauge(name, description) if PromGauge else None

    def set(self, value: float) -> None:
        with self._lock:
            self._value = float(value)
            if self._prom:
                try:
                    self._prom.set(value)  # type: ignore
                except Exception:  # pragma: no cover
                    pass

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount
            if self._prom:
                try:
                    self._prom.inc(amount)  # type: ignore
                except Exception:  # pragma: no cover
                    pass

    @property
    def value(self) -> float:  # pragma: no cover - trivial
        return self._value


@dataclass
class MetricsRegistry:
    counters: Dict[str, Counter] = field(default_factory=dict)
    gauges: Dict[str, Gauge] = field(default_factory=dict)

    def counter(self, name: str, description: str = "") -> Counter:
        if name not in self.counters:
            self.counters[name] = Counter(name, description)
        return self.counters[name]

    def gauge(self, name: str, description: str = "") -> Gauge:
        if name not in self.gauges:
            self.gauges[name] = Gauge(name, description)
        return self.gauges[name]


REGISTRY = MetricsRegistry()
