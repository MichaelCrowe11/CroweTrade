from __future__ import annotations

# coverage: ignore file

"""Lightweight in-process metrics registry (placeholder for Prometheus exporter).

Provides Counter and Gauge classes with thread-safe updates. Intended to be swapped
out by a real monitoring backend later; current design keeps API minimal.
"""

from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, Tuple


class Counter:
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._lock = Lock()
        self._value = 0.0

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    @property
    def value(self) -> float:
        return self._value


class Gauge:
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._lock = Lock()
        self._value = 0.0

    def set(self, value: float) -> None:
        with self._lock:
            self._value = float(value)

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    @property
    def value(self) -> float:
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
