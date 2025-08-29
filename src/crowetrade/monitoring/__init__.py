from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol

import numpy as np


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class Metric:
    name: str
    type: MetricType
    value: float
    timestamp: datetime
    labels: dict[str, str] = field(default_factory=dict)
    unit: str | None = None
    
    def to_prometheus(self) -> str:
        labels_str = ",".join(f'{k}="{v}"' for k, v in self.labels.items())
        if labels_str:
            labels_str = f"{{{labels_str}}}"
        return f"{self.name}{labels_str} {self.value} {int(self.timestamp.timestamp() * 1000)}"


@dataclass
class Alert:
    alert_id: str
    name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    
    source: str
    metric_name: str | None = None
    threshold: float | None = None
    actual_value: float | None = None
    
    labels: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    acknowledged: bool = False
    resolved: bool = False
    resolved_at: datetime | None = None
    
    def auto_resolve_after(self, duration: timedelta) -> bool:
        if not self.resolved and datetime.utcnow() - self.timestamp > duration:
            self.resolved = True
            self.resolved_at = datetime.utcnow()
            return True
        return False


@dataclass
class SLO:
    name: str
    description: str
    target: float  # e.g., 99.9 for 99.9%
    window: timedelta
    
    metric_name: str
    threshold: float
    comparison: str  # "lt", "lte", "gt", "gte", "eq"
    
    current_value: float | None = None
    is_met: bool = True
    error_budget_remaining: float = 100.0
    
    def evaluate(self, value: float) -> bool:
        self.current_value = value
        
        if self.comparison == "lt":
            self.is_met = value < self.threshold
        elif self.comparison == "lte":
            self.is_met = value <= self.threshold
        elif self.comparison == "gt":
            self.is_met = value > self.threshold
        elif self.comparison == "gte":
            self.is_met = value >= self.threshold
        elif self.comparison == "eq":
            self.is_met = abs(value - self.threshold) < 1e-9
        
        if self.is_met:
            self.error_budget_remaining = 100.0
        else:
            self.error_budget_remaining = max(0, 100 - ((1 - value/self.target) * 100))
        
        return self.is_met


@dataclass
class HealthCheck:
    name: str
    component: str
    status: HealthStatus
    timestamp: datetime
    
    latency_ms: float | None = None
    error_rate: float | None = None
    throughput: float | None = None
    
    dependencies: list[str] = field(default_factory=list)
    failed_dependencies: list[str] = field(default_factory=list)
    
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY


class MetricsCollector:
    def __init__(self):
        self.metrics: dict[str, list[Metric]] = {}
        self.alerts: list[Alert] = []
        self.slos: dict[str, SLO] = {}
    
    def record(self, name: str, value: float, metric_type: MetricType = MetricType.GAUGE,
              labels: dict[str, str] | None = None, unit: str | None = None):
        metric = Metric(
            name=name,
            type=metric_type,
            value=value,
            timestamp=datetime.utcnow(),
            labels=labels or {},
            unit=unit
        )
        
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(metric)
        
        self._check_thresholds(metric)
    
    def record_latency(self, operation: str, duration_ms: float, labels: dict[str, str] | None = None):
        full_labels = {"operation": operation}
        if labels:
            full_labels.update(labels)
        
        self.record(
            name="latency",
            value=duration_ms,
            metric_type=MetricType.HISTOGRAM,
            labels=full_labels,
            unit="ms"
        )
    
    def increment_counter(self, name: str, labels: dict[str, str] | None = None):
        current = self._get_current_value(name, labels)
        self.record(
            name=name,
            value=current + 1,
            metric_type=MetricType.COUNTER,
            labels=labels
        )
    
    def _get_current_value(self, name: str, labels: dict[str, str] | None = None) -> float:
        if name not in self.metrics:
            return 0.0
        
        for metric in reversed(self.metrics[name]):
            if metric.labels == (labels or {}):
                return metric.value
        
        return 0.0
    
    def _check_thresholds(self, metric: Metric):
        if metric.name == "latency" and metric.value > 100:
            self.create_alert(
                name="high_latency",
                severity=AlertSeverity.WARNING,
                message=f"High latency detected: {metric.value}ms",
                metric_name=metric.name,
                threshold=100,
                actual_value=metric.value
            )
        
        for slo in self.slos.values():
            if slo.metric_name == metric.name:
                if not slo.evaluate(metric.value):
                    self.create_alert(
                        name=f"slo_violation_{slo.name}",
                        severity=AlertSeverity.ERROR,
                        message=f"SLO violation: {slo.name}",
                        metric_name=metric.name,
                        threshold=slo.threshold,
                        actual_value=metric.value
                    )
    
    def create_alert(self, name: str, severity: AlertSeverity, message: str,
                    **kwargs) -> Alert:
        alert = Alert(
            alert_id=f"{name}_{datetime.utcnow().timestamp()}",
            name=name,
            severity=severity,
            message=message,
            timestamp=datetime.utcnow(),
            source="metrics_collector",
            **kwargs
        )
        
        self.alerts.append(alert)
        return alert
    
    def get_percentile(self, name: str, percentile: float,
                       window: timedelta | None = None) -> float:
        if name not in self.metrics:
            return np.nan
        
        cutoff = datetime.utcnow() - window if window else datetime.min
        values = [m.value for m in self.metrics[name] if m.timestamp > cutoff]
        
        if not values:
            return np.nan
        
        return np.percentile(values, percentile)


class DriftDetector(Protocol):
    def detect_feature_drift(self, features: np.ndarray) -> bool:
        ...
    
    def detect_prediction_drift(self, predictions: np.ndarray) -> bool:
        ...
    
    def get_drift_score(self) -> float:
        ...