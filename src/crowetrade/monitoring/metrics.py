"""
Production monitoring and metrics collection for CroweTrade.
Includes Prometheus metrics, health checks, and performance monitoring.
"""

import time
import logging
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import contextmanager
import psutil
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass 
class HealthStatus:
    """System health status."""
    healthy: bool
    checks: Dict[str, bool]
    metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)


class MetricsCollector:
    """Collects and exports metrics for monitoring."""
    
    def __init__(self):
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
        
    def counter_increment(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        with self._lock:
            key = self._make_key(name, labels)
            self._counters[key] += value
            self._record_point(name, self._counters[key], labels)
    
    def gauge_set(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric value."""
        with self._lock:
            key = self._make_key(name, labels)
            self._gauges[key] = value
            self._record_point(name, value, labels)
    
    def histogram_observe(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe a value for histogram metric."""
        with self._lock:
            key = self._make_key(name, labels)
            self._histograms[key].append(value)
            # Keep only last 1000 observations per histogram
            if len(self._histograms[key]) > 1000:
                self._histograms[key] = self._histograms[key][-1000:]
            self._record_point(name, value, labels)
    
    @contextmanager
    def timer(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager to time operations."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.histogram_observe(name, duration, labels)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        with self._lock:
            return {
                'counters': dict(self._counters),
                'gauges': dict(self._gauges),
                'histograms': {k: {
                    'count': len(v),
                    'sum': sum(v),
                    'avg': sum(v) / len(v) if v else 0,
                    'min': min(v) if v else 0,
                    'max': max(v) if v else 0,
                    'p50': self._percentile(v, 50),
                    'p95': self._percentile(v, 95),
                    'p99': self._percentile(v, 99),
                } for k, v in self._histograms.items()}
            }
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create a unique key for metric with labels."""
        if not labels:
            return name
        label_str = ','.join(f'{k}={v}' for k, v in sorted(labels.items()))
        return f'{name}[{label_str}]'
    
    def _record_point(self, name: str, value: float, labels: Optional[Dict[str, str]]):
        """Record a metric point."""
        point = MetricPoint(name, value, labels or {})
        self._metrics[name].append(point)
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]


class HealthChecker:
    """Performs system health checks."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self._checks: Dict[str, callable] = {}
        self._register_default_checks()
        
    def register_check(self, name: str, check_func: callable):
        """Register a custom health check."""
        self._checks[name] = check_func
        
    async def check_health(self) -> HealthStatus:
        """Perform all health checks."""
        checks = {}
        
        for name, check_func in self._checks.items():
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                checks[name] = bool(result)
                self.metrics.gauge_set(f'health_check_{name}', 1.0 if result else 0.0)
            except Exception as e:
                logger.error(f"Health check {name} failed: {e}")
                checks[name] = False
                self.metrics.gauge_set(f'health_check_{name}', 0.0)
        
        # System metrics
        system_metrics = self._collect_system_metrics()
        
        # Overall health
        healthy = all(checks.values())
        self.metrics.gauge_set('system_healthy', 1.0 if healthy else 0.0)
        
        return HealthStatus(
            healthy=healthy,
            checks=checks,
            metrics=system_metrics
        )
    
    def _register_default_checks(self):
        """Register default system health checks."""
        self.register_check('memory', self._check_memory)
        self.register_check('disk_space', self._check_disk_space)
        self.register_check('cpu', self._check_cpu)
        
    def _check_memory(self) -> bool:
        """Check memory usage."""
        memory = psutil.virtual_memory()
        usage_percent = memory.percent
        self.metrics.gauge_set('system_memory_usage_percent', usage_percent)
        return usage_percent < 90.0
    
    def _check_disk_space(self) -> bool:
        """Check disk space usage."""
        disk = psutil.disk_usage('/')
        usage_percent = (disk.used / disk.total) * 100
        self.metrics.gauge_set('system_disk_usage_percent', usage_percent)
        return usage_percent < 85.0
        
    def _check_cpu(self) -> bool:
        """Check CPU usage."""
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics.gauge_set('system_cpu_usage_percent', cpu_percent)
        return cpu_percent < 90.0
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect system performance metrics."""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        metrics = {
            'memory_usage_bytes': memory.used,
            'memory_available_bytes': memory.available,
            'memory_usage_percent': memory.percent,
            'disk_usage_bytes': disk.used,
            'disk_free_bytes': disk.free,
            'disk_usage_percent': (disk.used / disk.total) * 100,
            'cpu_usage_percent': psutil.cpu_percent(),
            'process_count': len(psutil.pids()),
        }
        
        # Record as gauge metrics
        for name, value in metrics.items():
            self.metrics.gauge_set(f'system_{name}', value)
            
        return metrics


class TradingMetrics:
    """Specialized metrics for trading operations."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        
    def record_signal_generated(self, instrument: str, strategy: str):
        """Record a trading signal generation."""
        labels = {'instrument': instrument, 'strategy': strategy}
        self.metrics.counter_increment('signals_generated_total', 1.0, labels)
    
    def record_order_submitted(self, instrument: str, order_type: str, quantity: float):
        """Record an order submission."""
        labels = {'instrument': instrument, 'order_type': order_type}
        self.metrics.counter_increment('orders_submitted_total', 1.0, labels)
        self.metrics.histogram_observe('order_quantity', abs(quantity), labels)
    
    def record_fill(self, instrument: str, quantity: float, price: float, slippage: float):
        """Record a trade fill."""
        labels = {'instrument': instrument}
        self.metrics.counter_increment('fills_total', 1.0, labels)
        self.metrics.histogram_observe('fill_quantity', abs(quantity), labels)
        self.metrics.histogram_observe('fill_price', price, labels)
        self.metrics.histogram_observe('slippage_bps', slippage * 10000, labels)
    
    def record_pnl(self, strategy: str, pnl: float, unrealized_pnl: float):
        """Record PnL for a strategy."""
        labels = {'strategy': strategy}
        self.metrics.gauge_set('realized_pnl', pnl, labels)
        self.metrics.gauge_set('unrealized_pnl', unrealized_pnl, labels)
        self.metrics.gauge_set('total_pnl', pnl + unrealized_pnl, labels)
    
    def record_risk_breach(self, breach_type: str, value: float, limit: float):
        """Record a risk limit breach."""
        labels = {'breach_type': breach_type}
        self.metrics.counter_increment('risk_breaches_total', 1.0, labels)
        self.metrics.gauge_set(f'risk_breach_{breach_type}_value', value, labels)
        self.metrics.gauge_set(f'risk_breach_{breach_type}_limit', limit, labels)
    
    def record_execution_latency(self, operation: str, latency_ms: float):
        """Record execution latency."""
        labels = {'operation': operation}
        self.metrics.histogram_observe('execution_latency_ms', latency_ms, labels)


class PerformanceTracker:
    """Tracks application performance metrics."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self._operation_stats: Dict[str, List[float]] = defaultdict(list)
        
    @contextmanager
    def track_operation(self, operation_name: str, labels: Optional[Dict[str, str]] = None):
        """Track the performance of an operation."""
        start_time = time.time()
        start_cpu = psutil.Process().cpu_percent()
        
        try:
            yield
            success = True
        except Exception:
            success = False
            raise
        finally:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            end_cpu = psutil.Process().cpu_percent()
            cpu_delta = end_cpu - start_cpu
            
            # Record metrics
            op_labels = dict(labels or {})
            op_labels['operation'] = operation_name
            op_labels['success'] = str(success).lower()
            
            self.metrics.histogram_observe('operation_duration_ms', duration_ms, op_labels)
            self.metrics.histogram_observe('operation_cpu_delta', cpu_delta, op_labels)
            self.metrics.counter_increment('operations_total', 1.0, op_labels)
            
            # Track operation statistics
            self._operation_stats[operation_name].append(duration_ms)
    
    def get_operation_stats(self) -> Dict[str, Dict[str, float]]:
        """Get operation performance statistics."""
        stats = {}
        for operation, durations in self._operation_stats.items():
            if durations:
                stats[operation] = {
                    'count': len(durations),
                    'avg_ms': sum(durations) / len(durations),
                    'min_ms': min(durations),
                    'max_ms': max(durations),
                    'p95_ms': self._percentile(durations, 95),
                    'p99_ms': self._percentile(durations, 99),
                }
        return stats
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]


# Global instances
_metrics_collector = MetricsCollector()
_health_checker = HealthChecker(_metrics_collector)
_trading_metrics = TradingMetrics(_metrics_collector)
_performance_tracker = PerformanceTracker(_metrics_collector)


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    return _metrics_collector


def get_health_checker() -> HealthChecker:
    """Get the global health checker."""
    return _health_checker


def get_trading_metrics() -> TradingMetrics:
    """Get the trading-specific metrics collector."""
    return _trading_metrics


def get_performance_tracker() -> PerformanceTracker:
    """Get the performance tracker."""
    return _performance_tracker