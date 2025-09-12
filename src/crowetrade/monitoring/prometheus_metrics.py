"""Prometheus metrics collection for CroweTrade platform"""
import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import time
from functools import wraps
from dataclasses import dataclass
from contextlib import contextmanager
import threading

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary, Info,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
        start_http_server
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Mock classes for when prometheus_client is not available
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def dec(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def time(self, *args, **kwargs): return contextmanager(lambda: iter([None]))()
        def labels(self, *args, **kwargs): return self
    
    class Summary:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Info:
        def __init__(self, *args, **kwargs): pass
        def info(self, *args, **kwargs): pass
    
    class CollectorRegistry:
        def __init__(self): pass
    
    def generate_latest(*args): return ""
    CONTENT_TYPE_LATEST = "text/plain"
    def start_http_server(*args, **kwargs): pass

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    INFO = "info"


@dataclass
class MetricDefinition:
    """Definition of a metric"""
    name: str
    description: str
    metric_type: MetricType
    labels: List[str]
    buckets: Optional[List[float]] = None  # For histograms


class PrometheusMetrics:
    """
    Centralized Prometheus metrics collection for CroweTrade
    """
    
    def __init__(
        self,
        registry: Optional[CollectorRegistry] = None,
        prefix: str = "crowetrade",
        enable_http_server: bool = True,
        http_port: int = 9090
    ):
        self.registry = registry or CollectorRegistry()
        self.prefix = prefix
        self.enable_http_server = enable_http_server
        self.http_port = http_port
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Storage for metrics
        self._metrics: Dict[str, Union[Counter, Gauge, Histogram, Summary, Info]] = {}
        
        # Initialize core metrics
        self._initialize_core_metrics()
        
        # Start HTTP server if enabled
        if self.enable_http_server and PROMETHEUS_AVAILABLE:
            try:
                start_http_server(self.http_port, registry=self.registry)
                logger.info(f"Prometheus metrics server started on port {self.http_port}")
            except Exception as e:
                logger.warning(f"Failed to start Prometheus HTTP server: {e}")
    
    def _initialize_core_metrics(self) -> None:
        """Initialize core CroweTrade metrics"""
        
        # Trading metrics
        self.register_counter(
            'trades_total',
            'Total number of trades executed',
            ['symbol', 'side', 'strategy', 'venue']
        )
        
        self.register_gauge(
            'trades_pnl_total',
            'Total PnL from trades',
            ['strategy', 'symbol']
        )
        
        self.register_histogram(
            'trade_size_shares',
            'Distribution of trade sizes in shares',
            ['symbol', 'side'],
            buckets=[10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
        )
        
        self.register_histogram(
            'trade_execution_duration_seconds',
            'Time to execute trades',
            ['algorithm', 'venue'],
            buckets=[0.001, 0.01, 0.1, 0.5, 1, 5, 10, 30, 60]
        )
        
        # Portfolio metrics
        self.register_gauge(
            'portfolio_value_usd',
            'Total portfolio value in USD',
            ['strategy', 'account']
        )
        
        self.register_gauge(
            'portfolio_positions',
            'Current position sizes',
            ['symbol', 'strategy']
        )
        
        self.register_gauge(
            'portfolio_weights',
            'Portfolio weights by symbol',
            ['symbol', 'strategy']
        )
        
        self.register_gauge(
            'portfolio_drawdown',
            'Current drawdown percentage',
            ['strategy']
        )
        
        # Risk metrics
        self.register_gauge(
            'risk_var_95',
            'Value at Risk (95% confidence)',
            ['strategy', 'horizon']
        )
        
        self.register_gauge(
            'risk_exposure_gross',
            'Gross exposure across all positions',
            ['strategy']
        )
        
        self.register_counter(
            'risk_breaches_total',
            'Number of risk limit breaches',
            ['limit_type', 'strategy']
        )
        
        # Execution metrics
        self.register_histogram(
            'execution_slippage_bps',
            'Execution slippage in basis points',
            ['symbol', 'algorithm'],
            buckets=[0, 1, 2, 5, 10, 20, 50, 100, 200]
        )
        
        self.register_gauge(
            'execution_fill_rate',
            'Order fill rate percentage',
            ['venue', 'order_type']
        )
        
        # Market data metrics
        self.register_counter(
            'market_data_updates_total',
            'Total market data updates received',
            ['symbol', 'data_type', 'source']
        )
        
        self.register_gauge(
            'market_data_latency_ms',
            'Market data latency in milliseconds',
            ['source', 'symbol']
        )
        
        # System metrics
        self.register_gauge(
            'agent_status',
            'Agent status (1=healthy, 0=unhealthy)',
            ['agent_id', 'agent_type']
        )
        
        self.register_counter(
            'events_processed_total',
            'Total events processed',
            ['event_type', 'source', 'status']
        )
        
        self.register_histogram(
            'event_processing_duration_seconds',
            'Time to process events',
            ['event_type'],
            buckets=[0.001, 0.01, 0.1, 0.5, 1, 5]
        )
        
        # Machine learning metrics
        self.register_gauge(
            'model_accuracy',
            'Model prediction accuracy',
            ['model_name', 'model_version']
        )
        
        self.register_counter(
            'model_predictions_total',
            'Total model predictions made',
            ['model_name', 'prediction_type']
        )
        
        self.register_histogram(
            'model_inference_duration_ms',
            'Model inference time in milliseconds',
            ['model_name'],
            buckets=[1, 5, 10, 50, 100, 500, 1000]
        )
    
    def register_counter(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None
    ) -> Counter:
        """Register a new counter metric"""
        
        full_name = f"{self.prefix}_{name}"
        labels = labels or []
        
        with self._lock:
            if full_name not in self._metrics:
                counter = Counter(
                    full_name,
                    description,
                    labels,
                    registry=self.registry
                )
                self._metrics[full_name] = counter
            
            return self._metrics[full_name]
    
    def register_gauge(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None
    ) -> Gauge:
        """Register a new gauge metric"""
        
        full_name = f"{self.prefix}_{name}"
        labels = labels or []
        
        with self._lock:
            if full_name not in self._metrics:
                gauge = Gauge(
                    full_name,
                    description,
                    labels,
                    registry=self.registry
                )
                self._metrics[full_name] = gauge
            
            return self._metrics[full_name]
    
    def register_histogram(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        buckets: Optional[List[float]] = None
    ) -> Histogram:
        """Register a new histogram metric"""
        
        full_name = f"{self.prefix}_{name}"
        labels = labels or []
        
        with self._lock:
            if full_name not in self._metrics:
                histogram = Histogram(
                    full_name,
                    description,
                    labels,
                    buckets=buckets,
                    registry=self.registry
                )
                self._metrics[full_name] = histogram
            
            return self._metrics[full_name]
    
    def register_summary(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None
    ) -> Summary:
        """Register a new summary metric"""
        
        full_name = f"{self.prefix}_{name}"
        labels = labels or []
        
        with self._lock:
            if full_name not in self._metrics:
                summary = Summary(
                    full_name,
                    description,
                    labels,
                    registry=self.registry
                )
                self._metrics[full_name] = summary
            
            return self._metrics[full_name]
    
    def get_metric(self, name: str) -> Optional[Union[Counter, Gauge, Histogram, Summary]]:
        """Get a registered metric by name"""
        full_name = f"{self.prefix}_{name}"
        return self._metrics.get(full_name)
    
    def increment_counter(
        self,
        name: str,
        value: float = 1,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Increment a counter metric"""
        
        metric = self.get_metric(name)
        if metric and isinstance(metric, Counter):
            if labels:
                metric.labels(**labels).inc(value)
            else:
                metric.inc(value)
    
    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Set a gauge metric value"""
        
        metric = self.get_metric(name)
        if metric and isinstance(metric, Gauge):
            if labels:
                metric.labels(**labels).set(value)
            else:
                metric.set(value)
    
    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Observe a value in a histogram"""
        
        metric = self.get_metric(name)
        if metric and isinstance(metric, Histogram):
            if labels:
                metric.labels(**labels).observe(value)
            else:
                metric.observe(value)
    
    @contextmanager
    def time_histogram(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None
    ):
        """Context manager to time an operation"""
        
        metric = self.get_metric(name)
        if metric and isinstance(metric, Histogram):
            if labels:
                with metric.labels(**labels).time():
                    yield
            else:
                with metric.time():
                    yield
        else:
            yield
    
    def record_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        strategy: str,
        venue: str,
        execution_duration: float,
        slippage_bps: float,
        pnl: float
    ) -> None:
        """Record trade metrics"""
        
        labels = {
            'symbol': symbol,
            'side': side,
            'strategy': strategy,
            'venue': venue
        }
        
        # Count the trade
        self.increment_counter('trades_total', 1, labels)
        
        # Record trade size
        self.observe_histogram('trade_size_shares', quantity, {
            'symbol': symbol,
            'side': side
        })
        
        # Record execution time
        self.observe_histogram('trade_execution_duration_seconds', execution_duration, {
            'algorithm': 'market',  # Default
            'venue': venue
        })
        
        # Record slippage
        self.observe_histogram('execution_slippage_bps', slippage_bps, {
            'symbol': symbol,
            'algorithm': 'market'
        })
        
        # Update PnL
        current_pnl = self.get_metric('trades_pnl_total')
        if current_pnl:
            # This is approximate - in practice you'd track cumulative PnL
            self.set_gauge('trades_pnl_total', pnl, {
                'strategy': strategy,
                'symbol': symbol
            })
    
    def record_portfolio_update(
        self,
        strategy: str,
        total_value: float,
        positions: Dict[str, float],
        weights: Dict[str, float],
        drawdown: float
    ) -> None:
        """Record portfolio metrics"""
        
        # Update total value
        self.set_gauge('portfolio_value_usd', total_value, {
            'strategy': strategy,
            'account': 'main'
        })
        
        # Update drawdown
        self.set_gauge('portfolio_drawdown', drawdown, {
            'strategy': strategy
        })
        
        # Update positions and weights
        for symbol, position in positions.items():
            self.set_gauge('portfolio_positions', position, {
                'symbol': symbol,
                'strategy': strategy
            })
            
            weight = weights.get(symbol, 0)
            self.set_gauge('portfolio_weights', weight, {
                'symbol': symbol,
                'strategy': strategy
            })
    
    def record_risk_metrics(
        self,
        strategy: str,
        var_95: float,
        gross_exposure: float
    ) -> None:
        """Record risk metrics"""
        
        self.set_gauge('risk_var_95', var_95, {
            'strategy': strategy,
            'horizon': '1d'
        })
        
        self.set_gauge('risk_exposure_gross', gross_exposure, {
            'strategy': strategy
        })
    
    def record_risk_breach(
        self,
        limit_type: str,
        strategy: str
    ) -> None:
        """Record a risk limit breach"""
        
        self.increment_counter('risk_breaches_total', 1, {
            'limit_type': limit_type,
            'strategy': strategy
        })
    
    def record_market_data_update(
        self,
        symbol: str,
        data_type: str,
        source: str,
        latency_ms: float
    ) -> None:
        """Record market data metrics"""
        
        self.increment_counter('market_data_updates_total', 1, {
            'symbol': symbol,
            'data_type': data_type,
            'source': source
        })
        
        self.set_gauge('market_data_latency_ms', latency_ms, {
            'source': source,
            'symbol': symbol
        })
    
    def record_agent_status(
        self,
        agent_id: str,
        agent_type: str,
        is_healthy: bool
    ) -> None:
        """Record agent health status"""
        
        self.set_gauge('agent_status', 1 if is_healthy else 0, {
            'agent_id': agent_id,
            'agent_type': agent_type
        })
    
    def record_event_processed(
        self,
        event_type: str,
        source: str,
        status: str,
        processing_duration: float
    ) -> None:
        """Record event processing metrics"""
        
        self.increment_counter('events_processed_total', 1, {
            'event_type': event_type,
            'source': source,
            'status': status
        })
        
        self.observe_histogram('event_processing_duration_seconds', processing_duration, {
            'event_type': event_type
        })
    
    def record_model_prediction(
        self,
        model_name: str,
        model_version: str,
        prediction_type: str,
        accuracy: Optional[float] = None,
        inference_duration_ms: Optional[float] = None
    ) -> None:
        """Record ML model metrics"""
        
        self.increment_counter('model_predictions_total', 1, {
            'model_name': model_name,
            'prediction_type': prediction_type
        })
        
        if accuracy is not None:
            self.set_gauge('model_accuracy', accuracy, {
                'model_name': model_name,
                'model_version': model_version
            })
        
        if inference_duration_ms is not None:
            self.observe_histogram('model_inference_duration_ms', inference_duration_ms, {
                'model_name': model_name
            })
    
    def get_metrics_text(self) -> str:
        """Get metrics in Prometheus text format"""
        if PROMETHEUS_AVAILABLE:
            return generate_latest(self.registry).decode('utf-8')
        else:
            return "# Prometheus client not available\n"
    
    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get current metrics as dictionary (for debugging)"""
        
        metrics_dict = {}
        
        for name, metric in self._metrics.items():
            try:
                # This is a simplified representation
                metrics_dict[name] = {
                    'type': type(metric).__name__,
                    'description': getattr(metric, '_documentation', 'No description')
                }
            except Exception as e:
                metrics_dict[name] = {'error': str(e)}
        
        return metrics_dict


# Global metrics instance
trade_metrics: Optional[PrometheusMetrics] = None
portfolio_metrics: Optional[PrometheusMetrics] = None
execution_metrics: Optional[PrometheusMetrics] = None
risk_metrics: Optional[PrometheusMetrics] = None


def initialize_metrics(
    enable_http_server: bool = True,
    http_port: int = 9090
) -> PrometheusMetrics:
    """Initialize global metrics instance"""
    
    global trade_metrics, portfolio_metrics, execution_metrics, risk_metrics
    
    if not PROMETHEUS_AVAILABLE:
        logger.warning(
            "Prometheus client not available. "
            "Install prometheus_client package to enable metrics collection."
        )
    
    metrics = PrometheusMetrics(
        enable_http_server=enable_http_server,
        http_port=http_port
    )
    
    # Set global references
    trade_metrics = metrics
    portfolio_metrics = metrics
    execution_metrics = metrics
    risk_metrics = metrics
    
    return metrics


def timing_decorator(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to time function execution"""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if trade_metrics:
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    trade_metrics.observe_histogram(metric_name, duration, labels)
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    error_labels = (labels or {}).copy()
                    error_labels['status'] = 'error'
                    trade_metrics.observe_histogram(metric_name, duration, error_labels)
                    raise
            else:
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def counter_decorator(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to count function calls"""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if trade_metrics:
                try:
                    result = func(*args, **kwargs)
                    success_labels = (labels or {}).copy()
                    success_labels['status'] = 'success'
                    trade_metrics.increment_counter(metric_name, 1, success_labels)
                    return result
                except Exception as e:
                    error_labels = (labels or {}).copy()
                    error_labels['status'] = 'error'
                    trade_metrics.increment_counter(metric_name, 1, error_labels)
                    raise
            else:
                return func(*args, **kwargs)
        
        return wrapper
    return decorator