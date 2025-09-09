"""Smart Order Router with venue scoring and intelligent routing."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from crowetrade.core.types import OrderId
from crowetrade.execution.brokers.base import Order, OrderStatus, OrderType

logger = logging.getLogger(__name__)


class VenueType(Enum):
    """Venue types."""
    
    EXCHANGE = "exchange"
    DARK_POOL = "dark_pool"
    BROKER = "broker"
    ECN = "ecn"
    SLP = "slp"  # Supplemental Liquidity Provider


@dataclass
class VenueMetrics:
    """Venue performance metrics."""
    
    venue_name: str
    venue_type: VenueType
    fill_rate: float = 1.0
    avg_latency_ms: float = 0.0
    avg_spread_bps: float = 0.0
    avg_impact_bps: float = 0.0
    rejection_rate: float = 0.0
    availability: float = 1.0
    total_volume: float = 0.0
    total_orders: int = 0
    last_update: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RoutingDecision:
    """Smart routing decision."""
    
    order: Order
    venue_allocations: Dict[str, float]  # venue -> percentage
    strategy: str
    scores: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.utcnow)


class VenueScorer:
    """Scores venues based on historical performance."""
    
    def __init__(
        self,
        latency_weight: float = 0.2,
        spread_weight: float = 0.3,
        impact_weight: float = 0.3,
        fill_weight: float = 0.2,
    ):
        """Initialize venue scorer.
        
        Args:
            latency_weight: Weight for latency score
            spread_weight: Weight for spread cost
            impact_weight: Weight for market impact
            fill_weight: Weight for fill rate
        """
        self.latency_weight = latency_weight
        self.spread_weight = spread_weight
        self.impact_weight = impact_weight
        self.fill_weight = fill_weight
        
        # Normalize weights
        total = latency_weight + spread_weight + impact_weight + fill_weight
        self.latency_weight /= total
        self.spread_weight /= total
        self.impact_weight /= total
        self.fill_weight /= total
    
    def score_venue(
        self,
        metrics: VenueMetrics,
        order_size: float,
        urgency: str = "normal",
    ) -> float:
        """Score a venue for an order.
        
        Args:
            metrics: Venue metrics
            order_size: Order size
            urgency: Order urgency
            
        Returns:
            Venue score (0-1, higher is better)
        """
        # Latency score (lower is better)
        latency_score = 1.0 / (1.0 + metrics.avg_latency_ms / 100)
        
        # Spread score (lower is better)
        spread_score = 1.0 / (1.0 + metrics.avg_spread_bps / 10)
        
        # Impact score (lower is better, adjusted for size)
        size_factor = np.sqrt(order_size / 1000)  # Sqrt for concave impact
        expected_impact = metrics.avg_impact_bps * size_factor
        impact_score = 1.0 / (1.0 + expected_impact / 10)
        
        # Fill rate score
        fill_score = metrics.fill_rate
        
        # Availability score
        availability_score = metrics.availability
        
        # Adjust weights based on urgency
        if urgency == "urgent":
            # Prioritize latency and fill rate
            latency_w = self.latency_weight * 1.5
            fill_w = self.fill_weight * 1.5
            spread_w = self.spread_weight * 0.5
            impact_w = self.impact_weight * 0.5
        elif urgency == "patient":
            # Prioritize cost
            latency_w = self.latency_weight * 0.5
            fill_w = self.fill_weight * 0.8
            spread_w = self.spread_weight * 1.5
            impact_w = self.impact_weight * 1.5
        else:
            latency_w = self.latency_weight
            fill_w = self.fill_weight
            spread_w = self.spread_weight
            impact_w = self.impact_weight
        
        # Calculate weighted score
        score = (
            latency_w * latency_score +
            spread_w * spread_score +
            impact_w * impact_score +
            fill_w * fill_score
        ) * availability_score
        
        return score


class SmartOrderRouter:
    """Intelligent order router with venue selection and allocation."""
    
    def __init__(self):
        """Initialize smart order router."""
        self.venues: Dict[str, VenueMetrics] = {}
        self.venue_adapters: Dict[str, Any] = {}
        self.scorer = VenueScorer()
        
        # Routing strategies
        self.strategies = {
            "best_execution": self._route_best_execution,
            "sweep": self._route_sweep,
            "iceberg": self._route_iceberg,
            "dark_first": self._route_dark_first,
            "spray": self._route_spray,
        }
        
        # Performance tracking
        self.routing_history: List[RoutingDecision] = []
        self.pending_orders: Dict[OrderId, RoutingDecision] = {}
    
    def register_venue(
        self,
        name: str,
        venue_type: VenueType,
        adapter: Any,
        initial_metrics: Optional[VenueMetrics] = None,
    ) -> None:
        """Register a venue with the router.
        
        Args:
            name: Venue name
            venue_type: Type of venue
            adapter: Venue adapter for order submission
            initial_metrics: Initial performance metrics
        """
        if initial_metrics:
            self.venues[name] = initial_metrics
        else:
            self.venues[name] = VenueMetrics(
                venue_name=name,
                venue_type=venue_type,
            )
        
        self.venue_adapters[name] = adapter
        logger.info(f"Registered venue {name} ({venue_type.value})")
    
    async def route_order(
        self,
        order: Order,
        strategy: str = "best_execution",
        urgency: str = "normal",
        constraints: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        """Route an order intelligently across venues.
        
        Args:
            order: Order to route
            strategy: Routing strategy
            urgency: Order urgency
            constraints: Additional constraints
            
        Returns:
            Routing decision with venue allocations
        """
        # Get routing function
        route_func = self.strategies.get(strategy, self._route_best_execution)
        
        # Score venues
        venue_scores = self._score_venues(order, urgency, constraints)
        
        # Generate routing decision
        decision = await route_func(order, venue_scores, constraints)
        
        # Record decision
        self.routing_history.append(decision)
        self.pending_orders[order.order_id] = decision
        
        # Execute routing
        await self._execute_routing(decision)
        
        return decision
    
    def _score_venues(
        self,
        order: Order,
        urgency: str,
        constraints: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Score all venues for an order.
        
        Args:
            order: Order to route
            urgency: Order urgency
            constraints: Routing constraints
            
        Returns:
            Venue scores
        """
        scores = {}
        
        for name, metrics in self.venues.items():
            # Check constraints
            if constraints:
                # Venue type constraint
                if "venue_types" in constraints:
                    if metrics.venue_type not in constraints["venue_types"]:
                        continue
                
                # Exclude venues
                if "exclude" in constraints:
                    if name in constraints["exclude"]:
                        continue
                
                # Min fill rate
                if "min_fill_rate" in constraints:
                    if metrics.fill_rate < constraints["min_fill_rate"]:
                        continue
            
            # Score venue
            score = self.scorer.score_venue(
                metrics,
                order.quantity,
                urgency,
            )
            
            scores[name] = score
        
        return scores
    
    async def _route_best_execution(
        self,
        order: Order,
        venue_scores: Dict[str, float],
        constraints: Optional[Dict[str, Any]],
    ) -> RoutingDecision:
        """Route for best execution to top venue.
        
        Args:
            order: Order to route
            venue_scores: Venue scores
            constraints: Constraints
            
        Returns:
            Routing decision
        """
        if not venue_scores:
            raise ValueError("No eligible venues for order")
        
        # Select best venue
        best_venue = max(venue_scores, key=venue_scores.get)
        
        return RoutingDecision(
            order=order,
            venue_allocations={best_venue: 1.0},
            strategy="best_execution",
            scores=venue_scores,
        )
    
    async def _route_sweep(
        self,
        order: Order,
        venue_scores: Dict[str, float],
        constraints: Optional[Dict[str, Any]],
    ) -> RoutingDecision:
        """Sweep order across multiple venues simultaneously.
        
        Args:
            order: Order to route
            venue_scores: Venue scores
            constraints: Constraints
            
        Returns:
            Routing decision
        """
        # Use top N venues
        max_venues = constraints.get("max_venues", 5) if constraints else 5
        
        # Sort venues by score
        sorted_venues = sorted(
            venue_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:max_venues]
        
        if not sorted_venues:
            raise ValueError("No eligible venues for order")
        
        # Allocate proportionally to scores
        total_score = sum(score for _, score in sorted_venues)
        allocations = {
            venue: score / total_score
            for venue, score in sorted_venues
        }
        
        return RoutingDecision(
            order=order,
            venue_allocations=allocations,
            strategy="sweep",
            scores=venue_scores,
        )
    
    async def _route_iceberg(
        self,
        order: Order,
        venue_scores: Dict[str, float],
        constraints: Optional[Dict[str, Any]],
    ) -> RoutingDecision:
        """Route as iceberg order with hidden quantity.
        
        Args:
            order: Order to route
            venue_scores: Venue scores
            constraints: Constraints
            
        Returns:
            Routing decision
        """
        # Find venues supporting iceberg
        iceberg_venues = {
            name: score
            for name, score in venue_scores.items()
            if self.venues[name].venue_type in [
                VenueType.EXCHANGE,
                VenueType.ECN,
            ]
        }
        
        if not iceberg_venues:
            # Fallback to best execution
            return await self._route_best_execution(order, venue_scores, constraints)
        
        # Use best iceberg venue
        best_venue = max(iceberg_venues, key=iceberg_venues.get)
        
        return RoutingDecision(
            order=order,
            venue_allocations={best_venue: 1.0},
            strategy="iceberg",
            scores=venue_scores,
        )
    
    async def _route_dark_first(
        self,
        order: Order,
        venue_scores: Dict[str, float],
        constraints: Optional[Dict[str, Any]],
    ) -> RoutingDecision:
        """Route to dark pools first for minimal impact.
        
        Args:
            order: Order to route
            venue_scores: Venue scores
            constraints: Constraints
            
        Returns:
            Routing decision
        """
        # Find dark pools
        dark_venues = {
            name: score
            for name, score in venue_scores.items()
            if self.venues[name].venue_type == VenueType.DARK_POOL
        }
        
        allocations = {}
        
        if dark_venues:
            # Allocate to dark pools first
            dark_allocation = min(0.7, constraints.get("max_dark", 0.7) if constraints else 0.7)
            
            # Distribute among dark pools
            total_dark_score = sum(dark_venues.values())
            for venue, score in dark_venues.items():
                allocations[venue] = dark_allocation * (score / total_dark_score)
            
            # Remaining to lit venues
            lit_allocation = 1.0 - dark_allocation
            lit_venues = {
                name: score
                for name, score in venue_scores.items()
                if name not in dark_venues
            }
            
            if lit_venues:
                best_lit = max(lit_venues, key=lit_venues.get)
                allocations[best_lit] = lit_allocation
        else:
            # No dark pools, use best execution
            return await self._route_best_execution(order, venue_scores, constraints)
        
        return RoutingDecision(
            order=order,
            venue_allocations=allocations,
            strategy="dark_first",
            scores=venue_scores,
        )
    
    async def _route_spray(
        self,
        order: Order,
        venue_scores: Dict[str, float],
        constraints: Optional[Dict[str, Any]],
    ) -> RoutingDecision:
        """Spray order across all venues to minimize signaling.
        
        Args:
            order: Order to route
            venue_scores: Venue scores
            constraints: Constraints
            
        Returns:
            Routing decision
        """
        if not venue_scores:
            raise ValueError("No eligible venues for order")
        
        # Equal allocation to all venues
        num_venues = len(venue_scores)
        equal_allocation = 1.0 / num_venues
        
        allocations = {
            venue: equal_allocation
            for venue in venue_scores
        }
        
        return RoutingDecision(
            order=order,
            venue_allocations=allocations,
            strategy="spray",
            scores=venue_scores,
        )
    
    async def _execute_routing(self, decision: RoutingDecision) -> None:
        """Execute routing decision by sending orders to venues.
        
        Args:
            decision: Routing decision to execute
        """
        tasks = []
        
        for venue, allocation in decision.venue_allocations.items():
            if allocation > 0 and venue in self.venue_adapters:
                # Calculate child order size
                child_size = decision.order.quantity * allocation
                
                # Create child order
                child_order = Order(
                    order_id=OrderId(f"{decision.order.order_id}_{venue}"),
                    symbol=decision.order.symbol,
                    quantity=child_size,
                    order_type=decision.order.order_type,
                    side=decision.order.side,
                    limit_price=decision.order.limit_price,
                    stop_price=decision.order.stop_price,
                    time_in_force=decision.order.time_in_force,
                    metadata={
                        "parent_id": decision.order.order_id,
                        "venue": venue,
                        "allocation": allocation,
                    },
                )
                
                # Submit to venue
                adapter = self.venue_adapters[venue]
                tasks.append(adapter.submit_order(child_order))
        
        # Execute all submissions in parallel
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def update_venue_metrics(
        self,
        venue: str,
        fill_rate: Optional[float] = None,
        latency_ms: Optional[float] = None,
        spread_bps: Optional[float] = None,
        impact_bps: Optional[float] = None,
        rejection: Optional[bool] = None,
    ) -> None:
        """Update venue metrics based on execution results.
        
        Args:
            venue: Venue name
            fill_rate: Fill rate update
            latency_ms: Latency observation
            spread_bps: Spread observation
            impact_bps: Impact observation
            rejection: Whether order was rejected
        """
        if venue not in self.venues:
            return
        
        metrics = self.venues[venue]
        decay = 0.95  # Exponential decay for updates
        
        if fill_rate is not None:
            metrics.fill_rate = decay * metrics.fill_rate + (1 - decay) * fill_rate
        
        if latency_ms is not None:
            metrics.avg_latency_ms = decay * metrics.avg_latency_ms + (1 - decay) * latency_ms
        
        if spread_bps is not None:
            metrics.avg_spread_bps = decay * metrics.avg_spread_bps + (1 - decay) * spread_bps
        
        if impact_bps is not None:
            metrics.avg_impact_bps = decay * metrics.avg_impact_bps + (1 - decay) * impact_bps
        
        if rejection is not None:
            if rejection:
                metrics.rejection_rate = decay * metrics.rejection_rate + (1 - decay)
            else:
                metrics.rejection_rate = decay * metrics.rejection_rate
        
        metrics.total_orders += 1
        metrics.last_update = datetime.utcnow()
    
    def get_venue_rankings(self) -> List[Tuple[str, float]]:
        """Get current venue rankings.
        
        Returns:
            List of (venue, score) tuples
        """
        # Score all venues with default order
        default_order = Order(
            order_id=OrderId("ranking"),
            symbol="DEFAULT",
            quantity=1000,
            order_type=OrderType.LIMIT,
            side="buy",
        )
        
        scores = self._score_venues(default_order, "normal", None)
        
        return sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )