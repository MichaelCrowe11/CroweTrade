"""Signal Gating Module - Stateless Signal Filtering"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class SignalStatus(Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    PENDING = "pending"


@dataclass
class Signal:
    signal_id: str
    symbol: str
    timestamp: datetime
    direction: str
    strength: float
    metadata: dict[str, Any] | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "direction": self.direction,
            "strength": self.strength,
            "metadata": self.metadata
        }


@dataclass
class GateResult:
    signal_id: str
    status: SignalStatus
    reasons: list[str]
    adjusted_signal: Signal | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "status": self.status.value,
            "reasons": self.reasons,
            "adjusted_signal": self.adjusted_signal.to_dict() if self.adjusted_signal else None
        }


class SignalGate:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.filters = []
        
    def add_filter(self, filter_fn: Any) -> None:
        self.filters.append(filter_fn)
        
    def process(self, signal: Signal, context: dict[str, Any]) -> GateResult:
        reasons = []
        status = SignalStatus.APPROVED
        adjusted_signal = signal
        
        for filter_fn in self.filters:
            result = filter_fn(adjusted_signal, context)
            
            if not result["passed"]:
                status = SignalStatus.REJECTED
                reasons.append(result.get("reason", "Filter failed"))
                break
                
            if "adjusted_signal" in result:
                adjusted_signal = result["adjusted_signal"]
                reasons.append(f"Signal adjusted: {result.get('adjustment_reason', 'Unknown')}")
        
        return GateResult(
            signal_id=signal.signal_id,
            status=status,
            reasons=reasons,
            adjusted_signal=adjusted_signal if status == SignalStatus.APPROVED else None
        )
    
    def batch_process(self, signals: list[Signal], context: dict[str, Any]) -> list[GateResult]:
        return [self.process(signal, context) for signal in signals]