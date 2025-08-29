from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import NewType, TypeAlias

OrderId = NewType('OrderId', str)
RequestId = NewType('RequestId', str)
PolicyId = NewType('PolicyId', str)
AgentId = NewType('AgentId', str)
Symbol: TypeAlias = str
Venue: TypeAlias = str
Timestamp: TypeAlias = datetime


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class TimeInForce(Enum):
    DAY = "DAY"
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"


class AgentState(Enum):
    INIT = "INIT"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    ERROR = "ERROR"
    STOPPED = "STOPPED"