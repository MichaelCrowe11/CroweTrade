from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from .events import Event
from .types import AgentId


@dataclass
class Subscription:
    subscriber_id: AgentId
    event_type: type
    handler: Callable
    filter_fn: Callable[[Event], bool] | None = None


class EventBus:
    def __init__(self):
        self._subscriptions: dict[type, list[Subscription]] = defaultdict(list)
        self._subscribers: dict[AgentId, set[type]] = defaultdict(set)
        self._event_history: list[Event] = []
        self._max_history = 10000
        self._lock = asyncio.Lock()
    
    async def subscribe(
        self,
        subscriber_id: AgentId,
        event_type: type,
        handler: Callable,
        filter_fn: Callable[[Event], bool] | None = None
    ):
        async with self._lock:
            sub = Subscription(
                subscriber_id=subscriber_id,
                event_type=event_type,
                handler=handler,
                filter_fn=filter_fn
            )
            self._subscriptions[event_type].append(sub)
            self._subscribers[subscriber_id].add(event_type)
    
    async def unsubscribe(self, subscriber_id: AgentId, event_type: type | None = None):
        async with self._lock:
            if event_type:
                self._subscriptions[event_type] = [
                    s for s in self._subscriptions[event_type]
                    if s.subscriber_id != subscriber_id
                ]
                self._subscribers[subscriber_id].discard(event_type)
            else:
                for evt_type in list(self._subscribers[subscriber_id]):
                    self._subscriptions[evt_type] = [
                        s for s in self._subscriptions[evt_type]
                        if s.subscriber_id != subscriber_id
                    ]
                del self._subscribers[subscriber_id]
    
    async def publish(self, event: Event):
        event.timestamp = datetime.utcnow()
        
        async with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)
            
            subscriptions = self._subscriptions.get(type(event), [])
        
        tasks = []
        for sub in subscriptions:
            if sub.filter_fn and not sub.filter_fn(event):
                continue
            
            task = asyncio.create_task(self._deliver(sub.handler, event))
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _deliver(self, handler: Callable, event: Event):
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)
        except Exception:
            pass
    
    def get_history(
        self,
        event_type: type | None = None,
        since: datetime | None = None,
        limit: int = 100
    ) -> list[Event]:
        history = self._event_history
        
        if event_type:
            history = [e for e in history if isinstance(e, event_type)]
        
        if since:
            history = [e for e in history if e.timestamp >= since]
        
        return history[-limit:]