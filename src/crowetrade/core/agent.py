from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .events import Event
from .types import AgentId, AgentState, PolicyId


@dataclass
class AgentConfig:
    agent_id: AgentId
    policy_id: PolicyId
    risk_limits: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    def __init__(self, config: AgentConfig):
        self.config = config
        self.state = AgentState.INIT
        self.start_time: datetime | None = None
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._handlers: dict[type, list[Callable]] = {}
        self._task: asyncio.Task | None = None
    
    @property
    def agent_id(self) -> AgentId:
        return self.config.agent_id
    
    @property
    def policy_id(self) -> PolicyId:
        return self.config.policy_id
    
    async def start(self):
        if self.state != AgentState.INIT:
            raise RuntimeError(f"Agent {self.agent_id} already started")
        
        self.state = AgentState.RUNNING
        self.start_time = datetime.utcnow()
        self._task = asyncio.create_task(self._run())
        await self.on_start()
    
    async def stop(self):
        if self.state != AgentState.RUNNING:
            return
        
        self.state = AgentState.STOPPED
        await self.on_stop()
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    
    async def pause(self):
        if self.state == AgentState.RUNNING:
            self.state = AgentState.PAUSED
            await self.on_pause()
    
    async def resume(self):
        if self.state == AgentState.PAUSED:
            self.state = AgentState.RUNNING
            await self.on_resume()
    
    async def emit(self, event: Event):
        event.source = self.agent_id
        await self._event_queue.put(event)
    
    def subscribe(self, event_type: type, handler: Callable):
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
    
    async def _run(self):
        while self.state in (AgentState.RUNNING, AgentState.PAUSED):
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(), 
                    timeout=0.1
                )
                
                if self.state == AgentState.RUNNING:
                    await self._handle_event(event)
                    
            except TimeoutError:
                continue
            except Exception as e:
                self.state = AgentState.ERROR
                await self.on_error(e)
    
    async def _handle_event(self, event: Event):
        handlers = self._handlers.get(type(event), [])
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                await self.on_error(e)
    
    @abstractmethod
    async def on_start(self):
        pass
    
    @abstractmethod
    async def on_stop(self):
        pass
    
    async def on_pause(self):
        pass
    
    async def on_resume(self):
        pass
    
    async def on_error(self, error: Exception):
        pass