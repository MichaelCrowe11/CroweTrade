from __future__ import annotations

from typing import Protocol


class Adapter(Protocol):
    async def submit_targets(
        self, targets: dict[str, float], prices: dict[str, float]
    ) -> None:  # pragma: no cover - protocol
        ...


class ExecutionRouter:
    """Routes target positions to broker/exchange adapters."""

    def __init__(self, adapters: list[Adapter]):
        self.adapters = adapters

    async def route(self, targets: dict[str, float], prices: dict[str, float]) -> None:
        for adapter in self.adapters:
            await adapter.submit_targets(targets, prices)
