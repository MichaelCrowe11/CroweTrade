from __future__ import annotations

import asyncio
import logging
from typing import Dict

from fastapi import FastAPI
from uvicorn import Config, Server
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest
from prometheus_client import Counter
from crowetrade.core.config import get_settings

app = FastAPI()
logger = logging.getLogger("execution-agent")

# Metrics
registry = CollectorRegistry()
requests_total = Counter("requests_total", "Total HTTP requests", ["path"], registry=registry)

@app.get("/health")
async def health() -> Dict[str, str]:
    requests_total.labels(path="/health").inc()
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    data = generate_latest(registry)
    return app.response_class(content=data, media_type=CONTENT_TYPE_LATEST)


async def main() -> None:
    settings = get_settings()
    logging.getLogger().setLevel(settings.log_level.upper())
    logger.info("Starting execution service", extra={
        "run_mode": settings.run_mode,
        "paper_mode": settings.paper_mode,
        "port": settings.port,
    })
    # Keep port 8080 to match Fly service mapping; can switch to settings.port if needed
    config = Config(app=app, host="0.0.0.0", port=8080, log_level=settings.log_level)
    server = Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
