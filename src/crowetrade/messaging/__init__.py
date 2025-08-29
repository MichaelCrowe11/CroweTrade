"""Message bus integration for CroweTrade platform."""

import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Protocol, Type
from dataclasses import asdict
from abc import ABC, abstractmethod

try:
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
    from aiokafka.errors import KafkaError
except ImportError:
    AIOKafkaConsumer = None
    AIOKafkaProducer = None
    KafkaError = Exception

from dataclasses import is_dataclass

logger = logging.getLogger(__name__)


class MessageBus(Protocol):
    """Protocol for message bus implementations."""
    
    async def publish(self, topic: str, event: Any) -> None:
        """Publish an event to a topic."""
        ...
    
    async def subscribe(
        self,
        topic: str,
        handler: Callable[[Any], None],
        event_type: Type[Any],
    ) -> None:
        """Subscribe to a topic with an event handler."""
        ...
    
    async def start(self) -> None:
        """Start the message bus."""
        ...
    
    async def stop(self) -> None:
        """Stop the message bus."""
        ...


class KafkaMessageBus:
    """Kafka-based message bus implementation."""
    
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        group_id: str = "crowetrade",
        auto_offset_reset: str = "latest",
        enable_auto_commit: bool = True,
        max_poll_records: int = 500,
        session_timeout_ms: int = 30000,
        heartbeat_interval_ms: int = 3000,
    ):
        if AIOKafkaConsumer is None:
            raise ImportError("aiokafka is required for KafkaMessageBus")
            
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.auto_offset_reset = auto_offset_reset
        self.enable_auto_commit = enable_auto_commit
        self.max_poll_records = max_poll_records
        self.session_timeout_ms = session_timeout_ms
        self.heartbeat_interval_ms = heartbeat_interval_ms
        
    self._producer = None
    self._consumer = None
    self._handlers: Dict[str, List[tuple[Callable, Type[Any]]]] = {}
    self._running = False
    self._tasks: List[asyncio.Task] = []
    
    async def start(self) -> None:
        """Start the Kafka message bus."""
        logger.info("Starting Kafka message bus")
        
        # Initialize producer
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            compression_type="snappy",
            acks='all',
            retries=3,
            max_in_flight_requests_per_connection=1,
            enable_idempotence=True,
        )
        await self._producer.start()
        
        # Initialize consumer if we have handlers
        if self._handlers:
            topics = list(self._handlers.keys())
            self._consumer = AIOKafkaConsumer(
                *topics,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset=self.auto_offset_reset,
                enable_auto_commit=self.enable_auto_commit,
                max_poll_records=self.max_poll_records,
                session_timeout_ms=self.session_timeout_ms,
                heartbeat_interval_ms=self.heartbeat_interval_ms,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            )
            await self._consumer.start()
            
            # Start consuming task
            self._running = True
            consume_task = asyncio.create_task(self._consume_loop())
            self._tasks.append(consume_task)
        
        logger.info("Kafka message bus started successfully")
    
    async def stop(self) -> None:
        """Stop the Kafka message bus."""
        logger.info("Stopping Kafka message bus")
        
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Stop consumer and producer
        if self._consumer:
            await self._consumer.stop()
        if self._producer:
            await self._producer.stop()
        
        logger.info("Kafka message bus stopped")
    
    async def publish(self, topic: str, event: Any) -> None:
        """Publish an event to a Kafka topic."""
        if not self._producer:
            raise RuntimeError("Message bus not started")
        
        try:
            # Convert event to dict for serialization
            if is_dataclass(event):
                payload = asdict(event)
            elif isinstance(event, dict):
                payload = event
            else:
                # Fallback: try best-effort serialization
                payload = json.loads(json.dumps(event, default=str))

            event_dict = {"event_type": event.__class__.__name__, "data": payload}
            
            await self._producer.send_and_wait(topic, event_dict)
            logger.debug(f"Published {event.__class__.__name__} to {topic}")
            
        except KafkaError as e:
            logger.error(f"Failed to publish event to {topic}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error publishing to {topic}: {e}")
            raise
    
    def subscribe(
        self,
        topic: str,
        handler: Callable[[Any], None],
        event_type: Type[Any],
    ) -> None:
        """Subscribe to a topic with an event handler."""
        if topic not in self._handlers:
            self._handlers[topic] = []
        
        self._handlers[topic].append((handler, event_type))
        logger.info(f"Subscribed to {topic} with {handler.__name__}")
    
    async def _consume_loop(self) -> None:
        """Main consumption loop."""
        if not self._consumer:
            return
        
        logger.info("Starting message consumption loop")
        
        try:
            async for msg in self._consumer:
                if not self._running:
                    break
                
                try:
                    await self._handle_message(msg)
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    # Don't break the loop for individual message errors
                    
        except asyncio.CancelledError:
            logger.info("Consumption loop cancelled")
        except Exception as e:
            logger.error(f"Error in consumption loop: {e}")
    
    async def _handle_message(self, msg) -> None:
        """Handle a single message."""
        topic = msg.topic
        handlers = self._handlers.get(topic, [])
        
        if not handlers:
            logger.warning(f"No handlers for topic {topic}")
            return
        
        try:
            event_dict = msg.value
            event_type_name = event_dict.get("event_type")
            event_data = event_dict.get("data", {})
            
            # Process with each handler
            for handler, expected_type in handlers:
                if event_type_name == expected_type.__name__:
                    try:
                        # Reconstruct the event object
                        event = expected_type(**event_data)
                        
                        # Call handler (async or sync)
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            handler(event)
                            
                        logger.debug(f"Processed {event_type_name} with {handler.__name__}")
                        
                    except Exception as e:
                        logger.error(f"Handler {handler.__name__} failed: {e}")
                        
        except Exception as e:
            logger.error(f"Failed to decode message from {topic}: {e}")


class InMemoryMessageBus:
    """In-memory message bus for testing and development."""
    
    def __init__(self):
        self._handlers: Dict[str, List[tuple[Callable, Type[Any]]]] = {}
        self._running = False
    
    async def start(self) -> None:
        """Start the in-memory message bus."""
        self._running = True
        logger.info("In-memory message bus started")
    
    async def stop(self) -> None:
        """Stop the in-memory message bus."""
        self._running = False
        logger.info("In-memory message bus stopped")
    
    async def publish(self, topic: str, event: Any) -> None:
        """Publish an event to handlers."""
        if not self._running:
            raise RuntimeError("Message bus not started")
        
        handlers = self._handlers.get(topic, [])
        
        for handler, expected_type in handlers:
            if isinstance(event, expected_type):
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                    logger.debug(f"Delivered {event.__class__.__name__} to {handler.__name__}")
                except Exception as e:
                    logger.error(f"Handler {handler.__name__} failed: {e}")
    
    def subscribe(
        self,
        topic: str,
        handler: Callable[[Any], None],
        event_type: Type[Any],
    ) -> None:
        """Subscribe to a topic with an event handler."""
        if topic not in self._handlers:
            self._handlers[topic] = []
        
        self._handlers[topic].append((handler, event_type))
        logger.info(f"Subscribed to {topic} with {handler.__name__}")


def create_message_bus(config: Dict[str, Any]) -> MessageBus:
    """Factory function to create a message bus based on configuration."""
    bus_type = config.get("type", "kafka")
    
    if bus_type == "kafka":
        return KafkaMessageBus(
            bootstrap_servers=config.get("bootstrap_servers", "localhost:9092"),
            group_id=config.get("group_id", "crowetrade"),
            auto_offset_reset=config.get("auto_offset_reset", "latest"),
        )
    elif bus_type == "memory":
        return InMemoryMessageBus()
    else:
        raise ValueError(f"Unknown message bus type: {bus_type}")
