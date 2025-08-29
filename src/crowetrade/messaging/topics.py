"""Topic management and schema validation for CroweTrade messaging."""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

try:
    from kafka.admin import KafkaAdminClient, ConfigResource, ConfigResourceType
    from kafka.admin.config_resource import ConfigResource
    from kafka import TopicPartition
    from kafka.structs import NewTopic
    from kafka.errors import TopicAlreadyExistsError, KafkaError
except ImportError:
    KafkaAdminClient = None
    NewTopic = None
    TopicAlreadyExistsError = Exception
    KafkaError = Exception

logger = logging.getLogger(__name__)


class TopicType(Enum):
    """Topic types with different characteristics."""
    MARKET_DATA = "market-data"
    SIGNALS = "signals"
    PORTFOLIO = "portfolio"
    EXECUTION = "execution"
    RISK = "risk"
    AUDIT = "audit"


@dataclass
class TopicConfig:
    """Configuration for a Kafka topic."""
    name: str
    partitions: int
    replication_factor: int
    cleanup_policy: str = "delete"
    retention_ms: Optional[int] = None
    compression_type: str = "snappy"
    min_in_sync_replicas: int = 1
    unclean_leader_election: bool = False
    segment_ms: Optional[int] = None
    max_message_bytes: int = 1048588  # ~1MB


class TopicManager:
    """Manages Kafka topics for the CroweTrade platform."""
    
    # Predefined topic configurations
    TOPIC_CONFIGS = {
        TopicType.MARKET_DATA: TopicConfig(
            name="market-data",
            partitions=12,
            replication_factor=3,
            retention_ms=7 * 24 * 60 * 60 * 1000,  # 7 days
            segment_ms=60 * 60 * 1000,  # 1 hour segments
            compression_type="lz4",
            max_message_bytes=10485760,  # 10MB for market data
        ),
        TopicType.SIGNALS: TopicConfig(
            name="signals",
            partitions=6,
            replication_factor=3,
            retention_ms=24 * 60 * 60 * 1000,  # 1 day
            compression_type="snappy",
        ),
        TopicType.PORTFOLIO: TopicConfig(
            name="portfolio-targets",
            partitions=3,
            replication_factor=3,
            retention_ms=30 * 24 * 60 * 60 * 1000,  # 30 days
            cleanup_policy="compact",  # Keep latest state
            min_in_sync_replicas=2,
        ),
        TopicType.EXECUTION: TopicConfig(
            name="executions",
            partitions=6,
            replication_factor=3,
            retention_ms=90 * 24 * 60 * 60 * 1000,  # 90 days
            compression_type="snappy",
            min_in_sync_replicas=2,
        ),
        TopicType.RISK: TopicConfig(
            name="risk-events",
            partitions=3,
            replication_factor=3,
            retention_ms=365 * 24 * 60 * 60 * 1000,  # 1 year
            min_in_sync_replicas=2,
            unclean_leader_election=False,  # Never allow data loss
        ),
        TopicType.AUDIT: TopicConfig(
            name="audit-trail",
            partitions=6,
            replication_factor=3,
            retention_ms=7 * 365 * 24 * 60 * 60 * 1000,  # 7 years
            cleanup_policy="compact",
            min_in_sync_replicas=3,  # High durability
            unclean_leader_election=False,
        ),
    }
    
    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        if KafkaAdminClient is None:
            raise ImportError("kafka-python is required for TopicManager")
            
        self.bootstrap_servers = bootstrap_servers
        self._admin_client: Optional[KafkaAdminClient] = None
    
    def connect(self) -> None:
        """Connect to Kafka cluster."""
        self._admin_client = KafkaAdminClient(
            bootstrap_servers=self.bootstrap_servers,
            client_id="crowetrade-topic-manager"
        )
        logger.info(f"Connected to Kafka cluster at {self.bootstrap_servers}")
    
    def disconnect(self) -> None:
        """Disconnect from Kafka cluster."""
        if self._admin_client:
            self._admin_client.close()
            self._admin_client = None
        logger.info("Disconnected from Kafka cluster")
    
    def create_all_topics(self) -> None:
        """Create all predefined topics."""
        if not self._admin_client:
            raise RuntimeError("Not connected to Kafka")
        
        topics_to_create = []
        
        for topic_type, config in self.TOPIC_CONFIGS.items():
            topics_to_create.append(self._create_new_topic(config))
        
        try:
            fs = self._admin_client.create_topics(topics_to_create, validate_only=False)
            
            # Wait for creation to complete
            for topic, future in fs.items():
                try:
                    future.result()  # Block until topic is created
                    logger.info(f"Created topic: {topic}")
                except TopicAlreadyExistsError:
                    logger.info(f"Topic already exists: {topic}")
                except Exception as e:
                    logger.error(f"Failed to create topic {topic}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to create topics: {e}")
            raise
    
    def create_topic(self, topic_type: TopicType, custom_config: Optional[Dict[str, Any]] = None) -> None:
        """Create a single topic."""
        if not self._admin_client:
            raise RuntimeError("Not connected to Kafka")
        
        config = self.TOPIC_CONFIGS[topic_type]
        
        # Apply custom configuration overrides
        if custom_config:
            for key, value in custom_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        new_topic = self._create_new_topic(config)
        
        try:
            fs = self._admin_client.create_topics([new_topic], validate_only=False)
            fs[config.name].result()  # Block until created
            logger.info(f"Created topic: {config.name}")
            
        except TopicAlreadyExistsError:
            logger.info(f"Topic already exists: {config.name}")
        except Exception as e:
            logger.error(f"Failed to create topic {config.name}: {e}")
            raise
    
    def delete_topic(self, topic_name: str) -> None:
        """Delete a topic."""
        if not self._admin_client:
            raise RuntimeError("Not connected to Kafka")
        
        try:
            fs = self._admin_client.delete_topics([topic_name])
            fs[topic_name].result()
            logger.info(f"Deleted topic: {topic_name}")
            
        except Exception as e:
            logger.error(f"Failed to delete topic {topic_name}: {e}")
            raise
    
    def list_topics(self) -> List[str]:
        """List all topics in the cluster."""
        if not self._admin_client:
            raise RuntimeError("Not connected to Kafka")
        
        try:
            metadata = self._admin_client.list_topics()
            topics = list(metadata.topics.keys())
            return [t for t in topics if not t.startswith('__')]  # Filter internal topics
            
        except Exception as e:
            logger.error(f"Failed to list topics: {e}")
            raise
    
    def get_topic_config(self, topic_name: str) -> Dict[str, str]:
        """Get configuration for a topic."""
        if not self._admin_client:
            raise RuntimeError("Not connected to Kafka")
        
        try:
            resource = ConfigResource(ConfigResourceType.TOPIC, topic_name)
            fs = self._admin_client.describe_configs([resource])
            config = fs[resource].result()
            
            return {entry.name: entry.value for entry in config.values()}
            
        except Exception as e:
            logger.error(f"Failed to get config for topic {topic_name}: {e}")
            raise
    
    def update_topic_config(self, topic_name: str, config_updates: Dict[str, str]) -> None:
        """Update configuration for a topic."""
        if not self._admin_client:
            raise RuntimeError("Not connected to Kafka")
        
        try:
            resource = ConfigResource(ConfigResourceType.TOPIC, topic_name)
            configs = {resource: config_updates}
            fs = self._admin_client.alter_configs(configs)
            fs[resource].result()
            
            logger.info(f"Updated config for topic {topic_name}: {config_updates}")
            
        except Exception as e:
            logger.error(f"Failed to update config for topic {topic_name}: {e}")
            raise
    
    def _create_new_topic(self, config: TopicConfig) -> Any:
        """Create a NewTopic object from TopicConfig."""
        topic_configs = {}
        
        if config.cleanup_policy:
            topic_configs["cleanup.policy"] = config.cleanup_policy
        if config.retention_ms:
            topic_configs["retention.ms"] = str(config.retention_ms)
        if config.compression_type:
            topic_configs["compression.type"] = config.compression_type
        if config.min_in_sync_replicas:
            topic_configs["min.insync.replicas"] = str(config.min_in_sync_replicas)
        if config.segment_ms:
            topic_configs["segment.ms"] = str(config.segment_ms)
        if config.max_message_bytes:
            topic_configs["max.message.bytes"] = str(config.max_message_bytes)
        
        topic_configs["unclean.leader.election.enable"] = str(config.unclean_leader_election).lower()
        
        return NewTopic(
            name=config.name,
            num_partitions=config.partitions,
            replication_factor=config.replication_factor,
            topic_configs=topic_configs
        )


@dataclass
class SchemaDefinition:
    """Schema definition for message validation."""
    name: str
    version: int
    schema: Dict[str, Any]
    compatibility: str = "FORWARD"  # FORWARD, BACKWARD, FULL, NONE


class SchemaRegistry:
    """Simple schema registry for message validation."""
    
    def __init__(self):
        self._schemas: Dict[str, Dict[int, SchemaDefinition]] = {}
    
    def register_schema(self, schema: SchemaDefinition) -> None:
        """Register a schema."""
        if schema.name not in self._schemas:
            self._schemas[schema.name] = {}
        
        self._schemas[schema.name][schema.version] = schema
        logger.info(f"Registered schema {schema.name} v{schema.version}")
    
    def get_schema(self, name: str, version: Optional[int] = None) -> Optional[SchemaDefinition]:
        """Get a schema by name and version."""
        if name not in self._schemas:
            return None
        
        if version is None:
            # Return latest version
            latest_version = max(self._schemas[name].keys())
            return self._schemas[name][latest_version]
        
        return self._schemas[name].get(version)
    
    def list_schemas(self) -> List[str]:
        """List all schema names."""
        return list(self._schemas.keys())
    
    def validate_message(self, schema_name: str, message: Dict[str, Any], version: Optional[int] = None) -> bool:
        """Validate a message against a schema."""
        schema = self.get_schema(schema_name, version)
        if not schema:
            logger.warning(f"Schema {schema_name} not found")
            return False
        
        # Simple validation - in production, use a proper JSON schema validator
        try:
            return self._validate_dict(message, schema.schema)
        except Exception as e:
            logger.error(f"Validation failed for {schema_name}: {e}")
            return False
    
    def _validate_dict(self, data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """Simple recursive validation."""
        required_fields = schema.get("required", [])
        properties = schema.get("properties", {})
        
        # Check required fields
        for field in required_fields:
            if field not in data:
                return False
        
        # Check field types
        for field, field_schema in properties.items():
            if field in data:
                field_type = field_schema.get("type")
                if field_type and not self._check_type(data[field], field_type):
                    return False
        
        return True
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        
        expected_python_type = type_map.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        
        return True  # Unknown type, allow it
