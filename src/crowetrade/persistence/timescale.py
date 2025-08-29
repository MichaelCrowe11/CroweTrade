"""TimescaleDB persistence layer for time-series data."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import asyncpg
import numpy as np

from crowetrade.config.settings import DatabaseConfig
from crowetrade.data.market_data import Bar, OrderBook, Quote, Trade

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesPoint:
    """Generic time-series data point."""
    
    timestamp: datetime
    symbol: str
    metric_name: str
    value: float
    metadata: Optional[Dict[str, Any]] = None


class TimescaleDB:
    """TimescaleDB client for time-series data persistence."""
    
    def __init__(self, config: DatabaseConfig):
        """Initialize TimescaleDB client.
        
        Args:
            config: Database configuration
        """
        self.config = config
        self.pool: Optional[asyncpg.Pool] = None
        self._initialized = False
    
    async def connect(self) -> None:
        """Connect to TimescaleDB and initialize schema."""
        logger.info("Connecting to TimescaleDB")
        
        # Parse connection URL
        # Format: postgresql://user:pass@host:port/database
        self.pool = await asyncpg.create_pool(
            self.config.timescale_url,
            min_size=5,
            max_size=self.config.max_connections,
            command_timeout=self.config.connection_timeout,
        )
        
        # Initialize schema if needed
        if not self._initialized:
            await self._initialize_schema()
            self._initialized = True
        
        logger.info("Connected to TimescaleDB")
    
    async def disconnect(self) -> None:
        """Disconnect from TimescaleDB."""
        if self.pool:
            await self.pool.close()
            self.pool = None
        logger.info("Disconnected from TimescaleDB")
    
    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[asyncpg.Connection]:
        """Acquire a database connection from pool."""
        if not self.pool:
            raise RuntimeError("Not connected to database")
        
        async with self.pool.acquire() as conn:
            yield conn
    
    async def _initialize_schema(self) -> None:
        """Initialize database schema with hypertables."""
        async with self.acquire() as conn:
            # Create TimescaleDB extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")
            
            # Create trades table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    time TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    price DOUBLE PRECISION NOT NULL,
                    size DOUBLE PRECISION NOT NULL,
                    exchange TEXT,
                    conditions TEXT[],
                    trade_id TEXT
                )
            """)
            
            # Convert to hypertable
            await conn.execute("""
                SELECT create_hypertable('trades', 'time', 
                    if_not_exists => TRUE,
                    chunk_time_interval => INTERVAL '1 day')
            """)
            
            # Create index
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_symbol_time 
                ON trades (symbol, time DESC)
            """)
            
            # Create quotes table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS quotes (
                    time TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    bid DOUBLE PRECISION NOT NULL,
                    ask DOUBLE PRECISION NOT NULL,
                    bid_size DOUBLE PRECISION NOT NULL,
                    ask_size DOUBLE PRECISION NOT NULL,
                    exchange TEXT
                )
            """)
            
            await conn.execute("""
                SELECT create_hypertable('quotes', 'time',
                    if_not_exists => TRUE,
                    chunk_time_interval => INTERVAL '1 day')
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_quotes_symbol_time 
                ON quotes (symbol, time DESC)
            """)
            
            # Create bars table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS bars (
                    time TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    open DOUBLE PRECISION NOT NULL,
                    high DOUBLE PRECISION NOT NULL,
                    low DOUBLE PRECISION NOT NULL,
                    close DOUBLE PRECISION NOT NULL,
                    volume DOUBLE PRECISION NOT NULL,
                    vwap DOUBLE PRECISION,
                    trades INTEGER
                )
            """)
            
            await conn.execute("""
                SELECT create_hypertable('bars', 'time',
                    if_not_exists => TRUE,
                    chunk_time_interval => INTERVAL '1 month')
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_bars_symbol_timeframe_time 
                ON bars (symbol, timeframe, time DESC)
            """)
            
            # Create order books table (snapshots)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS order_books (
                    time TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    bids JSONB NOT NULL,
                    asks JSONB NOT NULL,
                    sequence BIGINT
                )
            """)
            
            await conn.execute("""
                SELECT create_hypertable('order_books', 'time',
                    if_not_exists => TRUE,
                    chunk_time_interval => INTERVAL '1 hour')
            """)
            
            # Create metrics table for generic time-series
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    time TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    value DOUBLE PRECISION NOT NULL,
                    metadata JSONB
                )
            """)
            
            await conn.execute("""
                SELECT create_hypertable('metrics', 'time',
                    if_not_exists => TRUE,
                    chunk_time_interval => INTERVAL '1 day')
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_symbol_metric_time 
                ON metrics (symbol, metric_name, time DESC)
            """)
            
            # Create continuous aggregates for bars
            await conn.execute("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS bars_5m
                WITH (timescaledb.continuous) AS
                SELECT 
                    time_bucket('5 minutes', time) AS bucket,
                    symbol,
                    first(open, time) AS open,
                    max(high) AS high,
                    min(low) AS low,
                    last(close, time) AS close,
                    sum(volume) AS volume
                FROM bars
                WHERE timeframe = '1m'
                GROUP BY bucket, symbol
                WITH NO DATA
            """)
            
            # Create compression policies
            await conn.execute("""
                SELECT add_compression_policy('trades', 
                    INTERVAL '7 days',
                    if_not_exists => TRUE)
            """)
            
            await conn.execute("""
                SELECT add_compression_policy('quotes', 
                    INTERVAL '3 days',
                    if_not_exists => TRUE)
            """)
            
            logger.info("TimescaleDB schema initialized")
    
    async def insert_trade(self, trade: Trade) -> None:
        """Insert trade data.
        
        Args:
            trade: Trade to insert
        """
        async with self.acquire() as conn:
            await conn.execute("""
                INSERT INTO trades (time, symbol, price, size, exchange, conditions, trade_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, trade.timestamp, trade.symbol, trade.price, trade.size,
                trade.exchange, trade.conditions, trade.trade_id)
    
    async def insert_trades_batch(self, trades: List[Trade]) -> None:
        """Insert multiple trades efficiently.
        
        Args:
            trades: List of trades to insert
        """
        if not trades:
            return
        
        async with self.acquire() as conn:
            await conn.executemany("""
                INSERT INTO trades (time, symbol, price, size, exchange, conditions, trade_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, [(t.timestamp, t.symbol, t.price, t.size, t.exchange, 
                   t.conditions, t.trade_id) for t in trades])
    
    async def insert_quote(self, quote: Quote) -> None:
        """Insert quote data.
        
        Args:
            quote: Quote to insert
        """
        async with self.acquire() as conn:
            await conn.execute("""
                INSERT INTO quotes (time, symbol, bid, ask, bid_size, ask_size, exchange)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, quote.timestamp, quote.symbol, quote.bid, quote.ask,
                quote.bid_size, quote.ask_size, quote.exchange)
    
    async def insert_bar(self, bar: Bar, timeframe: str = "1m") -> None:
        """Insert bar data.
        
        Args:
            bar: Bar to insert
            timeframe: Bar timeframe
        """
        async with self.acquire() as conn:
            await conn.execute("""
                INSERT INTO bars (time, symbol, timeframe, open, high, low, close, volume, vwap, trades)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """, bar.timestamp, bar.symbol, timeframe, bar.open, bar.high,
                bar.low, bar.close, bar.volume, bar.vwap, bar.trades)
    
    async def insert_order_book(self, book: OrderBook) -> None:
        """Insert order book snapshot.
        
        Args:
            book: Order book to insert
        """
        # Convert to JSON for storage
        bids_json = [
            {"price": level.price, "size": level.size, "orders": level.orders}
            for level in book.bids
        ]
        asks_json = [
            {"price": level.price, "size": level.size, "orders": level.orders}
            for level in book.asks
        ]
        
        async with self.acquire() as conn:
            await conn.execute("""
                INSERT INTO order_books (time, symbol, bids, asks, sequence)
                VALUES ($1, $2, $3::jsonb, $4::jsonb, $5)
            """, book.timestamp, book.symbol, bids_json, asks_json, book.sequence)
    
    async def insert_metric(self, metric: TimeSeriesPoint) -> None:
        """Insert generic metric.
        
        Args:
            metric: Metric to insert
        """
        async with self.acquire() as conn:
            await conn.execute("""
                INSERT INTO metrics (time, symbol, metric_name, value, metadata)
                VALUES ($1, $2, $3, $4, $5::jsonb)
            """, metric.timestamp, metric.symbol, metric.metric_name,
                metric.value, metric.metadata)
    
    async def get_recent_trades(
        self,
        symbol: str,
        limit: int = 100,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Trade]:
        """Get recent trades for symbol.
        
        Args:
            symbol: Symbol to query
            limit: Maximum number of trades
            start_time: Start time filter
            end_time: End time filter
            
        Returns:
            List of trades
        """
        async with self.acquire() as conn:
            query = "SELECT * FROM trades WHERE symbol = $1"
            params = [symbol]
            
            if start_time:
                query += f" AND time >= ${len(params) + 1}"
                params.append(start_time)
            
            if end_time:
                query += f" AND time <= ${len(params) + 1}"
                params.append(end_time)
            
            query += f" ORDER BY time DESC LIMIT ${len(params) + 1}"
            params.append(limit)
            
            rows = await conn.fetch(query, *params)
            
            return [
                Trade(
                    symbol=row["symbol"],
                    price=row["price"],
                    size=row["size"],
                    timestamp=row["time"],
                    exchange=row["exchange"] or "",
                    conditions=row["conditions"] or [],
                    trade_id=row["trade_id"],
                )
                for row in rows
            ]
    
    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
    ) -> List[Bar]:
        """Get bars for symbol and timeframe.
        
        Args:
            symbol: Symbol to query
            timeframe: Bar timeframe
            start_time: Start time
            end_time: End time
            
        Returns:
            List of bars
        """
        async with self.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM bars 
                WHERE symbol = $1 AND timeframe = $2 
                AND time >= $3 AND time <= $4
                ORDER BY time ASC
            """, symbol, timeframe, start_time, end_time)
            
            return [
                Bar(
                    symbol=row["symbol"],
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"],
                    timestamp=row["time"],
                    vwap=row["vwap"],
                    trades=row["trades"] or 0,
                )
                for row in rows
            ]
    
    async def get_latest_quote(self, symbol: str) -> Optional[Quote]:
        """Get latest quote for symbol.
        
        Args:
            symbol: Symbol to query
            
        Returns:
            Latest quote or None
        """
        async with self.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM quotes 
                WHERE symbol = $1 
                ORDER BY time DESC 
                LIMIT 1
            """, symbol)
            
            if row:
                return Quote(
                    symbol=row["symbol"],
                    bid=row["bid"],
                    ask=row["ask"],
                    bid_size=row["bid_size"],
                    ask_size=row["ask_size"],
                    timestamp=row["time"],
                    exchange=row["exchange"] or "",
                )
            return None
    
    async def calculate_vwap(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
    ) -> float:
        """Calculate VWAP for period.
        
        Args:
            symbol: Symbol to calculate
            start_time: Start time
            end_time: End time
            
        Returns:
            VWAP value
        """
        async with self.acquire() as conn:
            result = await conn.fetchval("""
                SELECT 
                    SUM(price * size) / NULLIF(SUM(size), 0) AS vwap
                FROM trades
                WHERE symbol = $1 AND time >= $2 AND time <= $3
            """, symbol, start_time, end_time)
            
            return result or 0.0
    
    async def get_metrics(
        self,
        symbol: str,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
    ) -> List[TimeSeriesPoint]:
        """Get metrics for symbol.
        
        Args:
            symbol: Symbol to query
            metric_name: Metric name
            start_time: Start time
            end_time: End time
            
        Returns:
            List of metrics
        """
        async with self.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM metrics
                WHERE symbol = $1 AND metric_name = $2
                AND time >= $3 AND time <= $4
                ORDER BY time ASC
            """, symbol, metric_name, start_time, end_time)
            
            return [
                TimeSeriesPoint(
                    timestamp=row["time"],
                    symbol=row["symbol"],
                    metric_name=row["metric_name"],
                    value=row["value"],
                    metadata=row["metadata"],
                )
                for row in rows
            ]
    
    async def cleanup_old_data(self, retention_days: int = 30) -> None:
        """Clean up old data beyond retention period.
        
        Args:
            retention_days: Number of days to retain
        """
        cutoff = datetime.utcnow() - timedelta(days=retention_days)
        
        async with self.acquire() as conn:
            # Drop old chunks (most efficient for TimescaleDB)
            await conn.execute("""
                SELECT drop_chunks('trades', older_than => $1::timestamptz)
            """, cutoff)
            
            await conn.execute("""
                SELECT drop_chunks('quotes', older_than => $1::timestamptz)
            """, cutoff)
            
            logger.info(f"Cleaned up data older than {cutoff}")