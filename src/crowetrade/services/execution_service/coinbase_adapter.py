"""
Coinbase Pro Venue Adapter for CroweTrade AI Trading Infrastructure

This module provides comprehensive integration with Coinbase Pro (Advanced Trading API) 
for cryptocurrency trading with real-time market data, order execution, and wallet management.

Features:
- Real-time crypto market data feeds
- Order execution with Coinbase Pro API
- Portfolio and wallet management
- WebSocket streaming for live updates
- Crypto-specific risk controls
"""

import asyncio
import hashlib
import hmac
import json
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any
import aiohttp
import websockets
from dataclasses import dataclass, field

from crowetrade.core.contracts import Fill
from crowetrade.core.events import ChildOrder
from crowetrade.core.types import OrderSide, OrderType, OrderStatus
from crowetrade.services.execution_service.venue_adapter import VenueAdapter, VenueConfig


@dataclass 
class CoinbaseConfig(VenueConfig):
    """Configuration for Coinbase Pro integration"""
    api_key: str
    api_secret: str
    passphrase: str
    sandbox: bool = False
    base_url: str = field(init=False)
    ws_url: str = field(init=False)
    
    def __post_init__(self):
        if self.sandbox:
            self.base_url = "https://api-public.sandbox.pro.coinbase.com"
            self.ws_url = "wss://ws-feed-public.sandbox.pro.coinbase.com"
        else:
            self.base_url = "https://api.pro.coinbase.com"  
            self.ws_url = "wss://ws-feed.pro.coinbase.com"


@dataclass
class CoinbaseProduct:
    """Cryptocurrency trading pair information"""
    id: str  # e.g., "BTC-USD"
    base_currency: str  # e.g., "BTC"
    quote_currency: str  # e.g., "USD"
    base_min_size: Decimal
    base_max_size: Decimal
    quote_increment: Decimal
    base_increment: Decimal
    display_name: str
    min_market_funds: Decimal
    max_market_funds: Decimal
    margin_enabled: bool
    post_only: bool
    limit_only: bool
    cancel_only: bool
    trading_disabled: bool
    status: str
    status_message: str


@dataclass
class CoinbaseTicker:
    """Real-time market ticker data"""
    product_id: str
    price: Decimal
    size: Decimal
    bid: Decimal
    ask: Decimal
    volume: Decimal
    time: datetime
    trade_id: int
    
    
@dataclass
class CoinbaseAccount:
    """Coinbase account/wallet information"""
    id: str
    currency: str
    balance: Decimal
    available: Decimal
    hold: Decimal
    profile_id: str
    trading_enabled: bool


class CoinbaseProVenueAdapter(VenueAdapter):
    """Production Coinbase Pro venue adapter with comprehensive crypto trading capabilities"""
    
    def __init__(self, config: CoinbaseConfig):
        super().__init__(config)
        self.config: CoinbaseConfig = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws_connection: Optional[websockets.WebSocketServerProtocol] = None
        
        # Trading state
        self.products: Dict[str, CoinbaseProduct] = {}
        self.accounts: Dict[str, CoinbaseAccount] = {}
        self.tickers: Dict[str, CoinbaseTicker] = {}
        self.open_orders: Dict[str, Dict] = {}
        
        # Event queues
        self.fill_queue: List[Fill] = []
        self.market_data_queue: List[Dict] = []
        
        # Rate limiting
        self.last_request_time = 0
        self.request_count = 0
        self.rate_limit_window = 1.0  # 1 second
        self.max_requests_per_second = 10
        
        print(f"🏗️  Coinbase Pro Adapter initialized for {'sandbox' if config.sandbox else 'production'}")
    
    async def connect(self) -> bool:
        """Establish connection to Coinbase Pro"""
        try:
            # Initialize HTTP session
            connector = aiohttp.TCPConnector(limit=100, ttl_dns_cache=300)
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self.session = aiohttp.ClientSession(
                connector=connector, 
                timeout=timeout,
                headers={'User-Agent': 'CroweTrade-AI-Trading/1.0'}
            )
            
            # Test authentication
            accounts = await self._get_accounts()
            if accounts:
                print(f"✅ Coinbase Pro authenticated - {len(accounts)} accounts found")
                
                # Load trading products
                await self._load_products()
                
                # Establish WebSocket connection
                await self._connect_websocket()
                
                self.connected = True
                print(f"🚀 Coinbase Pro connection established - {len(self.products)} trading pairs available")
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Coinbase Pro connection failed: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Close connection to Coinbase Pro"""
        try:
            self.connected = False
            
            if self.ws_connection:
                await self.ws_connection.close()
                
            if self.session:
                await self.session.close()
                
            print("✅ Coinbase Pro disconnected")
            
        except Exception as e:
            print(f"⚠️ Error during Coinbase Pro disconnect: {e}")
    
    async def send_order(self, order: ChildOrder) -> str:
        """Send order to Coinbase Pro"""
        if not self.connected or not self.session:
            raise RuntimeError("Not connected to Coinbase Pro")
        
        try:
            # Rate limiting
            await self._rate_limit()
            
            # Validate product exists
            if order.instrument not in self.products:
                raise ValueError(f"Unknown product: {order.instrument}")
                
            product = self.products[order.instrument]
            
            # Build order parameters
            order_params = {
                "product_id": order.instrument,
                "side": "buy" if order.side == OrderSide.BUY else "sell",
                "client_oid": str(uuid.uuid4()),  # Unique client order ID
            }
            
            # Handle order types
            if order.order_type == OrderType.MARKET:
                if order.side == OrderSide.BUY:
                    # Market buy - specify funds (USD amount)
                    funds = float(order.qty * (order.limit_px or self.tickers.get(order.instrument, CoinbaseTicker("", Decimal("50000"), Decimal("0"), Decimal("0"), Decimal("0"), Decimal("0"), datetime.now(), 0)).price))
                    order_params["type"] = "market"
                    order_params["funds"] = str(funds)
                else:
                    # Market sell - specify size (crypto amount)
                    order_params["type"] = "market" 
                    order_params["size"] = str(order.qty)
                    
            elif order.order_type == OrderType.LIMIT:
                order_params.update({
                    "type": "limit",
                    "size": str(order.qty),
                    "price": str(order.limit_px),
                    "time_in_force": "GTC"  # Good Till Canceled
                })
                
            else:
                raise ValueError(f"Unsupported order type: {order.order_type}")
            
            # Submit order
            response = await self._authenticated_request("POST", "/orders", data=order_params)
            
            if response and "id" in response:
                order_id = response["id"]
                self.order_state[order_id] = "PENDING"
                
                print(f"📤 Coinbase order submitted: {order_id} | {order.instrument} {order.side.value} {order.qty}")
                return order_id
            else:
                raise Exception(f"Order submission failed: {response}")
                
        except Exception as e:
            print(f"❌ Coinbase order failed: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order on Coinbase Pro"""
        if not self.connected or not self.session:
            return False
            
        try:
            await self._rate_limit()
            
            response = await self._authenticated_request("DELETE", f"/orders/{order_id}")
            
            if response and "id" in response:
                self.order_state[order_id] = "CANCELLED"
                print(f"🚫 Coinbase order cancelled: {order_id}")
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Coinbase cancel failed: {e}")
            return False
    
    async def handle_fills(self) -> List[Fill]:
        """Process incoming fills from Coinbase Pro"""
        fills = self.fill_queue.copy()
        self.fill_queue.clear()
        return fills
    
    async def get_market_data(self) -> List[Dict]:
        """Get latest market data updates"""
        data = self.market_data_queue.copy()
        self.market_data_queue.clear()
        return data
    
    async def get_account_balances(self) -> Dict[str, CoinbaseAccount]:
        """Get current account balances"""
        if not self.connected:
            return {}
            
        try:
            accounts = await self._get_accounts()
            return {acc.currency: acc for acc in accounts}
            
        except Exception as e:
            print(f"❌ Failed to get balances: {e}")
            return {}
    
    async def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get order status from Coinbase Pro"""
        try:
            await self._rate_limit()
            response = await self._authenticated_request("GET", f"/orders/{order_id}")
            return response
            
        except Exception as e:
            print(f"❌ Failed to get order status: {e}")
            return None
    
    # Private methods for Coinbase Pro API integration
    
    async def _load_products(self):
        """Load available cryptocurrency trading pairs"""
        try:
            response = await self._public_request("GET", "/products")
            
            if response:
                for product_data in response:
                    if product_data.get("status") == "online" and not product_data.get("trading_disabled", False):
                        product = CoinbaseProduct(
                            id=product_data["id"],
                            base_currency=product_data["base_currency"],
                            quote_currency=product_data["quote_currency"],
                            base_min_size=Decimal(product_data["base_min_size"]),
                            base_max_size=Decimal(product_data["base_max_size"]),
                            quote_increment=Decimal(product_data["quote_increment"]),
                            base_increment=Decimal(product_data["base_increment"]),
                            display_name=product_data["display_name"],
                            min_market_funds=Decimal(product_data["min_market_funds"]),
                            max_market_funds=Decimal(product_data["max_market_funds"]),
                            margin_enabled=product_data["margin_enabled"],
                            post_only=product_data["post_only"],
                            limit_only=product_data["limit_only"],
                            cancel_only=product_data["cancel_only"],
                            trading_disabled=product_data["trading_disabled"],
                            status=product_data["status"],
                            status_message=product_data.get("status_message", "")
                        )
                        self.products[product.id] = product
                        
                print(f"📊 Loaded {len(self.products)} Coinbase trading pairs")
                
        except Exception as e:
            print(f"❌ Failed to load products: {e}")
    
    async def _get_accounts(self) -> List[CoinbaseAccount]:
        """Get account information"""
        try:
            response = await self._authenticated_request("GET", "/accounts")
            
            accounts = []
            if response:
                for acc_data in response:
                    account = CoinbaseAccount(
                        id=acc_data["id"],
                        currency=acc_data["currency"],
                        balance=Decimal(acc_data["balance"]),
                        available=Decimal(acc_data["available"]), 
                        hold=Decimal(acc_data["hold"]),
                        profile_id=acc_data["profile_id"],
                        trading_enabled=acc_data["trading_enabled"]
                    )
                    accounts.append(account)
                    self.accounts[account.currency] = account
                    
            return accounts
            
        except Exception as e:
            print(f"❌ Failed to get accounts: {e}")
            return []
    
    async def _connect_websocket(self):
        """Establish WebSocket connection for real-time data"""
        try:
            # Subscribe to major crypto pairs
            major_pairs = ["BTC-USD", "ETH-USD", "ADA-USD", "SOL-USD", "MATIC-USD"]
            available_pairs = [pair for pair in major_pairs if pair in self.products]
            
            if not available_pairs:
                print("⚠️ No major crypto pairs available for WebSocket subscription")
                return
            
            # WebSocket authentication  
            timestamp = str(time.time())
            message = timestamp + 'GET' + '/users/self/verify'
            signature = hmac.new(
                self.config.api_secret.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            subscribe_message = {
                "type": "subscribe",
                "product_ids": available_pairs,
                "channels": ["ticker", "matches", "full"],
                "signature": signature,
                "key": self.config.api_key,
                "passphrase": self.config.passphrase,
                "timestamp": timestamp
            }
            
            # Start WebSocket connection in background
            asyncio.create_task(self._websocket_handler(subscribe_message))
            
            print(f"🔗 WebSocket subscribed to {len(available_pairs)} crypto pairs")
            
        except Exception as e:
            print(f"❌ WebSocket connection failed: {e}")
    
    async def _websocket_handler(self, subscribe_message: Dict):
        """Handle WebSocket messages for real-time market data"""
        try:
            async with websockets.connect(self.config.ws_url) as websocket:
                self.ws_connection = websocket
                
                # Send subscription message
                await websocket.send(json.dumps(subscribe_message))
                
                print("🔊 Coinbase WebSocket connected - receiving live market data")
                
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        await self._process_websocket_message(data)
                        
                    except Exception as e:
                        print(f"⚠️ WebSocket message error: {e}")
                        
        except Exception as e:
            print(f"❌ WebSocket handler error: {e}")
            self.ws_connection = None
    
    async def _process_websocket_message(self, data: Dict):
        """Process incoming WebSocket market data"""
        try:
            msg_type = data.get("type")
            
            if msg_type == "ticker":
                # Update ticker data
                ticker = CoinbaseTicker(
                    product_id=data["product_id"],
                    price=Decimal(data["price"]),
                    size=Decimal(data["last_size"]),
                    bid=Decimal(data["best_bid"]), 
                    ask=Decimal(data["best_ask"]),
                    volume=Decimal(data["volume_24h"]),
                    time=datetime.fromisoformat(data["time"].replace('Z', '+00:00')),
                    trade_id=int(data["trade_id"])
                )
                self.tickers[ticker.product_id] = ticker
                self.market_data_queue.append({
                    "type": "ticker",
                    "symbol": ticker.product_id,
                    "price": float(ticker.price),
                    "bid": float(ticker.bid),
                    "ask": float(ticker.ask),
                    "volume": float(ticker.volume),
                    "timestamp": ticker.time
                })
                
            elif msg_type == "match":
                # Process trade/fill
                if data.get("maker_order_id") in self.order_state or data.get("taker_order_id") in self.order_state:
                    fill = Fill(
                        instrument=data["product_id"],
                        qty=float(data["size"]),
                        price=float(data["price"]),
                        ts=datetime.fromisoformat(data["time"].replace('Z', '+00:00')),
                        venue="coinbase_pro"
                    )
                    self.fill_queue.append(fill)
                    
                    # Update order state
                    order_id = data.get("maker_order_id") or data.get("taker_order_id")
                    if order_id:
                        self.order_state[order_id] = "FILLED"
                        
            elif msg_type == "done":
                # Order completed
                order_id = data.get("order_id")
                if order_id and order_id in self.order_state:
                    reason = data.get("reason", "filled")
                    self.order_state[order_id] = "FILLED" if reason == "filled" else "CANCELLED"
                    
        except Exception as e:
            print(f"⚠️ WebSocket message processing error: {e}")
    
    async def _rate_limit(self):
        """Enforce API rate limits"""
        now = time.time()
        
        if now - self.last_request_time >= self.rate_limit_window:
            self.request_count = 0
            self.last_request_time = now
        
        if self.request_count >= self.max_requests_per_second:
            sleep_time = self.rate_limit_window - (now - self.last_request_time)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                self.request_count = 0
                self.last_request_time = time.time()
        
        self.request_count += 1
    
    async def _public_request(self, method: str, path: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make public API request"""
        if not self.session:
            return None
            
        try:
            url = f"{self.config.base_url}{path}"
            
            async with self.session.request(method, url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    print(f"❌ Public API error {response.status}: {await response.text()}")
                    return None
                    
        except Exception as e:
            print(f"❌ Public API request failed: {e}")
            return None
    
    async def _authenticated_request(self, method: str, path: str, data: Optional[Dict] = None) -> Optional[Dict]:
        """Make authenticated API request with proper Coinbase Pro signatures"""
        if not self.session:
            return None
            
        try:
            timestamp = str(time.time())
            body = json.dumps(data) if data else ""
            
            # Create signature
            message = timestamp + method + path + body
            signature = hmac.new(
                self.config.api_secret.encode('utf-8'),
                message.encode('utf-8'), 
                hashlib.sha256
            ).hexdigest()
            
            headers = {
                'CB-ACCESS-KEY': self.config.api_key,
                'CB-ACCESS-SIGN': signature,
                'CB-ACCESS-TIMESTAMP': timestamp,
                'CB-ACCESS-PASSPHRASE': self.config.passphrase,
                'Content-Type': 'application/json'
            }
            
            url = f"{self.config.base_url}{path}"
            
            async with self.session.request(
                method, 
                url, 
                headers=headers, 
                data=body if body else None
            ) as response:
                
                if response.status in [200, 201]:
                    return await response.json()
                else:
                    error_text = await response.text()
                    print(f"❌ Coinbase API error {response.status}: {error_text}")
                    return None
                    
        except Exception as e:
            print(f"❌ Authenticated API request failed: {e}")
            return None


# Factory function for creating Coinbase adapters
def create_coinbase_adapter(
    api_key: str,
    api_secret: str, 
    passphrase: str,
    sandbox: bool = False
) -> CoinbaseProVenueAdapter:
    """Factory function to create Coinbase Pro venue adapter"""
    
    config = CoinbaseConfig(
        name="coinbase_pro",
        host="api.pro.coinbase.com",
        port=443,
        username="",
        password="",
        target_comp_id="",
        sender_comp_id="",
        api_key=api_key,
        api_secret=api_secret,
        passphrase=passphrase,
        sandbox=sandbox
    )
    
    return CoinbaseProVenueAdapter(config)
