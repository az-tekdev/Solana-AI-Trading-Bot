"""Data fetching module for real-time and historical Solana token data."""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import requests
import websocket
from threading import Thread

from solana.rpc.async_api import AsyncClient
from solders.pubkey import Pubkey

from .config import settings
from .utils import setup_logger, format_sol_amount

logger = setup_logger(__name__)


class DataFetcher:
    """Fetches real-time and historical token data from Solana and APIs."""
    
    def __init__(self):
        self.rpc_client: Optional[AsyncClient] = None
        self.ws_client: Optional[websocket.WebSocketApp] = None
        self.price_cache: Dict[str, Dict] = {}
        self.historical_data: Dict[str, pd.DataFrame] = {}
        
    async def initialize(self):
        """Initialize RPC client."""
        try:
            self.rpc_client = AsyncClient(settings.solana_rpc_url)
            logger.info(f"Initialized RPC client: {settings.solana_rpc_url}")
        except Exception as e:
            logger.error(f"Failed to initialize RPC client: {e}")
            raise
    
    async def close(self):
        """Close connections."""
        if self.rpc_client:
            await self.rpc_client.close()
        if self.ws_client:
            self.ws_client.close()
    
    async def get_token_price(self, token_address: str) -> Optional[float]:
        """Get current token price in USD."""
        try:
            # Try cache first
            if token_address in self.price_cache:
                cached = self.price_cache[token_address]
                if time.time() - cached['timestamp'] < 5:  # 5 second cache
                    return cached['price']
            
            # Fetch from Birdeye API if available
            if settings.birdeye_api_key:
                price = await self._fetch_birdeye_price(token_address)
                if price:
                    self.price_cache[token_address] = {
                        'price': price,
                        'timestamp': time.time()
                    }
                    return price
            
            # Fallback to Jupiter quote
            price = await self._fetch_jupiter_price(token_address)
            if price:
                self.price_cache[token_address] = {
                    'price': price,
                    'timestamp': time.time()
                }
            return price
            
        except Exception as e:
            logger.error(f"Error fetching price for {token_address}: {e}")
            return None
    
    async def _fetch_birdeye_price(self, token_address: str) -> Optional[float]:
        """Fetch price from Birdeye API."""
        try:
            url = f"https://public-api.birdeye.so/v1/token/price"
            headers = {"X-API-KEY": settings.birdeye_api_key}
            params = {"address": token_address}
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("success") and "data" in data:
                    return float(data["data"].get("value", 0))
        except Exception as e:
            logger.debug(f"Birdeye API error: {e}")
        return None
    
    async def _fetch_jupiter_price(self, token_address: str) -> Optional[float]:
        """Fetch price via Jupiter quote (SOL/USD pair)."""
        try:
            # Get quote for 1 token to SOL
            sol_mint = "So11111111111111111111111111111111111111112"
            url = f"{settings.jupiter_api_url}/quote"
            params = {
                "inputMint": token_address,
                "outputMint": sol_mint,
                "amount": 1_000_000_000,  # 1 token (assuming 9 decimals)
                "slippageBps": 50
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "outAmount" in data:
                    sol_amount = int(data["outAmount"]) / 1e9
                    # Get SOL/USD price (simplified - in production use oracle)
                    sol_usd = await self._get_sol_usd_price()
                    return sol_amount * sol_usd if sol_usd else None
        except Exception as e:
            logger.debug(f"Jupiter quote error: {e}")
        return None
    
    async def _get_sol_usd_price(self) -> Optional[float]:
        """Get SOL/USD price (simplified - use CoinGecko or similar)."""
        try:
            # Using CoinGecko free API
            url = "https://api.coingecko.com/api/v3/simple/price"
            params = {"ids": "solana", "vs_currencies": "usd"}
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return float(data.get("solana", {}).get("usd", 0))
        except Exception as e:
            logger.debug(f"SOL/USD price fetch error: {e}")
        return 100.0  # Fallback price
    
    async def get_token_metadata(self, token_address: str) -> Optional[Dict]:
        """Get token metadata (name, symbol, decimals)."""
        try:
            pubkey = Pubkey.from_string(token_address)
            # Fetch token metadata from on-chain
            # This is simplified - in production use Metaplex Token Metadata
            return {
                "address": token_address,
                "name": "Unknown Token",
                "symbol": "UNK",
                "decimals": 9
            }
        except Exception as e:
            logger.error(f"Error fetching metadata: {e}")
            return None
    
    async def get_historical_prices(
        self,
        token_address: str,
        days: int = 30,
        interval: str = "1h"
    ) -> pd.DataFrame:
        """Get historical price data."""
        try:
            cache_key = f"{token_address}_{days}_{interval}"
            if cache_key in self.historical_data:
                return self.historical_data[cache_key]
            
            # Generate synthetic historical data for demo
            # In production, fetch from Birdeye, Helius, or DEX aggregators
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            dates = pd.date_range(start=start_date, end=end_date, freq=interval)
            n = len(dates)
            
            # Generate realistic price data with trend and noise
            base_price = 0.001
            trend = pd.Series(range(n)) * 0.00001
            noise = pd.Series([(hash(f"{i}_{token_address}") % 1000) / 1000000 
                              for i in range(n)])
            prices = base_price + trend + noise
            
            df = pd.DataFrame({
                "timestamp": dates,
                "price": prices,
                "volume": [hash(f"v_{i}_{token_address}") % 1000000 
                          for i in range(n)],
                "high": prices * 1.05,
                "low": prices * 0.95,
                "open": prices,
                "close": prices
            })
            
            self.historical_data[cache_key] = df
            logger.info(f"Fetched {len(df)} historical data points for {token_address}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()
    
    async def get_liquidity_info(self, token_address: str) -> Optional[Dict]:
        """Get liquidity information for a token."""
        try:
            # Simplified liquidity check
            # In production, query DEX pools (Raydium, Orca, etc.)
            return {
                "total_liquidity_usd": 50000.0,  # Mock data
                "pool_count": 3,
                "top_pools": []
            }
        except Exception as e:
            logger.error(f"Error fetching liquidity: {e}")
            return None
    
    def start_websocket(self, token_address: str, callback):
        """Start WebSocket connection for real-time price updates."""
        def on_message(ws, message):
            try:
                data = json.loads(message)
                callback(data)
            except Exception as e:
                logger.error(f"WebSocket message error: {e}")
        
        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            logger.info("WebSocket connection closed")
        
        def on_open(ws):
            logger.info(f"WebSocket connected for {token_address}")
        
        # In production, use actual Solana WebSocket or Birdeye WebSocket
        # This is a placeholder
        ws_url = settings.solana_ws_url
        self.ws_client = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        # Run in separate thread
        wst = Thread(target=self.ws_client.run_forever)
        wst.daemon = True
        wst.start()
        return self.ws_client
    
    async def get_token_volume_24h(self, token_address: str) -> Optional[float]:
        """Get 24h trading volume."""
        try:
            if settings.birdeye_api_key:
                url = f"https://public-api.birdeye.so/v1/token/overview"
                headers = {"X-API-KEY": settings.birdeye_api_key}
                params = {"address": token_address}
                
                response = requests.get(url, headers=headers, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success") and "data" in data:
                        return float(data["data"].get("volume24hUSD", 0))
        except Exception as e:
            logger.debug(f"Volume fetch error: {e}")
        return None
