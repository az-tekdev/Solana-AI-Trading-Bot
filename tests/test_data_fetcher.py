"""Tests for data fetcher module."""

import pytest
import asyncio
from src.data_fetcher import DataFetcher


@pytest.mark.asyncio
async def test_data_fetcher_initialization():
    """Test data fetcher initialization."""
    fetcher = DataFetcher()
    await fetcher.initialize()
    assert fetcher.rpc_client is not None
    await fetcher.close()


@pytest.mark.asyncio
async def test_get_historical_prices():
    """Test fetching historical prices."""
    fetcher = DataFetcher()
    await fetcher.initialize()
    
    # Use a known token address (SOL)
    token_address = "So11111111111111111111111111111111111111112"
    df = await fetcher.get_historical_prices(token_address, days=7, interval="1h")
    
    assert not df.empty
    assert "price" in df.columns
    assert "timestamp" in df.columns
    assert len(df) > 0
    
    await fetcher.close()


@pytest.mark.asyncio
async def test_get_token_price():
    """Test getting current token price."""
    fetcher = DataFetcher()
    await fetcher.initialize()
    
    # Use SOL token
    token_address = "So11111111111111111111111111111111111111112"
    price = await fetcher.get_token_price(token_address)
    
    # Price should be a positive number or None
    assert price is None or price > 0
    
    await fetcher.close()
