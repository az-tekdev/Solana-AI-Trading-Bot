"""Tests for trader module."""

import pytest
import asyncio
from src.trader import Trader, Position
from datetime import datetime


@pytest.mark.asyncio
async def test_trader_initialization():
    """Test trader initialization."""
    trader = Trader()
    await trader.initialize()
    assert trader.rpc_client is not None
    await trader.close()


def test_position_creation():
    """Test position creation."""
    position = Position(
        token_address="test_token",
        entry_price=0.001,
        amount=1000.0,
        side="long",
        timestamp=datetime.now()
    )
    
    assert position.token_address == "test_token"
    assert position.entry_price == 0.001
    assert position.amount == 1000.0
    assert position.side == "long"


def test_position_pnl_calculation():
    """Test P&L calculation."""
    position = Position(
        token_address="test_token",
        entry_price=0.001,
        amount=1000.0,
        side="long",
        timestamp=datetime.now()
    )
    
    # Price increased by 10%
    pnl = position.calculate_pnl(0.0011)
    assert pnl == pytest.approx(10.0, rel=0.1)
    
    # Price decreased by 5%
    pnl = position.calculate_pnl(0.00095)
    assert pnl == pytest.approx(-5.0, rel=0.1)


@pytest.mark.asyncio
async def test_get_balance():
    """Test getting balance."""
    trader = Trader()
    await trader.initialize()
    
    # In dry run mode, balance might be 0
    balance = await trader.get_balance()
    assert balance >= 0
    
    await trader.close()
