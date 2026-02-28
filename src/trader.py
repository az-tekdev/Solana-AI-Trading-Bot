"""Trading execution module for Solana DEX operations."""

import asyncio
import base64
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import requests
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from solana.transaction import Transaction
import base58

from .config import settings
from .utils import setup_logger, format_sol_amount, truncate_address

logger = setup_logger(__name__)


class Position:
    """Represents an open trading position."""
    
    def __init__(
        self,
        token_address: str,
        entry_price: float,
        amount: float,
        side: str,
        timestamp: datetime
    ):
        self.token_address = token_address
        self.entry_price = entry_price
        self.amount = amount
        self.side = side  # 'long' or 'short'
        self.timestamp = timestamp
        self.stop_loss = None
        self.take_profit = None
    
    def calculate_pnl(self, current_price: float) -> float:
        """Calculate profit/loss percentage."""
        if self.side == 'long':
            return ((current_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - current_price) / self.entry_price) * 100
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "token_address": self.token_address,
            "entry_price": self.entry_price,
            "amount": self.amount,
            "side": self.side,
            "timestamp": self.timestamp.isoformat(),
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit
        }


class Trader:
    """Handles trade execution on Solana DEX via Jupiter."""
    
    def __init__(self):
        self.keypair: Optional[Keypair] = None
        self.rpc_client: Optional[AsyncClient] = None
        self.positions: List[Position] = []
        self.trade_history: List[Dict] = []
        self.daily_pnl: float = 0.0
        self.last_trade_time: float = 0.0
        
    async def initialize(self):
        """Initialize wallet and RPC connection."""
        try:
            # Initialize RPC client
            self.rpc_client = AsyncClient(settings.solana_rpc_url)
            
            # Load wallet if private key provided
            if settings.wallet_private_key and not settings.dry_run:
                try:
                    private_key_bytes = base58.b58decode(settings.wallet_private_key)
                    self.keypair = Keypair.from_bytes(private_key_bytes)
                    logger.info(f"Wallet loaded: {truncate_address(str(self.keypair.pubkey()))}")
                except Exception as e:
                    logger.error(f"Failed to load wallet: {e}")
                    raise
            else:
                logger.warning("Running in DRY RUN mode - no actual trades will be executed")
            
            logger.info("Trader initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize trader: {e}")
            raise
    
    async def close(self):
        """Close connections."""
        if self.rpc_client:
            await self.rpc_client.close()
    
    async def get_balance(self, token_address: Optional[str] = None) -> float:
        """Get SOL or token balance."""
        if not self.keypair:
            return 0.0
        
        try:
            if token_address is None:
                # Get SOL balance
                balance = await self.rpc_client.get_balance(
                    self.keypair.pubkey(),
                    commitment=Confirmed
                )
                return format_sol_amount(balance.value)
            else:
                # Get token balance (simplified)
                # In production, query SPL token accounts
                return 0.0
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return 0.0
    
    async def get_jupiter_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int = 50
    ) -> Optional[Dict]:
        """Get quote from Jupiter aggregator."""
        try:
            url = f"{settings.jupiter_api_url}/quote"
            params = {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": amount,
                "slippageBps": slippage_bps
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Jupiter quote error: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error getting Jupiter quote: {e}")
            return None
    
    async def execute_swap(
        self,
        input_mint: str,
        output_mint: str,
        amount: float,
        slippage_bps: Optional[int] = None
    ) -> Optional[Dict]:
        """Execute a token swap via Jupiter."""
        if settings.dry_run:
            logger.info(f"[DRY RUN] Would swap {amount} {input_mint} -> {output_mint}")
            return {
                "success": True,
                "dry_run": True,
                "input_mint": input_mint,
                "output_mint": output_mint,
                "amount": amount
            }
        
        if not self.keypair:
            logger.error("No wallet loaded")
            return None
        
        try:
            slippage_bps = slippage_bps or int(settings.slippage_tolerance * 100)
            
            # Convert amount to lamports/token units (assuming 9 decimals)
            amount_lamports = int(amount * 1_000_000_000)
            
            # Get quote
            quote = await self.get_jupiter_quote(
                input_mint,
                output_mint,
                amount_lamports,
                slippage_bps
            )
            
            if not quote:
                logger.error("Failed to get quote")
                return None
            
            # Get swap transaction
            swap_url = f"{settings.jupiter_api_url}/swap"
            swap_data = {
                "quoteResponse": quote,
                "userPublicKey": str(self.keypair.pubkey()),
                "wrapAndUnwrapSol": True,
                "dynamicComputeUnitLimit": True,
                "prioritizationFeeLamports": "auto"
            }
            
            response = requests.post(swap_url, json=swap_data, timeout=30)
            if response.status_code != 200:
                logger.error(f"Swap API error: {response.status_code}")
                return None
            
            swap_response = response.json()
            swap_transaction = swap_response.get("swapTransaction")
            
            if not swap_transaction:
                logger.error("No swap transaction in response")
                return None
            
            # Deserialize and sign transaction
            transaction_bytes = base64.b64decode(swap_transaction)
            transaction = Transaction.deserialize(transaction_bytes)
            transaction.sign(self.keypair)
            
            # Send transaction
            result = await self.rpc_client.send_transaction(
                transaction,
                self.keypair,
                opts={"skip_preflight": False, "max_retries": 3}
            )
            
            logger.info(f"Swap transaction sent: {result.value}")
            
            # Wait for confirmation
            await asyncio.sleep(2)
            confirmation = await self.rpc_client.confirm_transaction(
                result.value,
                commitment=Confirmed
            )
            
            if confirmation.value[0].err:
                logger.error(f"Transaction failed: {confirmation.value[0].err}")
                return None
            
            logger.info(f"Swap successful: {result.value}")
            
            return {
                "success": True,
                "transaction_signature": str(result.value),
                "input_mint": input_mint,
                "output_mint": output_mint,
                "amount": amount,
                "quote": quote
            }
            
        except Exception as e:
            logger.error(f"Error executing swap: {e}")
            return None
    
    async def buy_token(
        self,
        token_address: str,
        sol_amount: float,
        current_price: float
    ) -> Optional[Dict]:
        """Buy token with SOL."""
        # Check cooldown
        if time.time() - self.last_trade_time < settings.cooldown_seconds:
            logger.warning("Trade cooldown active")
            return None
        
        # Check position limit
        if len(self.positions) >= settings.max_positions:
            logger.warning(f"Max positions reached ({settings.max_positions})")
            return None
        
        # Check daily loss limit
        if self.daily_pnl < -settings.max_daily_loss:
            logger.warning("Daily loss limit reached")
            return None
        
        sol_mint = "So11111111111111111111111111111111111111112"
        
        result = await self.execute_swap(
            sol_mint,
            token_address,
            sol_amount
        )
        
        if result and result.get("success"):
            # Create position
            position = Position(
                token_address=token_address,
                entry_price=current_price,
                amount=sol_amount / current_price,  # Approximate
                side="long",
                timestamp=datetime.now()
            )
            position.stop_loss = current_price * (1 - settings.stop_loss_percent / 100)
            position.take_profit = current_price * (1 + settings.take_profit_percent / 100)
            
            self.positions.append(position)
            self.last_trade_time = time.time()
            
            # Record trade
            trade_record = {
                "type": "buy",
                "token_address": token_address,
                "amount": sol_amount,
                "price": current_price,
                "timestamp": datetime.now().isoformat(),
                "transaction": result.get("transaction_signature", "dry_run")
            }
            self.trade_history.append(trade_record)
            
            logger.info(f"Bought {token_address} at ${current_price:.6f}")
            return result
        
        return None
    
    async def sell_token(
        self,
        token_address: str,
        token_amount: Optional[float] = None,
        current_price: float = None
    ) -> Optional[Dict]:
        """Sell token for SOL."""
        # Find position
        position = None
        for pos in self.positions:
            if pos.token_address == token_address and pos.side == "long":
                position = pos
                break
        
        if not position:
            logger.warning(f"No position found for {token_address}")
            return None
        
        # Use position amount if not specified
        if token_amount is None:
            token_amount = position.amount
        
        sol_mint = "So11111111111111111111111111111111111111112"
        
        result = await self.execute_swap(
            token_address,
            sol_mint,
            token_amount
        )
        
        if result and result.get("success"):
            # Calculate P&L
            pnl_pct = position.calculate_pnl(current_price or position.entry_price)
            pnl_amount = (pnl_pct / 100) * (position.amount * position.entry_price)
            
            self.daily_pnl += pnl_pct / 100
            
            # Remove position
            self.positions.remove(position)
            self.last_trade_time = time.time()
            
            # Record trade
            trade_record = {
                "type": "sell",
                "token_address": token_address,
                "amount": token_amount,
                "price": current_price or position.entry_price,
                "pnl_pct": pnl_pct,
                "pnl_amount": pnl_amount,
                "timestamp": datetime.now().isoformat(),
                "transaction": result.get("transaction_signature", "dry_run")
            }
            self.trade_history.append(trade_record)
            
            logger.info(f"Sold {token_address} at ${current_price:.6f} (P&L: {pnl_pct:.2f}%)")
            return result
        
        return None
    
    def check_stop_loss_take_profit(self, token_address: str, current_price: float) -> bool:
        """Check if stop loss or take profit should trigger."""
        for position in self.positions:
            if position.token_address == token_address:
                pnl_pct = position.calculate_pnl(current_price)
                
                if position.stop_loss and current_price <= position.stop_loss:
                    logger.warning(f"Stop loss triggered for {token_address}")
                    return True
                
                if position.take_profit and current_price >= position.take_profit:
                    logger.info(f"Take profit triggered for {token_address}")
                    return True
        
        return False
    
    def get_positions(self) -> List[Dict]:
        """Get all open positions."""
        return [pos.to_dict() for pos in self.positions]
    
    def get_trade_history(self) -> List[Dict]:
        """Get trade history."""
        return self.trade_history
