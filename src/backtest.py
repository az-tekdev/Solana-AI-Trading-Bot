"""Backtesting module for strategy evaluation."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .config import settings
from .data_fetcher import DataFetcher
from .ai_model import AIModel
from .trader import Trader, Position
from .utils import setup_logger, format_usd_amount

logger = setup_logger(__name__)


class BacktestEngine:
    """Backtesting engine for trading strategies."""
    
    def __init__(self, initial_balance: float = 1000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions: List[Position] = []
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = [initial_balance]
        self.timestamps: List[datetime] = []
        
    async def run(
        self,
        token_address: str,
        start_date: str,
        end_date: str,
        model: AIModel,
        data_fetcher: DataFetcher
    ) -> Dict:
        """Run backtest on historical data."""
        logger.info(f"Starting backtest for {token_address}")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Initial balance: ${self.initial_balance:.2f}")
        
        # Fetch historical data
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        days = (end - start).days
        
        df = await data_fetcher.get_historical_prices(token_address, days=days, interval="1h")
        
        if df.empty:
            logger.error("No historical data available")
            return {}
        
        # Train model on first 70% of data
        train_split = int(0.7 * len(df))
        train_data = df.iloc[:train_split]
        test_data = df.iloc[train_split:]
        
        logger.info(f"Training model on {len(train_data)} samples")
        model.train(train_data)
        
        # Run backtest on test data
        logger.info(f"Running backtest on {len(test_data)} samples")
        
        for idx, row in test_data.iterrows():
            current_price = row['close']
            current_time = row['timestamp']
            
            # Get recent data for prediction
            recent_data = df[df['timestamp'] <= current_time].tail(60)
            
            if len(recent_data) < 60:
                continue
            
            # Make prediction
            prediction = model.predict(recent_data)
            signal = prediction.get("signal", "hold")
            confidence = prediction.get("confidence", 0.0)
            
            # Check existing positions
            self._check_stop_loss_take_profit(token_address, current_price)
            
            # Execute trades based on signal
            if signal == "buy" and confidence > 0.6:
                if len(self.positions) < settings.max_positions:
                    position_size = min(
                        self.balance * settings.max_position_size,
                        self.balance * 0.3
                    )
                    if position_size > 0:
                        self._open_position(
                            token_address,
                            current_price,
                            position_size,
                            current_time
                        )
            
            elif signal == "sell":
                self._close_positions(token_address, current_price, current_time)
            
            # Update equity curve
            self._update_equity(current_price)
            self.timestamps.append(current_time)
        
        # Close all positions at end
        final_price = test_data.iloc[-1]['close']
        self._close_all_positions(final_price, test_data.iloc[-1]['timestamp'])
        self._update_equity(final_price)
        
        # Calculate metrics
        results = self._calculate_metrics()
        
        logger.info("Backtest complete")
        logger.info(f"Final balance: ${self.balance:.2f}")
        logger.info(f"Total return: {results['total_return']:.2f}%")
        logger.info(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")
        
        return results
    
    def _open_position(
        self,
        token_address: str,
        entry_price: float,
        size: float,
        timestamp: datetime
    ):
        """Open a new position."""
        if size > self.balance:
            return
        
        position = Position(
            token_address=token_address,
            entry_price=entry_price,
            amount=size / entry_price,
            side="long",
            timestamp=timestamp
        )
        position.stop_loss = entry_price * (1 - settings.stop_loss_percent / 100)
        position.take_profit = entry_price * (1 + settings.take_profit_percent / 100)
        
        self.positions.append(position)
        self.balance -= size
        
        trade = {
            "type": "buy",
            "timestamp": timestamp,
            "price": entry_price,
            "amount": size,
            "token_address": token_address
        }
        self.trades.append(trade)
        
        logger.debug(f"Opened position: {token_address} @ ${entry_price:.6f}")
    
    def _close_positions(
        self,
        token_address: str,
        exit_price: float,
        timestamp: datetime
    ):
        """Close positions for a token."""
        positions_to_close = [
            p for p in self.positions
            if p.token_address == token_address
        ]
        
        for position in positions_to_close:
            pnl_pct = position.calculate_pnl(exit_price)
            pnl_amount = (pnl_pct / 100) * (position.amount * position.entry_price)
            
            self.balance += (position.amount * exit_price)
            
            trade = {
                "type": "sell",
                "timestamp": timestamp,
                "price": exit_price,
                "amount": position.amount * exit_price,
                "pnl_pct": pnl_pct,
                "pnl_amount": pnl_amount,
                "token_address": token_address
            }
            self.trades.append(trade)
            
            self.positions.remove(position)
            logger.debug(f"Closed position: {token_address} @ ${exit_price:.6f} (P&L: {pnl_pct:.2f}%)")
    
    def _close_all_positions(self, exit_price: float, timestamp: datetime):
        """Close all open positions."""
        for position in self.positions[:]:
            self._close_positions(position.token_address, exit_price, timestamp)
    
    def _check_stop_loss_take_profit(self, token_address: str, current_price: float):
        """Check and execute stop loss/take profit."""
        for position in self.positions[:]:
            if position.token_address == token_address:
                if position.stop_loss and current_price <= position.stop_loss:
                    self._close_positions(
                        token_address,
                        current_price,
                        datetime.now()
                    )
                elif position.take_profit and current_price >= position.take_profit:
                    self._close_positions(
                        token_address,
                        current_price,
                        datetime.now()
                    )
    
    def _update_equity(self, current_price: float):
        """Update equity curve."""
        equity = self.balance
        for position in self.positions:
            equity += position.amount * current_price
        self.equity_curve.append(equity)
    
    def _calculate_metrics(self) -> Dict:
        """Calculate backtest performance metrics."""
        if not self.equity_curve:
            return {}
        
        equity_array = np.array(self.equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        
        total_return = ((equity_array[-1] - equity_array[0]) / equity_array[0]) * 100
        max_equity = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - max_equity) / max_equity * 100
        max_drawdown = np.min(drawdown)
        
        # Sharpe ratio (annualized, assuming hourly data)
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(24 * 365)
        else:
            sharpe_ratio = 0.0
        
        # Win rate
        winning_trades = [t for t in self.trades if t.get("pnl_pct", 0) > 0]
        win_rate = (len(winning_trades) / len(self.trades) * 100) if self.trades else 0
        
        # Average win/loss
        wins = [t.get("pnl_pct", 0) for t in winning_trades]
        losses = [t.get("pnl_pct", 0) for t in self.trades if t.get("pnl_pct", 0) < 0]
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        return {
            "initial_balance": self.initial_balance,
            "final_balance": equity_array[-1],
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "total_trades": len(self.trades),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "equity_curve": equity_array.tolist(),
            "trades": self.trades
        }
    
    def plot_results(self, output_path: str = "data/backtest_results.png"):
        """Plot backtest results."""
        if not self.equity_curve:
            logger.warning("No data to plot")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Equity curve
        axes[0].plot(self.equity_curve, label="Equity", linewidth=2)
        axes[0].axhline(y=self.initial_balance, color='r', linestyle='--', label="Initial Balance")
        axes[0].set_title("Equity Curve")
        axes[0].set_xlabel("Time Step")
        axes[0].set_ylabel("Balance ($)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Drawdown
        equity_array = np.array(self.equity_curve)
        max_equity = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - max_equity) / max_equity * 100
        
        axes[1].fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        axes[1].set_title("Drawdown")
        axes[1].set_xlabel("Time Step")
        axes[1].set_ylabel("Drawdown (%)")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Backtest plot saved to {output_path}")
        plt.close()
