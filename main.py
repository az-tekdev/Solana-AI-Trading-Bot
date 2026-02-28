"""Main entry point for the Solana AI Trading Bot."""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import settings, ensure_directories
from src.data_fetcher import DataFetcher
from src.ai_model import AIModel
from src.trader import Trader
from src.backtest import BacktestEngine
from src.utils import setup_logger

logger = setup_logger(__name__)


async def run_backtest(args):
    """Run backtesting mode."""
    logger.info("=" * 60)
    logger.info("BACKTEST MODE")
    logger.info("=" * 60)
    
    ensure_directories()
    
    # Initialize components
    data_fetcher = DataFetcher()
    await data_fetcher.initialize()
    
    model = AIModel(model_type=args.model_type or settings.ai_model_type)
    
    # Load or train model
    if Path(settings.model_path).exists():
        logger.info(f"Loading model from {settings.model_path}")
        model.load(settings.model_path)
    else:
        logger.info("Model not found, will train during backtest")
    
    # Run backtest
    engine = BacktestEngine(initial_balance=args.initial_balance or settings.initial_balance)
    
    results = await engine.run(
        token_address=args.token,
        start_date=args.start_date or settings.backtest_start_date,
        end_date=args.end_date or settings.backtest_end_date,
        model=model,
        data_fetcher=data_fetcher
    )
    
    # Print results
    if results:
        logger.info("\n" + "=" * 60)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Initial Balance: ${results['initial_balance']:.2f}")
        logger.info(f"Final Balance: ${results['final_balance']:.2f}")
        logger.info(f"Total Return: {results['total_return']:.2f}%")
        logger.info(f"Max Drawdown: {results['max_drawdown']:.2f}%")
        logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"Total Trades: {results['total_trades']}")
        logger.info(f"Win Rate: {results['win_rate']:.2f}%")
        logger.info(f"Avg Win: {results['avg_win']:.2f}%")
        logger.info(f"Avg Loss: {results['avg_loss']:.2f}%")
        logger.info("=" * 60)
        
        # Plot results
        if args.plot:
            engine.plot_results()
    
    await data_fetcher.close()


async def run_simulation(args):
    """Run simulation mode (paper trading)."""
    logger.info("=" * 60)
    logger.info("SIMULATION MODE")
    logger.info("=" * 60)
    
    ensure_directories()
    
    # Initialize components
    data_fetcher = DataFetcher()
    await data_fetcher.initialize()
    
    model = AIModel(model_type=args.model_type or settings.ai_model_type)
    
    # Load model
    if Path(settings.model_path).exists():
        model.load(settings.model_path)
    else:
        logger.error("Model not found. Please train a model first or run backtest.")
        return
    
    trader = Trader()
    await trader.initialize()
    
    # Main trading loop
    token_address = args.token
    logger.info(f"Starting simulation for token: {token_address}")
    
    try:
        while True:
            # Get current price
            current_price = await data_fetcher.get_token_price(token_address)
            
            if not current_price:
                logger.warning("Failed to fetch price, retrying...")
                await asyncio.sleep(10)
                continue
            
            # Get historical data for prediction
            df = await data_fetcher.get_historical_prices(token_address, days=7, interval="1h")
            
            if len(df) < 60:
                logger.warning("Insufficient historical data")
                await asyncio.sleep(60)
                continue
            
            # Make prediction
            prediction = model.predict(df)
            signal = prediction.get("signal", "hold")
            confidence = prediction.get("confidence", 0.0)
            
            logger.info(f"Price: ${current_price:.6f} | Signal: {signal} | Confidence: {confidence:.2f}")
            
            # Check stop loss/take profit
            trader.check_stop_loss_take_profit(token_address, current_price)
            
            # Execute trades
            if signal == "buy" and confidence > 0.6:
                balance = await trader.get_balance()
                position_size = min(balance * settings.max_position_size, balance * 0.3)
                
                if position_size > 0.01:  # Minimum 0.01 SOL
                    await trader.buy_token(token_address, position_size, current_price)
            
            elif signal == "sell":
                await trader.sell_token(token_address, current_price=current_price)
            
            # Log positions
            positions = trader.get_positions()
            if positions:
                logger.info(f"Open positions: {len(positions)}")
            
            # Wait before next iteration
            await asyncio.sleep(args.interval or 60)
            
    except KeyboardInterrupt:
        logger.info("Simulation stopped by user")
    finally:
        await trader.close()
        await data_fetcher.close()


async def run_live(args):
    """Run live trading mode."""
    if settings.dry_run:
        logger.warning("DRY RUN mode is enabled. No real trades will be executed.")
    else:
        logger.warning("LIVE TRADING MODE - Real funds will be used!")
        response = input("Are you sure you want to continue? (yes/no): ")
        if response.lower() != "yes":
            logger.info("Live trading cancelled")
            return
    
    logger.info("=" * 60)
    logger.info("LIVE TRADING MODE")
    logger.info("=" * 60)
    
    ensure_directories()
    
    # Initialize components
    data_fetcher = DataFetcher()
    await data_fetcher.initialize()
    
    model = AIModel(model_type=args.model_type or settings.ai_model_type)
    
    # Load model
    if Path(settings.model_path).exists():
        model.load(settings.model_path)
    else:
        logger.error("Model not found. Please train a model first.")
        return
    
    trader = Trader()
    await trader.initialize()
    
    # Main trading loop (same as simulation but with real wallet)
    token_address = args.token
    logger.info(f"Starting live trading for token: {token_address}")
    
    try:
        while True:
            current_price = await data_fetcher.get_token_price(token_address)
            
            if not current_price:
                await asyncio.sleep(10)
                continue
            
            df = await data_fetcher.get_historical_prices(token_address, days=7, interval="1h")
            
            if len(df) < 60:
                await asyncio.sleep(60)
                continue
            
            prediction = model.predict(df)
            signal = prediction.get("signal", "hold")
            confidence = prediction.get("confidence", 0.0)
            
            logger.info(f"Price: ${current_price:.6f} | Signal: {signal} | Confidence: {confidence:.2f}")
            
            trader.check_stop_loss_take_profit(token_address, current_price)
            
            if signal == "buy" and confidence > 0.6:
                balance = await trader.get_balance()
                position_size = min(balance * settings.max_position_size, balance * 0.3)
                
                if position_size > 0.01:
                    await trader.buy_token(token_address, position_size, current_price)
            
            elif signal == "sell":
                await trader.sell_token(token_address, current_price=current_price)
            
            await asyncio.sleep(args.interval or 60)
            
    except KeyboardInterrupt:
        logger.info("Live trading stopped by user")
    finally:
        await trader.close()
        await data_fetcher.close()


async def train_model(args):
    """Train AI model on historical data."""
    logger.info("=" * 60)
    logger.info("TRAINING MODE")
    logger.info("=" * 60)
    
    ensure_directories()
    
    data_fetcher = DataFetcher()
    await data_fetcher.initialize()
    
    model = AIModel(model_type=args.model_type or settings.ai_model_type)
    
    # Fetch historical data
    logger.info(f"Fetching historical data for {args.token}")
    df = await data_fetcher.get_historical_prices(
        args.token,
        days=args.days or 90,
        interval="1h"
    )
    
    if df.empty:
        logger.error("No historical data available")
        return
    
    # Train model
    results = model.train(df)
    
    # Save model
    model.save(settings.model_path)
    
    logger.info("Model training complete!")
    logger.info(f"Model saved to {settings.model_path}")
    
    await data_fetcher.close()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Solana AI Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="mode", help="Trading mode")
    
    # Backtest mode
    backtest_parser = subparsers.add_parser("backtest", help="Run backtesting")
    backtest_parser.add_argument("--token", required=True, help="Token address")
    backtest_parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    backtest_parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    backtest_parser.add_argument("--initial-balance", type=float, help="Initial balance")
    backtest_parser.add_argument("--model-type", choices=["lstm", "random_forest"], help="Model type")
    backtest_parser.add_argument("--plot", action="store_true", help="Generate plot")
    
    # Simulation mode
    sim_parser = subparsers.add_parser("simulate", help="Run simulation (paper trading)")
    sim_parser.add_argument("--token", required=True, help="Token address")
    sim_parser.add_argument("--interval", type=int, help="Update interval in seconds")
    sim_parser.add_argument("--model-type", choices=["lstm", "random_forest"], help="Model type")
    
    # Live trading mode
    live_parser = subparsers.add_parser("live", help="Run live trading")
    live_parser.add_argument("--token", required=True, help="Token address")
    live_parser.add_argument("--interval", type=int, help="Update interval in seconds")
    live_parser.add_argument("--model-type", choices=["lstm", "random_forest"], help="Model type")
    
    # Train mode
    train_parser = subparsers.add_parser("train", help="Train AI model")
    train_parser.add_argument("--token", required=True, help="Token address")
    train_parser.add_argument("--days", type=int, help="Days of historical data")
    train_parser.add_argument("--model-type", choices=["lstm", "random_forest"], help="Model type")
    
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        return
    
    # Run appropriate mode
    if args.mode == "backtest":
        asyncio.run(run_backtest(args))
    elif args.mode == "simulate":
        asyncio.run(run_simulation(args))
    elif args.mode == "live":
        asyncio.run(run_live(args))
    elif args.mode == "train":
        asyncio.run(train_model(args))


if __name__ == "__main__":
    main()
