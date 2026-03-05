"""Quickstart example for Solana AI Trading Bot."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings, ensure_directories
from src.data_fetcher import DataFetcher
from src.ai_model import AIModel
from src.utils import setup_logger

logger = setup_logger(__name__)


async def quickstart_example():
    """Quickstart example demonstrating basic bot functionality."""
    logger.info("=" * 60)
    logger.info("SOLANA AI TRADING BOT - QUICKSTART EXAMPLE")
    logger.info("=" * 60)
    
    ensure_directories()
    
    # Initialize data fetcher
    logger.info("Initializing data fetcher...")
    data_fetcher = DataFetcher()
    await data_fetcher.initialize()
    
    # Example: Fetch price for SOL
    sol_address = "So11111111111111111111111111111111111111112"
    logger.info(f"Fetching price for SOL ({sol_address})...")
    
    price = await data_fetcher.get_token_price(sol_address)
    if price:
        logger.info(f"Current SOL price: ${price:.2f}")
    else:
        logger.warning("Could not fetch price (using mock data)")
    
    # Example: Get historical data
    logger.info("Fetching historical data...")
    df = await data_fetcher.get_historical_prices(sol_address, days=7, interval="1h")
    logger.info(f"Retrieved {len(df)} data points")
    
    if not df.empty:
        logger.info(f"Price range: ${df['price'].min():.6f} - ${df['price'].max():.6f}")
        logger.info(f"Average price: ${df['price'].mean():.6f}")
    
    # Example: Train a simple model
    logger.info("\nTraining AI model...")
    model = AIModel(model_type="lstm")
    
    if len(df) >= 200:
        logger.info("Training on historical data...")
        results = model.train(df)
        logger.info(f"Training complete! Test loss: {results.get('test_loss', 'N/A')}")
        
        # Example: Make a prediction
        logger.info("\nMaking prediction...")
        prediction = model.predict(df)
        logger.info(f"Signal: {prediction['signal']}")
        logger.info(f"Confidence: {prediction['confidence']:.2%}")
        if prediction.get('predicted_price'):
            logger.info(f"Predicted price: ${prediction['predicted_price']:.6f}")
    else:
        logger.warning("Insufficient data for training (need at least 200 samples)")
    
    # Cleanup
    await data_fetcher.close()
    
    logger.info("\n" + "=" * 60)
    logger.info("Quickstart example complete!")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("1. Configure your .env file with API keys")
    logger.info("2. Train a model: python main.py train --token <TOKEN_ADDRESS>")
    logger.info("3. Run backtest: python main.py backtest --token <TOKEN_ADDRESS>")
    logger.info("4. Start simulation: python main.py simulate --token <TOKEN_ADDRESS>")


if __name__ == "__main__":
    asyncio.run(quickstart_example())
