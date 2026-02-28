"""Configuration management for the trading bot."""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Solana Configuration
    solana_rpc_url: str = Field(
        default="https://api.mainnet-beta.solana.com",
        env="SOLANA_RPC_URL"
    )
    solana_ws_url: str = Field(
        default="wss://api.mainnet-beta.solana.com",
        env="SOLANA_WS_URL"
    )
    wallet_private_key: Optional[str] = Field(
        default=None,
        env="WALLET_PRIVATE_KEY"
    )
    
    # Jupiter API
    jupiter_api_url: str = Field(
        default="https://quote-api.jup.ag/v6",
        env="JUPITER_API_URL"
    )
    
    # AI/ML Configuration
    ai_model_type: str = Field(default="lstm", env="AI_MODEL_TYPE")
    model_path: str = Field(default="models/lstm_model.pth", env="MODEL_PATH")
    use_llm: bool = Field(default=False, env="USE_LLM")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4", env="OPENAI_MODEL")
    
    # Trading Configuration
    dry_run: bool = Field(default=True, env="DRY_RUN")
    max_position_size: float = Field(default=0.1, env="MAX_POSITION_SIZE")
    slippage_tolerance: float = Field(default=1.0, env="SLIPPAGE_TOLERANCE")
    stop_loss_percent: float = Field(default=5.0, env="STOP_LOSS_PERCENT")
    take_profit_percent: float = Field(default=10.0, env="TAKE_PROFIT_PERCENT")
    min_liquidity_usd: float = Field(default=10000.0, env="MIN_LIQUIDITY_USD")
    
    # Risk Management
    max_daily_loss: float = Field(default=0.05, env="MAX_DAILY_LOSS")
    max_positions: int = Field(default=3, env="MAX_POSITIONS")
    cooldown_seconds: int = Field(default=60, env="COOLDOWN_SECONDS")
    
    # Data Sources
    birdeye_api_key: Optional[str] = Field(default=None, env="BIRDEYE_API_KEY")
    helius_api_key: Optional[str] = Field(default=None, env="HELIUS_API_KEY")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/trading_bot.log", env="LOG_FILE")
    
    # Backtesting
    backtest_start_date: str = Field(
        default="2024-01-01",
        env="BACKTEST_START_DATE"
    )
    backtest_end_date: str = Field(
        default="2024-12-31",
        env="BACKTEST_END_DATE"
    )
    initial_balance: float = Field(default=1000.0, env="INITIAL_BALANCE")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def ensure_directories():
    """Ensure required directories exist."""
    directories = [
        "data",
        "models",
        "logs",
        Path(settings.model_path).parent
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
