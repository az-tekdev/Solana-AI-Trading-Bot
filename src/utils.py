"""Utility functions for logging and helpers."""

import logging
import sys
from pathlib import Path
from typing import Optional
import colorlog
from datetime import datetime

from .config import settings


def setup_logger(name: str = "trading_bot") -> logging.Logger:
    """Set up colored console and file logging."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.log_level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler with colors
    console_handler = colorlog.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_format = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        }
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    log_file_path = Path(settings.log_file)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


def format_sol_amount(lamports: int) -> float:
    """Convert lamports to SOL."""
    return lamports / 1_000_000_000


def format_usd_amount(amount: float) -> str:
    """Format USD amount for display."""
    return f"${amount:,.2f}"


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change."""
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100


def truncate_address(address: str, length: int = 8) -> str:
    """Truncate Solana address for display."""
    if len(address) <= length * 2:
        return address
    return f"{address[:length]}...{address[-length:]}"


def validate_solana_address(address: str) -> bool:
    """Basic validation for Solana address format."""
    return len(address) >= 32 and len(address) <= 44


def get_timestamp() -> str:
    """Get current timestamp as string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
