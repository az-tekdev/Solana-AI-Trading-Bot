# Solana AI Trading Bot

A production-ready, AI-powered trading bot for the Solana blockchain that autonomously makes buy/sell decisions using machine learning models, executes trades via Jupiter DEX aggregator, and handles real-time on-chain data.

## 🚀 Features

- **AI-Powered Decision Making**: Uses LSTM neural networks or Random Forest classifiers to predict price movements
- **Real-Time Data**: Fetches live token prices and on-chain metrics via Solana RPC and APIs
- **DEX Integration**: Executes trades through Jupiter aggregator for optimal routing
- **Risk Management**: Built-in stop-loss, take-profit, position sizing, and daily loss limits
- **Multiple Modes**: Backtesting, simulation (paper trading), and live trading
- **Comprehensive Logging**: Detailed trade history, P&L tracking, and error monitoring
- **Docker Support**: Containerized deployment for easy setup

## 📋 Requirements

- Python 3.10+
- Solana wallet with private key (for live trading)
- API keys (optional): Birdeye, Helius, OpenAI (for LLM mode)

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Solana-AI-Trading-Bot.git
cd Solana-AI-Trading-Bot
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the example environment file and fill in your values:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# Solana Configuration
SOLANA_RPC_URL=https://api.mainnet-beta.solana.com
WALLET_PRIVATE_KEY=your_wallet_private_key_base58_here

# Trading Configuration
DRY_RUN=true  # Set to false for live trading
MAX_POSITION_SIZE=0.1
SLIPPAGE_TOLERANCE=1.0
STOP_LOSS_PERCENT=5.0
TAKE_PROFIT_PERCENT=10.0

# AI Configuration
AI_MODEL_TYPE=lstm
MODEL_PATH=models/lstm_model.pth
```

## 📖 Usage

### Training a Model

Train an AI model on historical data:

```bash
python main.py train --token So11111111111111111111111111111111111111112 --days 90 --model-type lstm
```

### Backtesting

Test your strategy on historical data:

```bash
python main.py backtest \
    --token So11111111111111111111111111111111111111112 \
    --start-date 2024-01-01 \
    --end-date 2024-12-31 \
    --initial-balance 1000 \
    --plot
```

### Simulation Mode (Paper Trading)

Run the bot in simulation mode without real funds:

```bash
python main.py simulate \
    --token So11111111111111111111111111111111111111112 \
    --interval 60 \
    --model-type lstm
```

### Live Trading

⚠️ **WARNING**: Live trading uses real funds. Use at your own risk!

```bash
python main.py live \
    --token So11111111111111111111111111111111111111112 \
    --interval 60 \
    --model-type lstm
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Main Entry Point                      │
│                      (main.py)                           │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
┌───────▼──────┐ ┌───▼────┐ ┌────▼──────┐
│ Data Fetcher │ │  AI    │ │  Trader   │
│              │ │ Model  │ │           │
│ - RPC Client │ │ - LSTM │ │ - Jupiter │
│ - WebSocket  │ │ - RF   │ │ - Wallet  │
│ - APIs       │ │ - LLM  │ │ - Risk    │
└───────┬──────┘ └───┬────┘ └────┬──────┘
        │            │            │
        └────────────┼────────────┘
                     │
            ┌────────▼────────┐
            │  Backtest Engine│
            │  - Simulation   │
            │  - Metrics      │
            │  - Visualization│
            └─────────────────┘
```

### Core Modules

- **`config.py`**: Centralized configuration management using Pydantic
- **`data_fetcher.py`**: Fetches real-time and historical token data
- **`ai_model.py`**: Implements LSTM and Random Forest models for predictions
- **`trader.py`**: Handles trade execution via Jupiter DEX
- **`backtest.py`**: Backtesting engine with performance metrics
- **`utils.py`**: Logging and utility functions

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SOLANA_RPC_URL` | Solana RPC endpoint | `https://api.mainnet-beta.solana.com` |
| `WALLET_PRIVATE_KEY` | Base58 encoded private key | None |
| `DRY_RUN` | Enable dry-run mode | `true` |
| `MAX_POSITION_SIZE` | Max position size (fraction of balance) | `0.1` |
| `SLIPPAGE_TOLERANCE` | Slippage tolerance (%) | `1.0` |
| `STOP_LOSS_PERCENT` | Stop loss percentage | `5.0` |
| `TAKE_PROFIT_PERCENT` | Take profit percentage | `10.0` |
| `AI_MODEL_TYPE` | Model type (`lstm` or `random_forest`) | `lstm` |
| `MAX_DAILY_LOSS` | Max daily loss limit | `0.05` |
| `MAX_POSITIONS` | Maximum concurrent positions | `3` |

### Strategy Parameters

The bot supports configurable strategies:

- **Momentum-based**: Uses price trends and moving averages
- **ML-based**: Uses trained LSTM or Random Forest models
- **LLM-based**: Uses OpenAI GPT models for sentiment analysis (optional)

## 🐳 Docker Deployment

### Build and Run

```bash
docker build -t solana-ai-trading-bot .
docker run -v $(pwd)/.env:/app/.env solana-ai-trading-bot python main.py simulate --token <TOKEN_ADDRESS>
```

### Docker Compose

```bash
docker-compose up -d
```

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

## 📊 Backtesting Results

The backtest engine provides comprehensive metrics:

- **Total Return**: Overall profit/loss percentage
- **Sharpe Ratio**: Risk-adjusted return metric
- **Max Drawdown**: Maximum peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Average Win/Loss**: Average profit/loss per trade

Example output:

```
BACKTEST RESULTS
============================================================
Initial Balance: $1000.00
Final Balance: $1250.50
Total Return: 25.05%
Max Drawdown: -8.32%
Sharpe Ratio: 1.45
Total Trades: 47
Win Rate: 55.32%
Avg Win: 5.23%
Avg Loss: -3.12%
============================================================
```

## 📝 Logging

Logs are written to both console (with colors) and file:

- **Console**: Real-time colored output
- **File**: Detailed logs in `logs/trading_bot.log`

Log levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

## 🙏 Acknowledgments

This project draws inspiration from:

- [solana-py](https://github.com/michaelhly/solana-py) - Solana Python SDK
- [Jupiter Aggregator](https://jup.ag/) - DEX aggregator API
- Various AI trading bot examples and research papers

## 📚 References

- [Solana Documentation](https://docs.solana.com/)
- [Jupiter API Documentation](https://station.jup.ag/docs/apis/swap-api)
- [Birdeye API](https://birdeye.so/docs)
- [PyTorch Documentation](https://pytorch.org/docs/)

## 📧 Support

- telegram: https://t.me/az_tekDev
- twitter:  https://x.com/az_tekDev
