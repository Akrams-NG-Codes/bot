# Akrams-NG Trading Bot

An advanced MetaTrader 5 trading bot that uses machine learning to dynamically select and optimize trading strategies based on market conditions.

## Features

- Dynamic strategy selection using unsupervised learning (K-means, DBSCAN)
- Multiple technical indicators (RSI, EMA, MACD, Bollinger Bands)
- Real-time market analysis and strategy adaptation
- Advanced risk management and position sizing
- Sentiment analysis integration
- Comprehensive logging and monitoring
- Email notifications for trades and system status
- Backtesting capabilities
- Multi-threading for performance
- Crash recovery and error handling

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/akrams-ng.git
cd akrams-ng
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Install MetaTrader 5 platform and ensure it's running

## Configuration

1. Set up your MetaTrader 5 credentials in `.env`
2. Configure email notifications
3. Set up MongoDB for logging (optional)
4. Configure Telegram bot (optional)

## Usage

1. Start the bot:
```bash
python main.py
```

2. Monitor trades and system status through:
   - Email notifications
   - Telegram bot
   - Log files
   - Web dashboard (coming soon)

## Project Structure

```
akrams-ng/
├── config/
│   ├── config.py
│   └── .env
├── core/
│   ├── bot.py
│   ├── mt5_connector.py
│   └── risk_manager.py
├── strategies/
│   ├── base_strategy.py
│   ├── rsi_strategy.py
│   ├── ema_strategy.py
│   ├── macd_strategy.py
│   └── bollinger_strategy.py
├── ml/
│   ├── market_classifier.py
│   ├── strategy_selector.py
│   └── model_trainer.py
├── utils/
│   ├── logger.py
│   ├── notifier.py
│   └── data_processor.py
├── backtesting/
│   └── backtest_engine.py
├── tests/
│   └── test_*.py
├── main.py
├── requirements.txt
└── README.md
```

## Risk Warning

Trading involves significant risk. This bot is for educational purposes only. Always test thoroughly with demo accounts before using real money.

## License

MIT License

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests. 