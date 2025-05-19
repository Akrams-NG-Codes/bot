import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MetaTrader 5 Configuration
MT5_LOGIN = os.getenv('MT5_LOGIN')
MT5_PASSWORD = os.getenv('MT5_PASSWORD')
MT5_SERVER = os.getenv('MT5_SERVER')
MT5_TIMEOUT = int(os.getenv('MT5_TIMEOUT', '60000'))

# Trading Configuration
SYMBOLS = os.getenv('SYMBOLS', 'EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD').split(',')
TIMEFRAMES = {
    'M1': 1,
    'M5': 5,
    'M15': 15,
    'M30': 30,
    'H1': 60,
    'H4': 240,
    'D1': 1440
}

# Risk Management
MAX_DRAWDOWN = float(os.getenv('MAX_DRAWDOWN', '0.02'))  # 2% max drawdown
MAX_DAILY_TRADES = int(os.getenv('MAX_DAILY_TRADES', '10'))
RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', '0.01'))  # 1% risk per trade
MAX_OPEN_POSITIONS = int(os.getenv('MAX_OPEN_POSITIONS', '5'))

# Strategy Parameters
STRATEGIES = {
    'RSI': {
        'period': 14,
        'overbought': 70,
        'oversold': 30
    },
    'EMA': {
        'fast_period': 9,
        'slow_period': 21
    },
    'MACD': {
        'fast_period': 12,
        'slow_period': 26,
        'signal_period': 9
    },
    'Bollinger': {
        'period': 20,
        'std_dev': 2
    }
}

# Machine Learning Configuration
ML_CONFIG = {
    'model_path': 'models',
    'retrain_interval': 24,  # hours
    'min_training_samples': 1000,
    'prediction_threshold': 0.7
}

# Notification Configuration
EMAIL_CONFIG = {
    'enabled': False,
    'sender': os.getenv('EMAIL_SENDER'),
    'password': os.getenv('EMAIL_PASSWORD'),
    'recipient': os.getenv('EMAIL_RECIPIENT'),
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587
}

TELEGRAM_CONFIG = {
    'enabled': False,
    'bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
    'chat_id': os.getenv('TELEGRAM_CHAT_ID')
}

# Logging Configuration
LOG_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': 'logs/trading_bot.log',
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': True
        }
    }
}

# MongoDB Configuration (for logging)
MONGODB_CONFIG = {
    'uri': os.getenv('MONGODB_URI', ''),
    'database': os.getenv('MONGODB_DATABASE', 'akrams_ng'),
    'collections': {
        'trades': 'trades',
        'performance': 'performance',
        'logs': 'logs'
    }
}

# Sentiment Analysis Configuration
SENTIMENT_CONFIG = {
    'twitter_api_key': os.getenv('TWITTER_API_KEY', ''),
    'twitter_api_secret': os.getenv('TWITTER_API_SECRET', ''),
    'twitter_access_token': os.getenv('TWITTER_ACCESS_TOKEN', ''),
    'twitter_access_token_secret': os.getenv('TWITTER_ACCESS_TOKEN_SECRET', ''),
    'news_api_key': os.getenv('NEWS_API_KEY', '')
}

# Backtesting Configuration
BACKTEST_CONFIG = {
    'start_date': os.getenv('BACKTEST_START_DATE', '2023-01-01'),
    'end_date': os.getenv('BACKTEST_END_DATE', '2023-12-31'),
    'initial_balance': float(os.getenv('BACKTEST_INITIAL_BALANCE', '10000')),
    'commission': float(os.getenv('BACKTEST_COMMISSION', '0.0001'))
} 