STRATEGY_CONFIG = {
    # General strategy settings
    'general': {
        'enabled': True,
        'symbols': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'],
        'timeframes': ['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
        'max_positions': 5,
        'magic_number': 123456,
        'slippage': 10,
    },
    
    # Risk management settings
    'risk': {
        'max_risk_per_trade': 0.02,  # 2% of account
        'max_daily_risk': 0.05,  # 5% of account
        'max_drawdown': 0.15,  # 15% of account
        'stop_loss_atr_multiplier': 2.0,
        'take_profit_atr_multiplier': 3.0,
        'trailing_stop': True,
        'trailing_stop_activation': 0.5,  # 50% of take profit
        'trailing_stop_distance': 1.0,  # ATR multiplier
    },
    
    # Position sizing settings
    'position_sizing': {
        'method': 'risk_based',  # 'risk_based', 'fixed', 'kelly'
        'fixed_size': 0.1,  # Fixed lot size
        'max_position_size': 1.0,  # Maximum lot size
        'min_position_size': 0.01,  # Minimum lot size
        'size_increment': 0.01,  # Lot size increment
    },
    
    # Entry settings
    'entry': {
        'confirmation_timeframes': ['5m', '15m', '1h'],
        'min_volume': 100,
        'min_volatility': 0.0001,
        'max_spread': 0.0005,
        'entry_filters': {
            'trend': True,
            'momentum': True,
            'volatility': True,
            'volume': True,
            'time': True
        }
    },
    
    # Exit settings
    'exit': {
        'use_trailing_stop': True,
        'use_break_even': True,
        'break_even_activation': 0.5,  # 50% of take profit
        'partial_exit': True,
        'partial_exit_levels': [0.5, 0.75, 1.0],  # Take profit levels
        'partial_exit_sizes': [0.3, 0.3, 0.4],  # Position sizes to close
    },
    
    # Technical analysis settings
    'technical': {
        'indicators': {
            'sma': {
                'enabled': True,
                'periods': [20, 50, 200]
            },
            'ema': {
                'enabled': True,
                'periods': [9, 21, 50]
            },
            'rsi': {
                'enabled': True,
                'period': 14,
                'overbought': 70,
                'oversold': 30
            },
            'macd': {
                'enabled': True,
                'fast_period': 12,
                'slow_period': 26,
                'signal_period': 9
            },
            'bollinger_bands': {
                'enabled': True,
                'period': 20,
                'std_dev': 2
            },
            'atr': {
                'enabled': True,
                'period': 14
            }
        },
        'patterns': {
            'candlestick': True,
            'chart': True,
            'harmonic': True
        }
    },
    
    # Time-based filters
    'time_filters': {
        'trading_hours': {
            'enabled': True,
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        },
        'session_filters': {
            'london': True,
            'new_york': True,
            'tokyo': True,
            'sydney': True
        },
        'day_filters': {
            'monday': True,
            'tuesday': True,
            'wednesday': True,
            'thursday': True,
            'friday': True,
            'saturday': False,
            'sunday': False
        }
    },
    
    # News filters
    'news_filters': {
        'enabled': True,
        'high_impact': True,
        'medium_impact': True,
        'low_impact': False,
        'pre_news_time': 60,  # minutes
        'post_news_time': 30,  # minutes
        'currencies': ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD']
    },
    
    # Performance monitoring
    'monitoring': {
        'enabled': True,
        'metrics_interval': 60,  # seconds
        'alert_thresholds': {
            'max_drawdown': 0.15,
            'min_win_rate': 0.4,
            'max_loss_streak': 5,
            'min_profit_factor': 1.5
        }
    },
    
    # Logging settings
    'logging': {
        'level': 'INFO',
        'save_trades': True,
        'save_signals': True,
        'log_path': 'logs/strategy'
    }
} 