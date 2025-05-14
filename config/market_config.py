MARKET_CONFIG = {
    # Market regime detection
    'regime_period': 20,         # Days for regime detection
    'regime_threshold': 0.7,     # Threshold for regime change
    'regime_features': [         # Features for regime detection
        'returns',
        'volatility',
        'rsi',
        'macd'
    ],
    
    # Volatility analysis
    'volatility_period': 14,     # Days for volatility calculation
    'volatility_methods': [      # Volatility calculation methods
        'historical',
        'parkinson',
        'garman_klass'
    ],
    'volatility_rank_period': 365,  # Days for volatility ranking
    
    # Trend analysis
    'trend_periods': [           # Periods for trend analysis
        20, 50, 100, 200
    ],
    'trend_indicators': [        # Trend indicators to use
        'sma',
        'ema',
        'macd',
        'adx'
    ],
    'trend_threshold': 25,       # ADX threshold for trend strength
    
    # Support/Resistance
    'sr_period': 100,           # Periods for S/R calculation
    'sr_clusters': 5,           # Number of S/R levels to identify
    'sr_window': 5,             # Window for local min/max detection
    
    # Correlation analysis
    'correlation_period': 30,    # Days for correlation calculation
    'correlation_threshold': 0.7,  # Threshold for high correlation
    'correlation_method': 'pearson',  # Correlation method
    
    # Market data
    'timeframe': 'H1',          # Default timeframe
    'data_cache_size': 1000,    # Number of periods to cache
    'update_interval': 60,      # Data update interval in seconds
    
    # Technical indicators
    'indicators': {
        'rsi': {
            'period': 14,
            'overbought': 70,
            'oversold': 30
        },
        'macd': {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9
        },
        'bollinger': {
            'period': 20,
            'std_dev': 2
        },
        'atr': {
            'period': 14
        }
    },
    
    # Market filters
    'min_volume': 1000,         # Minimum volume for analysis
    'min_spread': 0.0001,       # Maximum allowed spread
    'min_volatility': 0.0001,   # Minimum volatility for analysis
    'max_volatility': 0.05,     # Maximum volatility for analysis
    
    # Market hours
    'trading_hours': {
        'forex': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        },
        'stocks': {
            'start': '09:30',
            'end': '16:00',
            'timezone': 'America/New_York'
        }
    },
    
    # Market regimes
    'regime_characteristics': {
        'trending': {
            'min_adx': 25,
            'min_trend_duration': 5
        },
        'ranging': {
            'max_adx': 20,
            'min_bb_width': 0.01
        },
        'volatile': {
            'min_volatility': 0.02,
            'min_atr': 0.001
        }
    },
    
    # Performance metrics
    'performance_metrics': [
        'sharpe_ratio',
        'sortino_ratio',
        'calmar_ratio',
        'max_drawdown',
        'win_rate',
        'profit_factor'
    ],
    
    # Visualization
    'plot_settings': {
        'figsize': (15, 10),
        'style': 'seaborn',
        'save_format': 'png',
        'dpi': 300
    }
} 