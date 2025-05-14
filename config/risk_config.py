RISK_CONFIG = {
    'risk': {
        'max_risk_per_trade': 0.02,  # 2% risk per trade
        'max_daily_risk': 0.05,      # 5% max daily risk
        'max_drawdown': 0.15,        # 15% max drawdown
        'stop_loss_atr_multiplier': 2.0,  # ATR multiplier for stop loss
        'take_profit_atr_multiplier': 3.0,  # ATR multiplier for take profit
        'trailing_stop_atr_multiplier': 1.5,  # ATR multiplier for trailing stop
        'break_even_atr_multiplier': 1.0,  # ATR multiplier for break even
    },
    
    'position_sizing': {
        'method': 'risk_based',  # Options: 'risk_based', 'fixed', 'kelly'
        'fixed_size': 0.1,       # Fixed lot size if method is 'fixed'
        'max_position_size': 1.0,  # Maximum position size in lots
        'min_position_size': 0.01,  # Minimum position size in lots
        'size_increment': 0.01,   # Position size increment
        'max_positions': 5,       # Maximum number of concurrent positions
        'max_positions_per_symbol': 1,  # Maximum positions per symbol
    },
    
    'stop_loss': {
        'enabled': True,
        'type': 'atr',  # Options: 'atr', 'fixed', 'percentage'
        'fixed_pips': 50,  # Fixed stop loss in pips
        'percentage': 0.01,  # Percentage stop loss
        'trailing_enabled': True,
        'trailing_activation': 0.5,  # Activate trailing stop at 50% of take profit
        'trailing_step': 10,  # Trailing stop step in pips
    },
    
    'take_profit': {
        'enabled': True,
        'type': 'atr',  # Options: 'atr', 'fixed', 'percentage'
        'fixed_pips': 100,  # Fixed take profit in pips
        'percentage': 0.02,  # Percentage take profit
        'partial_exit': {
            'enabled': True,
            'levels': [0.5, 0.25, 0.25],  # Exit 50%, 25%, 25% at each level
            'targets': [0.5, 0.75, 1.0],  # Take profit targets as percentage of initial target
        },
    },
    
    'break_even': {
        'enabled': True,
        'activation': 0.5,  # Activate break even at 50% of take profit
        'offset': 5,  # Break even offset in pips
    },
    
    'correlation': {
        'enabled': True,
        'max_correlation': 0.7,  # Maximum allowed correlation between positions
        'lookback_period': 20,  # Period for correlation calculation
        'symbols': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'],  # Symbols to check correlation
    },
    
    'volatility': {
        'enabled': True,
        'max_atr': 100,  # Maximum allowed ATR in pips
        'min_atr': 20,   # Minimum required ATR in pips
        'period': 14,    # ATR period
    },
    
    'time_based': {
        'enabled': True,
        'max_holding_time': 24,  # Maximum holding time in hours
        'time_exit': {
            'enabled': True,
            'hours': [16, 20],  # Exit positions at these hours
            'days': [4],  # Exit positions on these days (0=Monday, 6=Sunday)
        },
    },
    
    'news': {
        'enabled': True,
        'pre_news_time': 30,  # Minutes before news to avoid trading
        'post_news_time': 15,  # Minutes after news to avoid trading
        'high_impact_only': True,  # Only consider high impact news
        'currencies': ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD'],  # Currencies to monitor
    },
    
    'monitoring': {
        'enabled': True,
        'metrics_interval': 300,  # Seconds between metrics updates
        'alert_thresholds': {
            'drawdown': 0.1,  # Alert at 10% drawdown
            'win_rate': 0.4,  # Alert if win rate below 40%
            'loss_streak': 5,  # Alert after 5 consecutive losses
            'profit_factor': 1.5,  # Alert if profit factor below 1.5
        },
    },
    
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file': 'logs/risk_manager.log',
        'max_size': 10485760,  # 10MB
        'backup_count': 5,
    },
    
    'persistence': {
        'enabled': True,
        'metrics_path': 'data/risk_metrics',
        'save_interval': 3600,  # Save metrics every hour
    },
} 