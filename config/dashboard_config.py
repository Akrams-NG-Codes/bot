DASHBOARD_CONFIG = {
    # Server settings
    'server': {
        'host': 'localhost',
        'port': 8050,
        'debug': False,
        'threaded': True
    },
    
    # Update intervals
    'update_intervals': {
        'metrics': 1000,  # Update metrics every second
        'charts': 5000,   # Update charts every 5 seconds
        'trade_log': 1000  # Update trade log every second
    },
    
    # Display settings
    'display': {
        'theme': 'light',
        'colors': {
            'primary': '#2c3e50',
            'secondary': '#34495e',
            'success': '#27ae60',
            'danger': '#c0392b',
            'warning': '#f39c12',
            'info': '#3498db'
        },
        'font_family': 'Arial, sans-serif',
        'font_size': {
            'header': '24px',
            'subheader': '18px',
            'text': '14px'
        }
    },
    
    # Chart settings
    'charts': {
        'equity_curve': {
            'height': 400,
            'width': '100%',
            'template': 'plotly_white',
            'show_grid': True,
            'show_legend': True
        },
        'trade_distribution': {
            'height': 300,
            'width': '100%',
            'template': 'plotly_white',
            'show_grid': True,
            'bins': 50
        },
        'time_analysis': {
            'height': 600,
            'width': '100%',
            'template': 'plotly_white',
            'show_grid': True
        }
    },
    
    # Metrics display
    'metrics': {
        'account': [
            'balance',
            'daily_pnl',
            'open_positions'
        ],
        'performance': [
            'win_rate',
            'profit_factor',
            'sharpe_ratio'
        ],
        'risk': [
            'current_drawdown',
            'max_drawdown',
            'var_95'
        ]
    },
    
    # Trade log settings
    'trade_log': {
        'max_trades': 10,
        'columns': [
            'timestamp',
            'symbol',
            'direction',
            'entry_price',
            'exit_price',
            'pnl'
        ],
        'sort_by': 'timestamp',
        'sort_order': 'desc'
    },
    
    # Alert settings
    'alerts': {
        'enable': True,
        'thresholds': {
            'drawdown': 0.1,  # Alert at 10% drawdown
            'daily_loss': 0.05,  # Alert at 5% daily loss
            'win_rate': 0.4,  # Alert if win rate falls below 40%
            'profit_factor': 1.0  # Alert if profit factor falls below 1.0
        },
        'notification': {
            'sound': True,
            'desktop': True,
            'email': False
        }
    },
    
    # Data retention
    'data_retention': {
        'max_trades': 1000,
        'max_equity_points': 10000,
        'cleanup_interval': 3600  # Clean up old data every hour
    },
    
    # Export settings
    'export': {
        'enable': True,
        'format': 'csv',
        'interval': 'daily',
        'path': 'exports/dashboard'
    },
    
    # Security settings
    'security': {
        'require_auth': False,
        'allowed_ips': ['127.0.0.1'],
        'session_timeout': 3600  # 1 hour
    }
} 