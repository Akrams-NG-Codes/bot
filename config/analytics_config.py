ANALYTICS_CONFIG = {
    # Performance metrics
    'risk_free_rate': 0.02,  # 2% annual risk-free rate
    'initial_balance': 10000,
    
    # Trade analysis
    'trade_metrics': [
        'win_rate',
        'profit_factor',
        'sharpe_ratio',
        'sortino_ratio',
        'calmar_ratio',
        'max_drawdown',
        'avg_win',
        'avg_loss',
        'largest_win',
        'largest_loss',
        'avg_trade_duration',
        'trades_per_day'
    ],
    
    # Time-based analysis
    'time_periods': {
        'daily': True,
        'weekly': True,
        'monthly': True,
        'yearly': True
    },
    
    # Visualization settings
    'plot_settings': {
        'equity_curve': {
            'color': 'blue',
            'line_width': 2,
            'show_grid': True
        },
        'drawdown': {
            'color': 'red',
            'line_width': 2,
            'show_grid': True
        },
        'pnl_distribution': {
            'bins': 50,
            'color': 'green',
            'show_kde': True
        },
        'win_rate': {
            'color': 'purple',
            'show_grid': True
        },
        'monthly_returns': {
            'color': 'orange',
            'show_grid': True
        },
        'heatmap': {
            'cmap': 'RdYlGn',
            'center': 50,
            'annot': True,
            'fmt': '.1f'
        }
    },
    
    # Trade journal settings
    'journal_columns': [
        'timestamp',
        'symbol',
        'direction',
        'entry_price',
        'exit_price',
        'pnl',
        'duration',
        'win',
        'return',
        'cumulative_return',
        'hour',
        'day_of_week',
        'month'
    ],
    
    # Performance thresholds
    'thresholds': {
        'min_win_rate': 0.5,
        'min_profit_factor': 1.5,
        'min_sharpe_ratio': 1.0,
        'min_sortino_ratio': 1.0,
        'max_drawdown': 0.2,
        'min_trades_per_day': 1,
        'max_trades_per_day': 10
    },
    
    # Risk metrics
    'risk_metrics': {
        'var_confidence': 0.95,
        'var_window': 252,
        'cvar_confidence': 0.95,
        'max_drawdown_window': 252
    },
    
    # Correlation analysis
    'correlation': {
        'window': 30,
        'min_correlation': 0.7,
        'max_correlation': -0.7
    },
    
    # Trade clustering
    'clustering': {
        'n_clusters': 5,
        'features': [
            'pnl',
            'duration',
            'hour',
            'day_of_week'
        ]
    },
    
    # Export settings
    'export': {
        'format': 'csv',
        'include_trades': True,
        'include_metrics': True,
        'include_plots': True,
        'save_path': 'results/analytics'
    },
    
    # Notification settings
    'notifications': {
        'enable': True,
        'threshold_breach': True,
        'daily_summary': True,
        'weekly_summary': True,
        'monthly_summary': True
    }
} 