import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
from typing import Dict, List
import MetaTrader5 as mt5
from strategies.rsi_strategy import RSIStrategy
from strategies.ema_strategy import EMAStrategy
from strategies.macd_strategy import MACDStrategy
from strategies.bollinger_strategy import BollingerStrategy
from backtest_engine import BacktestEngine
from optimizer import StrategyOptimizer
from walk_forward import WalkForwardAnalyzer
from monte_carlo import MonteCarloSimulator
from config.config import BACKTEST_CONFIG

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('backtest.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_historical_data(symbol: str, timeframe: int, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Get historical data from MT5."""
    # Initialize MT5
    if not mt5.initialize():
        logger.error("Failed to initialize MT5")
        return None
    
    # Convert dates to MT5 format
    start_date = int(start_date.timestamp())
    end_date = int(end_date.timestamp())
    
    # Get historical data
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    
    if rates is None:
        logger.error(f"Failed to get historical data for {symbol}")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    return df

def run_strategy_analysis(strategy_class, strategy_params: Dict, data: pd.DataFrame) -> Dict:
    """Run comprehensive strategy analysis."""
    # Create strategy instance
    strategy = strategy_class(**strategy_params)
    
    # Create backtest engine
    engine = BacktestEngine(strategy, BACKTEST_CONFIG)
    
    # Run initial backtest
    metrics = engine.run(data)
    
    # Create results directory
    results_dir = f"backtest_results/{strategy.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save initial backtest results
    engine.plot_results(f"{results_dir}/backtest_results.png")
    
    # Run parameter optimization
    param_grid = {
        'rsi_period': [7, 14, 21] if strategy.name == 'RSI' else None,
        'fast_period': [5, 9, 13] if strategy.name in ['EMA', 'MACD'] else None,
        'slow_period': [21, 34, 55] if strategy.name in ['EMA', 'MACD'] else None,
        'signal_period': [5, 9, 13] if strategy.name in ['EMA', 'MACD'] else None,
        'period': [10, 20, 30] if strategy.name == 'Bollinger' else None,
        'std_dev': [1.5, 2.0, 2.5] if strategy.name == 'Bollinger' else None,
        'risk_reward_ratio': [1.5, 2.0, 2.5]
    }
    
    # Remove None values
    param_grid = {k: v for k, v in param_grid.items() if v is not None}
    
    optimizer = StrategyOptimizer(strategy_class, param_grid)
    opt_results = optimizer.optimize(data)
    opt_results.to_csv(f"{results_dir}/optimization_results.csv")
    optimizer.plot_optimization_results(f"{results_dir}/optimization_results.png")
    
    # Get best parameters
    best_params = optimizer.get_best_params()
    
    # Run walk-forward analysis
    walk_forward = WalkForwardAnalyzer(
        strategy_class,
        param_grid,
        train_size=len(data) // 3,  # Use 1/3 of data for training
        test_size=len(data) // 6,   # Use 1/6 of data for testing
        step_size=len(data) // 12   # Move forward by 1/12 of data
    )
    wf_results = walk_forward.run_analysis(data)
    wf_results.to_csv(f"{results_dir}/walk_forward_results.csv")
    walk_forward.plot_results(f"{results_dir}/walk_forward_results.png")
    
    # Run Monte Carlo simulation
    monte_carlo = MonteCarloSimulator(engine, n_simulations=1000)
    mc_results = monte_carlo.run_simulation()
    mc_results.to_csv(f"{results_dir}/monte_carlo_results.csv")
    monte_carlo.plot_results(f"{results_dir}/monte_carlo_results.png")
    
    # Compile comprehensive results
    comprehensive_results = {
        'strategy': strategy.name,
        'initial_params': strategy_params,
        'best_params': best_params,
        'backtest_metrics': metrics,
        'optimization_results': opt_results.to_dict(),
        'walk_forward_metrics': wf_results.to_dict(),
        'monte_carlo_metrics': monte_carlo.get_risk_metrics(),
        'probability_of_profit': monte_carlo.get_probability_of_profit(),
        'parameter_stability': walk_forward.get_parameter_stability().to_dict(),
        'performance_stability': walk_forward.get_performance_stability()
    }
    
    return comprehensive_results

def main():
    """Main function to run comprehensive strategy analysis."""
    global logger
    logger = setup_logging()
    
    # Get historical data
    symbol = "EURUSD"
    timeframe = mt5.TIMEFRAME_H1
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)  # 90 days of data
    
    data = get_historical_data(symbol, timeframe, start_date, end_date)
    if data is None:
        return
    
    # Define strategies to test
    strategies = [
        {
            'class': RSIStrategy,
            'params': {
                'rsi_period': 14,
                'overbought': 70,
                'oversold': 30,
                'risk_reward_ratio': 2.0
            }
        },
        {
            'class': EMAStrategy,
            'params': {
                'fast_period': 9,
                'slow_period': 21,
                'signal_period': 9,
                'risk_reward_ratio': 2.0
            }
        },
        {
            'class': MACDStrategy,
            'params': {
                'fast_period': 12,
                'slow_period': 26,
                'signal_period': 9,
                'risk_reward_ratio': 2.0
            }
        },
        {
            'class': BollingerStrategy,
            'params': {
                'period': 20,
                'std_dev': 2.0,
                'risk_reward_ratio': 2.0
            }
        }
    ]
    
    # Run analysis for each strategy
    all_results = []
    for strategy in strategies:
        logger.info(f"Running analysis for {strategy['class'].__name__}")
        results = run_strategy_analysis(strategy['class'], strategy['params'], data)
        all_results.append(results)
    
    # Save comprehensive results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('backtest_results/comprehensive_results.csv')
    logger.info("\nAnalysis complete. Results saved in backtest_results directory.")

if __name__ == "__main__":
    main() 