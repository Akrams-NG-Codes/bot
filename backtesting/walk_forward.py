import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta
from .backtest_engine import BacktestEngine
from .optimizer import StrategyOptimizer

class WalkForwardAnalyzer:
    def __init__(self, strategy_class, param_grid: Dict[str, List], 
                 train_size: int, test_size: int, step_size: int):
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        self.logger = logging.getLogger(__name__)
        self.results = []

    def run_analysis(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run walk-forward analysis."""
        total_bars = len(data)
        current_pos = 0
        
        while current_pos + self.train_size + self.test_size <= total_bars:
            # Split data into training and testing sets
            train_data = data.iloc[current_pos:current_pos + self.train_size]
            test_data = data.iloc[current_pos + self.train_size:current_pos + self.train_size + self.test_size]
            
            # Optimize parameters on training data
            optimizer = StrategyOptimizer(self.strategy_class, self.param_grid)
            train_results = optimizer.optimize(train_data)
            best_params = optimizer.get_best_params()
            
            # Test strategy with optimized parameters
            strategy = self.strategy_class(**best_params)
            engine = BacktestEngine(strategy)
            test_metrics = engine.run(test_data)
            
            # Record results
            self.results.append({
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'best_params': best_params,
                'test_metrics': test_metrics
            })
            
            # Move forward
            current_pos += self.step_size
        
        return self._summarize_results()

    def _summarize_results(self) -> pd.DataFrame:
        """Summarize walk-forward analysis results."""
        summary = []
        
        for result in self.results:
            metrics = result['test_metrics']
            summary.append({
                'period_start': result['test_start'],
                'period_end': result['test_end'],
                'total_trades': metrics['total_trades'],
                'win_rate': metrics['win_rate'],
                'profit_factor': metrics['profit_factor'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown'],
                'total_profit': metrics['total_profit'],
                'total_loss': metrics['total_loss'],
                'average_win': metrics['average_win'],
                'average_loss': metrics['average_loss']
            })
        
        return pd.DataFrame(summary)

    def plot_results(self, save_path: str = None):
        """Plot walk-forward analysis results."""
        if not self.results:
            raise ValueError("No walk-forward analysis results available")
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create summary DataFrame
        summary_df = self._summarize_results()
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot win rate over time
        sns.lineplot(data=summary_df, x='period_start', y='win_rate', ax=ax1)
        ax1.set_title('Win Rate Over Time')
        ax1.set_xlabel('Period')
        ax1.set_ylabel('Win Rate')
        ax1.grid(True)
        
        # Plot profit factor over time
        sns.lineplot(data=summary_df, x='period_start', y='profit_factor', ax=ax2)
        ax2.set_title('Profit Factor Over Time')
        ax2.set_xlabel('Period')
        ax2.set_ylabel('Profit Factor')
        ax2.grid(True)
        
        # Plot Sharpe ratio over time
        sns.lineplot(data=summary_df, x='period_start', y='sharpe_ratio', ax=ax3)
        ax3.set_title('Sharpe Ratio Over Time')
        ax3.set_xlabel('Period')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.grid(True)
        
        # Plot max drawdown over time
        sns.lineplot(data=summary_df, x='period_start', y='max_drawdown', ax=ax4)
        ax4.set_title('Maximum Drawdown Over Time')
        ax4.set_xlabel('Period')
        ax4.set_ylabel('Maximum Drawdown')
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def get_parameter_stability(self) -> pd.DataFrame:
        """Analyze parameter stability across walk-forward periods."""
        if not self.results:
            raise ValueError("No walk-forward analysis results available")
        
        # Extract parameters for each period
        param_data = []
        for result in self.results:
            params = result['best_params']
            params['period'] = result['test_start']
            param_data.append(params)
        
        # Convert to DataFrame
        param_df = pd.DataFrame(param_data)
        
        # Calculate parameter statistics
        param_stats = []
        for param in self.param_grid.keys():
            stats = param_df[param].describe()
            param_stats.append({
                'parameter': param,
                'mean': stats['mean'],
                'std': stats['std'],
                'min': stats['min'],
                'max': stats['max'],
                'stability': 1 - (stats['std'] / stats['mean']) if stats['mean'] != 0 else 0
            })
        
        return pd.DataFrame(param_stats).sort_values('stability', ascending=False)

    def get_performance_stability(self) -> Dict:
        """Calculate performance stability metrics."""
        if not self.results:
            raise ValueError("No walk-forward analysis results available")
        
        summary_df = self._summarize_results()
        
        stability_metrics = {
            'win_rate': {
                'mean': summary_df['win_rate'].mean(),
                'std': summary_df['win_rate'].std(),
                'stability': 1 - (summary_df['win_rate'].std() / summary_df['win_rate'].mean())
            },
            'profit_factor': {
                'mean': summary_df['profit_factor'].mean(),
                'std': summary_df['profit_factor'].std(),
                'stability': 1 - (summary_df['profit_factor'].std() / summary_df['profit_factor'].mean())
            },
            'sharpe_ratio': {
                'mean': summary_df['sharpe_ratio'].mean(),
                'std': summary_df['sharpe_ratio'].std(),
                'stability': 1 - (summary_df['sharpe_ratio'].std() / summary_df['sharpe_ratio'].mean())
            }
        }
        
        return stability_metrics 