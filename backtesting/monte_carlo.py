import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from .backtest_engine import BacktestEngine

class MonteCarloSimulator:
    def __init__(self, backtest_engine: BacktestEngine, n_simulations: int = 1000):
        self.engine = backtest_engine
        self.n_simulations = n_simulations
        self.logger = logging.getLogger(__name__)
        self.simulation_results = []

    def run_simulation(self) -> pd.DataFrame:
        """Run Monte Carlo simulation on trade history."""
        if not self.engine.trades:
            raise ValueError("No trade history available for simulation")
        
        # Get trade returns
        trade_returns = [t['pnl'] for t in self.engine.trades]
        n_trades = len(trade_returns)
        
        # Run simulations
        for _ in range(self.n_simulations):
            # Shuffle trade returns
            shuffled_returns = np.random.permutation(trade_returns)
            
            # Calculate cumulative returns
            cumulative_returns = np.cumsum(shuffled_returns)
            
            # Calculate metrics
            final_equity = self.engine.initial_balance + cumulative_returns[-1]
            max_drawdown = self._calculate_drawdown(cumulative_returns)
            sharpe_ratio = self._calculate_sharpe_ratio(shuffled_returns)
            
            self.simulation_results.append({
                'final_equity': final_equity,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'returns': shuffled_returns,
                'cumulative_returns': cumulative_returns
            })
        
        return self._summarize_results()

    def _calculate_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns."""
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / running_max
        return np.max(drawdown)

    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio from returns."""
        if len(returns) < 2:
            return 0.0
        return np.sqrt(252) * np.mean(returns) / np.std(returns)

    def _summarize_results(self) -> pd.DataFrame:
        """Summarize Monte Carlo simulation results."""
        summary = []
        
        for result in self.simulation_results:
            summary.append({
                'final_equity': result['final_equity'],
                'max_drawdown': result['max_drawdown'],
                'sharpe_ratio': result['sharpe_ratio']
            })
        
        return pd.DataFrame(summary)

    def plot_results(self, save_path: str = None):
        """Plot Monte Carlo simulation results."""
        if not self.simulation_results:
            raise ValueError("No simulation results available")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot equity curves
        for result in self.simulation_results:
            cumulative_returns = result['cumulative_returns']
            equity_curve = self.engine.initial_balance + cumulative_returns
            ax1.plot(equity_curve, alpha=0.1, color='blue')
        
        ax1.set_title('Monte Carlo Equity Curves')
        ax1.set_xlabel('Trade Number')
        ax1.set_ylabel('Equity')
        ax1.grid(True)
        
        # Plot final equity distribution
        final_equities = [r['final_equity'] for r in self.simulation_results]
        sns.histplot(final_equities, ax=ax2, bins=50)
        ax2.set_title('Final Equity Distribution')
        ax2.set_xlabel('Final Equity')
        ax2.set_ylabel('Frequency')
        ax2.grid(True)
        
        # Plot max drawdown distribution
        max_drawdowns = [r['max_drawdown'] for r in self.simulation_results]
        sns.histplot(max_drawdowns, ax=ax3, bins=50)
        ax3.set_title('Maximum Drawdown Distribution')
        ax3.set_xlabel('Maximum Drawdown')
        ax3.set_ylabel('Frequency')
        ax3.grid(True)
        
        # Plot Sharpe ratio distribution
        sharpe_ratios = [r['sharpe_ratio'] for r in self.simulation_results]
        sns.histplot(sharpe_ratios, ax=ax4, bins=50)
        ax4.set_title('Sharpe Ratio Distribution')
        ax4.set_xlabel('Sharpe Ratio')
        ax4.set_ylabel('Frequency')
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def get_risk_metrics(self) -> Dict:
        """Calculate risk metrics from simulation results."""
        if not self.simulation_results:
            raise ValueError("No simulation results available")
        
        summary_df = self._summarize_results()
        
        risk_metrics = {
            'final_equity': {
                'mean': summary_df['final_equity'].mean(),
                'std': summary_df['final_equity'].std(),
                'min': summary_df['final_equity'].min(),
                'max': summary_df['final_equity'].max(),
                'var_95': np.percentile(summary_df['final_equity'], 5),
                'var_99': np.percentile(summary_df['final_equity'], 1)
            },
            'max_drawdown': {
                'mean': summary_df['max_drawdown'].mean(),
                'std': summary_df['max_drawdown'].std(),
                'max': summary_df['max_drawdown'].max(),
                'var_95': np.percentile(summary_df['max_drawdown'], 95),
                'var_99': np.percentile(summary_df['max_drawdown'], 99)
            },
            'sharpe_ratio': {
                'mean': summary_df['sharpe_ratio'].mean(),
                'std': summary_df['sharpe_ratio'].std(),
                'min': summary_df['sharpe_ratio'].min(),
                'max': summary_df['sharpe_ratio'].max()
            }
        }
        
        return risk_metrics

    def get_probability_of_profit(self) -> float:
        """Calculate probability of ending with a profit."""
        if not self.simulation_results:
            raise ValueError("No simulation results available")
        
        final_equities = [r['final_equity'] for r in self.simulation_results]
        profitable_simulations = sum(1 for equity in final_equities if equity > self.engine.initial_balance)
        
        return profitable_simulations / self.n_simulations 