import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class PerformanceAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.trades = []
        self.equity_curve = pd.Series()
        self.initial_balance = config.get('initial_balance', 10000)
        self.risk_free_rate = config.get('risk_free_rate', 0.02)  # 2% annual risk-free rate

    def add_trade(self, trade: Dict):
        """Add a trade to the analysis."""
        self.trades.append(trade)
        self._update_equity_curve()

    def _update_equity_curve(self):
        """Update equity curve based on trades."""
        if not self.trades:
            return
        
        # Create DataFrame from trades
        trades_df = pd.DataFrame(self.trades)
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        trades_df.set_index('timestamp', inplace=True)
        
        # Calculate cumulative P&L
        self.equity_curve = self.initial_balance + trades_df['pnl'].cumsum()

    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not self.trades:
            return {}
        
        trades_df = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L metrics
        total_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        total_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
        net_profit = total_profit - total_loss
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Risk metrics
        returns = trades_df['pnl'] / self.initial_balance
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        calmar_ratio = self._calculate_calmar_ratio()
        max_drawdown = self._calculate_max_drawdown()
        
        # Trade metrics
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
        largest_win = trades_df['pnl'].max()
        largest_loss = trades_df['pnl'].min()
        
        # Time-based metrics
        avg_trade_duration = trades_df['duration'].mean()
        trades_per_day = total_trades / ((trades_df.index[-1] - trades_df.index[0]).days + 1)
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'net_profit': net_profit,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'avg_trade_duration': avg_trade_duration,
            'trades_per_day': trades_per_day
        }

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        excess_returns = returns - self.risk_free_rate / 252  # Daily risk-free rate
        return np.sqrt(252) * excess_returns.mean() / returns.std()

    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        if len(returns) < 2:
            return 0.0
        excess_returns = returns - self.risk_free_rate / 252
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf')
        downside_std = np.sqrt(np.mean(downside_returns ** 2))
        return np.sqrt(252) * excess_returns.mean() / downside_std

    def _calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio."""
        if len(self.equity_curve) < 2:
            return 0.0
        annual_return = (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) ** (252 / len(self.equity_curve)) - 1
        max_drawdown = self._calculate_max_drawdown()
        return annual_return / max_drawdown if max_drawdown > 0 else float('inf')

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if len(self.equity_curve) < 2:
            return 0.0
        rolling_max = self.equity_curve.expanding().max()
        drawdowns = (self.equity_curve - rolling_max) / rolling_max
        return abs(drawdowns.min())

    def plot_performance(self, save_path: Optional[str] = None):
        """Create comprehensive performance visualization."""
        if not self.trades:
            return
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Equity Curve', 'Drawdown',
                'Trade P&L Distribution', 'Win Rate by Hour',
                'Monthly Returns', 'Trade Duration vs P&L'
            )
        )
        
        # Plot equity curve
        fig.add_trace(
            go.Scatter(x=self.equity_curve.index, y=self.equity_curve.values,
                      name='Equity', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Plot drawdown
        rolling_max = self.equity_curve.expanding().max()
        drawdowns = (self.equity_curve - rolling_max) / rolling_max * 100
        fig.add_trace(
            go.Scatter(x=drawdowns.index, y=drawdowns.values,
                      name='Drawdown', line=dict(color='red')),
            row=1, col=2
        )
        
        # Plot P&L distribution
        trades_df = pd.DataFrame(self.trades)
        fig.add_trace(
            go.Histogram(x=trades_df['pnl'], name='P&L Distribution',
                        nbinsx=50, marker_color='green'),
            row=2, col=1
        )
        
        # Plot win rate by hour
        trades_df['hour'] = pd.to_datetime(trades_df['timestamp']).dt.hour
        win_rate_by_hour = trades_df.groupby('hour')['pnl'].apply(
            lambda x: (x > 0).mean() * 100
        )
        fig.add_trace(
            go.Bar(x=win_rate_by_hour.index, y=win_rate_by_hour.values,
                  name='Win Rate by Hour', marker_color='purple'),
            row=2, col=2
        )
        
        # Plot monthly returns
        trades_df['month'] = pd.to_datetime(trades_df['timestamp']).dt.to_period('M')
        monthly_returns = trades_df.groupby('month')['pnl'].sum()
        fig.add_trace(
            go.Bar(x=monthly_returns.index.astype(str), y=monthly_returns.values,
                  name='Monthly Returns', marker_color='orange'),
            row=3, col=1
        )
        
        # Plot trade duration vs P&L
        fig.add_trace(
            go.Scatter(x=trades_df['duration'], y=trades_df['pnl'],
                      mode='markers', name='Duration vs P&L',
                      marker=dict(
                          color=trades_df['pnl'],
                          colorscale='RdYlGn',
                          showscale=True
                      )),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            width=1600,
            title_text='Trading Performance Analysis',
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()

    def generate_trade_journal(self, save_path: Optional[str] = None) -> pd.DataFrame:
        """Generate detailed trade journal."""
        if not self.trades:
            return pd.DataFrame()
        
        trades_df = pd.DataFrame(self.trades)
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        
        # Add additional metrics
        trades_df['win'] = trades_df['pnl'] > 0
        trades_df['return'] = trades_df['pnl'] / self.initial_balance
        trades_df['cumulative_return'] = trades_df['return'].cumsum()
        
        # Calculate trade statistics
        trades_df['hour'] = trades_df['timestamp'].dt.hour
        trades_df['day_of_week'] = trades_df['timestamp'].dt.day_name()
        trades_df['month'] = trades_df['timestamp'].dt.month_name()
        
        if save_path:
            trades_df.to_csv(save_path, index=False)
        
        return trades_df

    def get_heatmap_data(self) -> pd.DataFrame:
        """Generate heatmap data for trade analysis."""
        if not self.trades:
            return pd.DataFrame()
        
        trades_df = pd.DataFrame(self.trades)
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        
        # Create hour-day matrix
        trades_df['hour'] = trades_df['timestamp'].dt.hour
        trades_df['day'] = trades_df['timestamp'].dt.day_name()
        
        # Calculate win rate for each hour-day combination
        heatmap_data = trades_df.groupby(['day', 'hour'])['pnl'].apply(
            lambda x: (x > 0).mean() * 100
        ).unstack()
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data = heatmap_data.reindex(day_order)
        
        return heatmap_data

    def plot_heatmap(self, save_path: Optional[str] = None):
        """Plot trade heatmap."""
        heatmap_data = self.get_heatmap_data()
        if heatmap_data.empty:
            return
        
        plt.figure(figsize=(15, 8))
        sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', center=50,
                   fmt='.1f', cbar_kws={'label': 'Win Rate (%)'})
        plt.title('Trade Win Rate by Hour and Day')
        plt.xlabel('Hour')
        plt.ylabel('Day')
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show() 