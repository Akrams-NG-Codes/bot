import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from strategies.base_strategy import BaseStrategy
from config.config import BACKTEST_CONFIG

class BacktestEngine:
    def __init__(self, strategy: BaseStrategy, config: Dict = None):
        self.strategy = strategy
        self.config = config or BACKTEST_CONFIG
        self.logger = logging.getLogger(__name__)
        
        # Initialize results
        self.trades = []
        self.equity_curve = []
        self.initial_balance = self.config['initial_balance']
        self.current_balance = self.initial_balance
        self.positions = []
        
        # Performance metrics
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'risk_reward_ratio': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0
        }

    def run(self, data: pd.DataFrame) -> Dict:
        """Run backtest on historical data."""
        self.logger.info(f"Starting backtest for {self.strategy.name} strategy")
        
        # Calculate indicators
        data = self.strategy.calculate_indicators(data)
        
        # Generate signals
        signals = self.strategy.generate_signals(data)
        
        # Process each bar
        for i in range(len(data)):
            current_bar = data.iloc[i]
            current_signal = signals.iloc[i]['signal']
            
            # Update existing positions
            self._update_positions(current_bar)
            
            # Check for new trade signals
            if current_signal != 0:
                self._process_signal(current_bar, current_signal)
            
            # Update equity curve
            self._update_equity_curve(current_bar)
        
        # Calculate performance metrics
        self._calculate_metrics()
        
        return self.metrics

    def _process_signal(self, bar: pd.Series, signal: int):
        """Process a new trading signal."""
        # Check if we can open a new position
        if len(self.positions) >= self.config.get('max_positions', 1):
            return
        
        # Calculate entry price
        entry_price = bar['close']
        
        # Calculate stop loss and take profit
        stop_loss = self.strategy.calculate_stop_loss(
            'symbol',  # Symbol is not used in backtest
            'BUY' if signal == 1 else 'SELL',
            entry_price
        )
        
        take_profit = self.strategy.calculate_take_profit(
            'symbol',  # Symbol is not used in backtest
            'BUY' if signal == 1 else 'SELL',
            entry_price,
            stop_loss
        )
        
        # Calculate position size
        position_size = self.strategy.calculate_position_size(
            'symbol',  # Symbol is not used in backtest
            'BUY' if signal == 1 else 'SELL',
            entry_price,
            stop_loss,
            self.current_balance
        )
        
        # Add position
        position = {
            'entry_time': bar.name,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'type': 'BUY' if signal == 1 else 'SELL'
        }
        
        self.positions.append(position)

    def _update_positions(self, bar: pd.Series):
        """Update existing positions and check for exits."""
        for position in self.positions[:]:  # Copy list to allow removal
            # Check for stop loss
            if position['type'] == 'BUY':
                if bar['low'] <= position['stop_loss']:
                    self._close_position(position, position['stop_loss'], bar.name, 'Stop Loss')
                    continue
                if bar['high'] >= position['take_profit']:
                    self._close_position(position, position['take_profit'], bar.name, 'Take Profit')
                    continue
            else:  # SELL
                if bar['high'] >= position['stop_loss']:
                    self._close_position(position, position['stop_loss'], bar.name, 'Stop Loss')
                    continue
                if bar['low'] <= position['take_profit']:
                    self._close_position(position, position['take_profit'], bar.name, 'Take Profit')
                    continue

    def _close_position(self, position: Dict, exit_price: float, exit_time: datetime, reason: str):
        """Close a position and record the trade."""
        # Calculate profit/loss
        if position['type'] == 'BUY':
            pnl = (exit_price - position['entry_price']) * position['position_size']
        else:  # SELL
            pnl = (position['entry_price'] - exit_price) * position['position_size']
        
        # Apply commission
        pnl -= self.config['commission'] * position['position_size']
        
        # Update balance
        self.current_balance += pnl
        
        # Record trade
        trade = {
            'entry_time': position['entry_time'],
            'exit_time': exit_time,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'position_size': position['position_size'],
            'type': position['type'],
            'pnl': pnl,
            'reason': reason
        }
        
        self.trades.append(trade)
        self.positions.remove(position)

    def _update_equity_curve(self, bar: pd.Series):
        """Update equity curve with current balance and unrealized P&L."""
        unrealized_pnl = 0
        for position in self.positions:
            if position['type'] == 'BUY':
                unrealized_pnl += (bar['close'] - position['entry_price']) * position['position_size']
            else:  # SELL
                unrealized_pnl += (position['entry_price'] - bar['close']) * position['position_size']
        
        self.equity_curve.append({
            'time': bar.name,
            'equity': self.current_balance + unrealized_pnl
        })

    def _calculate_metrics(self):
        """Calculate performance metrics."""
        if not self.trades:
            return
        
        # Basic metrics
        self.metrics['total_trades'] = len(self.trades)
        self.metrics['winning_trades'] = len([t for t in self.trades if t['pnl'] > 0])
        self.metrics['losing_trades'] = len([t for t in self.trades if t['pnl'] <= 0])
        
        # Profit metrics
        self.metrics['total_profit'] = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
        self.metrics['total_loss'] = abs(sum(t['pnl'] for t in self.trades if t['pnl'] <= 0))
        
        # Win rate
        self.metrics['win_rate'] = self.metrics['winning_trades'] / self.metrics['total_trades']
        
        # Profit factor
        if self.metrics['total_loss'] > 0:
            self.metrics['profit_factor'] = self.metrics['total_profit'] / self.metrics['total_loss']
        
        # Average win/loss
        if self.metrics['winning_trades'] > 0:
            self.metrics['average_win'] = self.metrics['total_profit'] / self.metrics['winning_trades']
        if self.metrics['losing_trades'] > 0:
            self.metrics['average_loss'] = self.metrics['total_loss'] / self.metrics['losing_trades']
        
        # Risk/reward ratio
        if self.metrics['average_loss'] > 0:
            self.metrics['risk_reward_ratio'] = self.metrics['average_win'] / self.metrics['average_loss']
        
        # Calculate drawdown
        equity_series = pd.Series([e['equity'] for e in self.equity_curve])
        rolling_max = equity_series.expanding().max()
        drawdowns = (equity_series - rolling_max) / rolling_max
        self.metrics['max_drawdown'] = abs(drawdowns.min())
        
        # Calculate Sharpe ratio
        returns = pd.Series([t['pnl'] for t in self.trades])
        if len(returns) > 1:
            self.metrics['sharpe_ratio'] = np.sqrt(252) * returns.mean() / returns.std()
        
        # Calculate Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1:
            self.metrics['sortino_ratio'] = np.sqrt(252) * returns.mean() / downside_returns.std()

    def plot_results(self, save_path: Optional[str] = None):
        """Plot backtest results."""
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Plot equity curve
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('time', inplace=True)
        equity_df['equity'].plot(ax=ax1, label='Equity')
        ax1.set_title('Equity Curve')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Equity')
        ax1.grid(True)
        
        # Plot drawdown
        rolling_max = equity_df['equity'].expanding().max()
        drawdown = (equity_df['equity'] - rolling_max) / rolling_max
        drawdown.plot(ax=ax2, label='Drawdown', color='red')
        ax2.set_title('Drawdown')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Drawdown')
        ax2.grid(True)
        
        # Plot trade distribution
        trade_pnls = [t['pnl'] for t in self.trades]
        sns.histplot(trade_pnls, ax=ax3, bins=50)
        ax3.set_title('Trade P&L Distribution')
        ax3.set_xlabel('P&L')
        ax3.set_ylabel('Frequency')
        ax3.grid(True)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as a DataFrame."""
        return pd.DataFrame(self.trades)

    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as a DataFrame."""
        return pd.DataFrame(self.equity_curve) 