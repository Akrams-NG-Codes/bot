from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

class BaseStrategy(ABC):
    def __init__(self, name: str, params: Dict):
        self.name = name
        self.params = params
        self.logger = logging.getLogger(f"strategy.{name}")
        self.performance_metrics = {
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
            'risk_reward_ratio': 0.0
        }

    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the strategy."""
        pass

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on indicators."""
        pass

    @abstractmethod
    def calculate_position_size(
        self,
        symbol: str,
        signal: str,
        price: float,
        stop_loss: float,
        account_balance: float
    ) -> float:
        """Calculate position size based on risk management rules."""
        pass

    @abstractmethod
    def calculate_stop_loss(
        self,
        symbol: str,
        signal: str,
        price: float
    ) -> float:
        """Calculate stop loss level for the trade."""
        pass

    @abstractmethod
    def calculate_take_profit(
        self,
        symbol: str,
        signal: str,
        price: float,
        stop_loss: float
    ) -> float:
        """Calculate take profit level for the trade."""
        pass

    def update_performance_metrics(self, trade_result: Dict) -> None:
        """Update strategy performance metrics after each trade."""
        self.performance_metrics['total_trades'] += 1
        
        if trade_result['profit'] > 0:
            self.performance_metrics['winning_trades'] += 1
            self.performance_metrics['total_profit'] += trade_result['profit']
        else:
            self.performance_metrics['losing_trades'] += 1
            self.performance_metrics['total_loss'] += abs(trade_result['profit'])

        # Update win rate
        self.performance_metrics['win_rate'] = (
            self.performance_metrics['winning_trades'] / 
            self.performance_metrics['total_trades']
        )

        # Update profit factor
        if self.performance_metrics['total_loss'] > 0:
            self.performance_metrics['profit_factor'] = (
                self.performance_metrics['total_profit'] / 
                self.performance_metrics['total_loss']
            )

        # Update average win/loss
        if self.performance_metrics['winning_trades'] > 0:
            self.performance_metrics['average_win'] = (
                self.performance_metrics['total_profit'] / 
                self.performance_metrics['winning_trades']
            )
        
        if self.performance_metrics['losing_trades'] > 0:
            self.performance_metrics['average_loss'] = (
                self.performance_metrics['total_loss'] / 
                self.performance_metrics['losing_trades']
            )

        # Update risk/reward ratio
        if self.performance_metrics['average_loss'] > 0:
            self.performance_metrics['risk_reward_ratio'] = (
                self.performance_metrics['average_win'] / 
                self.performance_metrics['average_loss']
            )

        # Update max drawdown
        current_drawdown = trade_result.get('drawdown', 0)
        self.performance_metrics['max_drawdown'] = max(
            self.performance_metrics['max_drawdown'],
            current_drawdown
        )

    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics."""
        return self.performance_metrics

    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        return True

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess market data before analysis."""
        # Remove any missing values
        data = data.dropna()
        
        # Ensure data is sorted by time
        if 'time' in data.columns:
            data = data.sort_values('time')
        
        return data

    def postprocess_signals(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Postprocess trading signals."""
        # Remove duplicate signals
        signals = signals.drop_duplicates(subset=['time'])
        
        # Ensure signals are properly formatted
        if 'signal' in signals.columns:
            signals['signal'] = signals['signal'].map({
                1: 'BUY',
                -1: 'SELL',
                0: 'NEUTRAL'
            })
        
        return signals

    def get_strategy_info(self) -> Dict:
        """Get strategy information and parameters."""
        return {
            'name': self.name,
            'parameters': self.params,
            'performance': self.performance_metrics
        }

    def reset_performance_metrics(self) -> None:
        """Reset all performance metrics to initial values."""
        self.performance_metrics = {
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
            'risk_reward_ratio': 0.0
        } 