import pandas as pd
import numpy as np
from typing import Dict
from .base_strategy import BaseStrategy
import ta
from config.config import RISK_PER_TRADE

class MACDStrategy(BaseStrategy):
    def __init__(self, params: Dict = None):
        default_params = {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
            'risk_reward_ratio': 2.0,
            'atr_period': 14,
            'atr_multiplier': 2.0
        }
        params = params or default_params
        super().__init__('MACD', params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicators."""
        # Calculate MACD
        macd = ta.trend.MACD(
            close=data['close'],
            window_slow=self.params['slow_period'],
            window_fast=self.params['fast_period'],
            window_sign=self.params['signal_period']
        )
        
        data['macd'] = macd.macd()
        data['macd_signal'] = macd.macd_signal()
        data['macd_histogram'] = macd.macd_diff()

        # Calculate ATR for stop loss and take profit
        data['atr'] = ta.volatility.AverageTrueRange(
            high=data['high'],
            low=data['low'],
            close=data['close'],
            window=self.params['atr_period']
        ).average_true_range()

        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on MACD crossovers."""
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0

        # Generate buy signals when MACD crosses above signal line
        buy_condition = (
            (data['macd'] > data['macd_signal']) & 
            (data['macd'].shift(1) <= data['macd_signal'].shift(1))
        )
        signals.loc[buy_condition, 'signal'] = 1

        # Generate sell signals when MACD crosses below signal line
        sell_condition = (
            (data['macd'] < data['macd_signal']) & 
            (data['macd'].shift(1) >= data['macd_signal'].shift(1))
        )
        signals.loc[sell_condition, 'signal'] = -1

        return self.postprocess_signals(signals)

    def calculate_position_size(
        self,
        symbol: str,
        signal: str,
        price: float,
        stop_loss: float,
        account_balance: float
    ) -> float:
        """Calculate position size based on risk management rules."""
        # Calculate risk amount in account currency
        risk_amount = account_balance * RISK_PER_TRADE

        # Calculate price difference between entry and stop loss
        price_diff = abs(price - stop_loss)

        if price_diff == 0:
            return 0.0

        # Calculate position size
        position_size = risk_amount / price_diff

        # Round position size to appropriate lot size
        position_size = round(position_size, 2)

        return position_size

    def calculate_stop_loss(
        self,
        symbol: str,
        signal: str,
        price: float
    ) -> float:
        """Calculate stop loss level using ATR."""
        # Get the latest ATR value
        atr = self.params.get('atr', 0.001)  # Default value if ATR not available
        
        # Calculate stop loss based on signal type
        if signal == 'BUY':
            stop_loss = price - (atr * self.params['atr_multiplier'])
        else:  # SELL
            stop_loss = price + (atr * self.params['atr_multiplier'])

        return round(stop_loss, 5)

    def calculate_take_profit(
        self,
        symbol: str,
        signal: str,
        price: float,
        stop_loss: float
    ) -> float:
        """Calculate take profit level based on risk-reward ratio."""
        # Calculate the distance to stop loss
        sl_distance = abs(price - stop_loss)

        # Calculate take profit based on risk-reward ratio
        if signal == 'BUY':
            take_profit = price + (sl_distance * self.params['risk_reward_ratio'])
        else:  # SELL
            take_profit = price - (sl_distance * self.params['risk_reward_ratio'])

        return round(take_profit, 5)

    def validate_parameters(self) -> bool:
        """Validate MACD strategy parameters."""
        required_params = ['fast_period', 'slow_period', 'signal_period', 'risk_reward_ratio']
        
        # Check if all required parameters are present
        if not all(param in self.params for param in required_params):
            self.logger.error("Missing required parameters")
            return False

        # Validate parameter values
        if not (0 < self.params['fast_period'] < self.params['slow_period']):
            self.logger.error("Invalid MACD periods")
            return False

        if not (0 < self.params['signal_period'] <= self.params['fast_period']):
            self.logger.error("Invalid signal period")
            return False

        if not (0 < self.params['risk_reward_ratio'] <= 10):
            self.logger.error("Invalid risk-reward ratio")
            return False

        return True 