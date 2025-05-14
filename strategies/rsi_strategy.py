import pandas as pd
import numpy as np
from typing import Dict
from .base_strategy import BaseStrategy
import ta
from config.config import RISK_PER_TRADE

class RSIStrategy(BaseStrategy):
    def __init__(self, params: Dict = None):
        default_params = {
            'period': 14,
            'overbought': 70,
            'oversold': 30,
            'risk_reward_ratio': 2.0,
            'atr_period': 14,
            'atr_multiplier': 2.0
        }
        params = params or default_params
        super().__init__('RSI', params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI and ATR indicators."""
        # Calculate RSI
        data['rsi'] = ta.momentum.RSIIndicator(
            close=data['close'],
            window=self.params['period']
        ).rsi()

        # Calculate ATR for stop loss and take profit
        data['atr'] = ta.volatility.AverageTrueRange(
            high=data['high'],
            low=data['low'],
            close=data['close'],
            window=self.params['atr_period']
        ).average_true_range()

        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on RSI values."""
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0

        # Generate buy signals when RSI crosses below oversold
        buy_condition = (
            (data['rsi'] < self.params['oversold']) & 
            (data['rsi'].shift(1) >= self.params['oversold'])
        )
        signals.loc[buy_condition, 'signal'] = 1

        # Generate sell signals when RSI crosses above overbought
        sell_condition = (
            (data['rsi'] > self.params['overbought']) & 
            (data['rsi'].shift(1) <= self.params['overbought'])
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
        """Validate RSI strategy parameters."""
        required_params = ['period', 'overbought', 'oversold', 'risk_reward_ratio']
        
        # Check if all required parameters are present
        if not all(param in self.params for param in required_params):
            self.logger.error("Missing required parameters")
            return False

        # Validate parameter values
        if not (0 < self.params['period'] <= 100):
            self.logger.error("Invalid RSI period")
            return False

        if not (0 < self.params['oversold'] < self.params['overbought'] < 100):
            self.logger.error("Invalid RSI levels")
            return False

        if not (0 < self.params['risk_reward_ratio'] <= 10):
            self.logger.error("Invalid risk-reward ratio")
            return False

        return True 