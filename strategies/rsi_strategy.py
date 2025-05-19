import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from .base_strategy import BaseStrategy
import ta
from scipy import stats
from config.config import RISK_PER_TRADE

class RSIStrategy(BaseStrategy):
    def __init__(self, params: Dict = None):
        default_params = {
            'rsi_period': 14,
            'overbought': 70,
            'oversold': 30,
            'rsi_smooth': 3,
            'volume_threshold': 1.5,
            'trend_period': 20,
            'confirmation_period': 5,
            'divergence_lookback': 10,
            'volatility_threshold': 0.002,
            'min_volume': 1000,
            'max_spread': 0.0002
        }
        super().__init__('RSI', params or default_params)
        self.required_params = [
            'rsi_period',
            'overbought',
            'oversold',
            'rsi_smooth',
            'volume_threshold',
            'trend_period',
            'confirmation_period',
            'divergence_lookback',
            'volatility_threshold',
            'min_volume',
            'max_spread'
        ]

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI and additional indicators."""
        # Calculate RSI
        data['rsi'] = ta.momentum.RSIIndicator(
            close=data['close'],
            window=self.params['rsi_period']
        ).rsi()
        
        # Smooth RSI
        data['rsi_smooth'] = data['rsi'].rolling(
            window=self.params['rsi_smooth']
        ).mean()
        
        # Calculate trend
        data['trend'] = self._calculate_trend(data)
        
        # Calculate volume indicators
        data['volume_ma'] = data['volume'].rolling(
            window=self.params['trend_period']
        ).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma']
        
        # Calculate volatility
        data['volatility'] = data['close'].pct_change().rolling(
            window=self.params['trend_period']
        ).std()
        
        # Calculate price momentum
        data['momentum'] = data['close'].pct_change(
            periods=self.params['confirmation_period']
        )
        
        # Calculate RSI divergence
        data['bullish_divergence'] = self._calculate_bullish_divergence(data)
        data['bearish_divergence'] = self._calculate_bearish_divergence(data)
        
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals with enhanced filtering."""
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Basic RSI signals
        signals.loc[data['rsi_smooth'] < self.params['oversold'], 'signal'] = 1
        signals.loc[data['rsi_smooth'] > self.params['overbought'], 'signal'] = -1
        
        # Apply trend filter
        signals.loc[data['trend'] < 0, 'signal'] = 0  # No long positions in downtrend
        signals.loc[data['trend'] > 0, 'signal'] = 0  # No short positions in uptrend
        
        # Apply volume filter
        signals.loc[data['volume_ratio'] < self.params['volume_threshold'], 'signal'] = 0
        
        # Apply volatility filter
        signals.loc[data['volatility'] > self.params['volatility_threshold'], 'signal'] = 0
        
        # Apply divergence signals
        signals.loc[data['bullish_divergence'], 'signal'] = 1
        signals.loc[data['bearish_divergence'], 'signal'] = -1
        
        # Apply spread filter
        signals.loc[data['spread'] > self.params['max_spread'], 'signal'] = 0
        
        # Apply minimum volume filter
        signals.loc[data['volume'] < self.params['min_volume'], 'signal'] = 0
        
        # Calculate signal strength
        signals['strength'] = self._calculate_signal_strength(data, signals)
        
        # Apply strength threshold
        signals.loc[signals['strength'] < 0.6, 'signal'] = 0
        
        return signals

    def calculate_position_size(
        self,
        symbol: str,
        signal: str,
        price: float,
        stop_loss: float,
        account_balance: float
    ) -> float:
        """Calculate position size with enhanced risk management."""
        # Calculate base position size
        risk_amount = account_balance * 0.01  # 1% risk per trade
        price_diff = abs(price - stop_loss)
        
        if price_diff == 0:
            return 0.0
        
        position_size = risk_amount / price_diff
        
        # Adjust for signal strength
        signal_strength = self._get_signal_strength(symbol)
        position_size *= signal_strength
        
        # Adjust for market volatility
        volatility_factor = self._get_volatility_factor(symbol)
        position_size *= volatility_factor
        
        # Apply position limits
        position_size = min(position_size, account_balance * 0.1)  # Max 10% of account
        position_size = max(position_size, 0.01)  # Min 0.01 lots
        
        return round(position_size, 2)

    def calculate_stop_loss(
        self,
        symbol: str,
        signal: str,
        price: float
    ) -> float:
        """Calculate stop loss with enhanced risk management."""
        # Calculate ATR-based stop loss
        atr = self._calculate_atr(symbol)
        
        if signal == 'BUY':
            stop_loss = price - (atr * 2)
        else:
            stop_loss = price + (atr * 2)
        
        # Adjust for volatility
        volatility = self._get_volatility(symbol)
        stop_loss = self._adjust_stop_loss_for_volatility(
            stop_loss,
            price,
            volatility
        )
        
        return round(stop_loss, 5)

    def calculate_take_profit(
        self,
        symbol: str,
        signal: str,
        price: float,
        stop_loss: float
    ) -> float:
        """Calculate take profit with enhanced risk management."""
        # Calculate risk-reward based take profit
        risk = abs(price - stop_loss)
        reward = risk * 2  # 1:2 risk-reward ratio
        
        if signal == 'BUY':
            take_profit = price + reward
        else:
            take_profit = price - reward
        
        # Adjust for market conditions
        take_profit = self._adjust_take_profit_for_market_conditions(
            take_profit,
            price,
            signal
        )
        
        return round(take_profit, 5)

    def _calculate_trend(self, data: pd.DataFrame) -> pd.Series:
        """Calculate trend using multiple indicators."""
        # Calculate moving averages
        ma_fast = data['close'].rolling(window=10).mean()
        ma_slow = data['close'].rolling(window=20).mean()
        
        # Calculate trend strength
        trend = (ma_fast - ma_slow) / ma_slow
        
        # Normalize trend
        trend = (trend - trend.min()) / (trend.max() - trend.min())
        
        return trend

    def _calculate_bullish_divergence(self, data: pd.DataFrame) -> pd.Series:
        """Calculate bullish RSI divergence."""
        divergence = pd.Series(False, index=data.index)
        
        for i in range(self.params['divergence_lookback'], len(data)):
            # Find local minimum in price
            if data['low'].iloc[i] < data['low'].iloc[i-1] and \
               data['low'].iloc[i] < data['low'].iloc[i+1]:
                # Find corresponding RSI value
                rsi_value = data['rsi'].iloc[i]
                
                # Look for higher RSI value in previous period
                for j in range(i-self.params['divergence_lookback'], i):
                    if data['low'].iloc[j] < data['low'].iloc[i] and \
                       data['rsi'].iloc[j] > rsi_value:
                        divergence.iloc[i] = True
                        break
        
        return divergence

    def _calculate_bearish_divergence(self, data: pd.DataFrame) -> pd.Series:
        """Calculate bearish RSI divergence."""
        divergence = pd.Series(False, index=data.index)
        
        for i in range(self.params['divergence_lookback'], len(data)):
            # Find local maximum in price
            if data['high'].iloc[i] > data['high'].iloc[i-1] and \
               data['high'].iloc[i] > data['high'].iloc[i+1]:
                # Find corresponding RSI value
                rsi_value = data['rsi'].iloc[i]
                
                # Look for lower RSI value in previous period
                for j in range(i-self.params['divergence_lookback'], i):
                    if data['high'].iloc[j] > data['high'].iloc[i] and \
                       data['rsi'].iloc[j] < rsi_value:
                        divergence.iloc[i] = True
                        break
        
        return divergence

    def _calculate_signal_strength(self, data: pd.DataFrame, signals: pd.DataFrame) -> pd.Series:
        """Calculate signal strength based on multiple factors."""
        strength = pd.Series(0.0, index=data.index)
        
        # RSI strength
        rsi_strength = 1 - abs(data['rsi_smooth'] - 50) / 50
        
        # Volume strength
        volume_strength = data['volume_ratio'].clip(0, 2) / 2
        
        # Trend strength
        trend_strength = abs(data['trend'])
        
        # Divergence strength
        divergence_strength = (data['bullish_divergence'] | data['bearish_divergence']).astype(float)
        
        # Combine factors
        strength = (
            rsi_strength * 0.3 +
            volume_strength * 0.2 +
            trend_strength * 0.2 +
            divergence_strength * 0.3
        )
        
        return strength

    def _get_signal_strength(self, symbol: str) -> float:
        """Get current signal strength for a symbol."""
        # This should be implemented to get actual signal strength
        return 0.8

    def _get_volatility_factor(self, symbol: str) -> float:
        """Get volatility adjustment factor."""
        # This should be implemented to get actual volatility factor
        return 0.8

    def _get_volatility(self, symbol: str) -> float:
        """Get current volatility for a symbol."""
        # This should be implemented to get actual volatility
        return 0.002

    def _calculate_atr(self, symbol: str) -> float:
        """Calculate Average True Range."""
        # This should be implemented to get actual ATR
        return 0.001

    def _adjust_stop_loss_for_volatility(
        self,
        stop_loss: float,
        price: float,
        volatility: float
    ) -> float:
        """Adjust stop loss based on volatility."""
        # Implement volatility-based stop loss adjustment
        return stop_loss

    def _adjust_take_profit_for_market_conditions(
        self,
        take_profit: float,
        price: float,
        signal: str
    ) -> float:
        """Adjust take profit based on market conditions."""
        # Implement market condition-based take profit adjustment
        return take_profit

    def get_required_parameters(self) -> List[str]:
        """Get list of required parameters."""
        return self.required_params

    def validate_parameters(self) -> bool:
        """Validate RSI strategy parameters."""
        required_params = ['rsi_period', 'overbought', 'oversold', 'rsi_smooth', 'volume_threshold', 'trend_period', 'confirmation_period', 'divergence_lookback', 'volatility_threshold', 'min_volume', 'max_spread']
        
        # Check if all required parameters are present
        if not all(param in self.params for param in required_params):
            self.logger.error("Missing required parameters")
            return False

        # Validate parameter values
        if not (0 < self.params['rsi_period'] <= 100):
            self.logger.error("Invalid RSI period")
            return False

        if not (0 < self.params['oversold'] < self.params['overbought'] < 100):
            self.logger.error("Invalid RSI levels")
            return False

        if not (0 < self.params['rsi_smooth'] <= 100):
            self.logger.error("Invalid RSI smoothing period")
            return False

        if not (0 < self.params['volume_threshold'] <= 10):
            self.logger.error("Invalid volume threshold")
            return False

        if not (0 < self.params['trend_period'] <= 100):
            self.logger.error("Invalid trend period")
            return False

        if not (0 < self.params['confirmation_period'] <= 100):
            self.logger.error("Invalid confirmation period")
            return False

        if not (0 < self.params['divergence_lookback'] <= 100):
            self.logger.error("Invalid divergence lookback period")
            return False

        if not (0 < self.params['volatility_threshold'] <= 0.01):
            self.logger.error("Invalid volatility threshold")
            return False

        if not (0 < self.params['min_volume'] <= 1000000):
            self.logger.error("Invalid minimum volume")
            return False

        if not (0 < self.params['max_spread'] <= 0.01):
            self.logger.error("Invalid maximum spread")
            return False

        return True 