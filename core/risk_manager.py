import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from config.config import (
    MAX_DRAWDOWN,
    MAX_DAILY_TRADES,
    RISK_PER_TRADE,
    MAX_OPEN_POSITIONS
)

class RiskManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.daily_trades = 0
        self.last_trade_reset = datetime.now()
        self.trade_history = []
        self.current_drawdown = 0.0
        self.peak_balance = 0.0
        self.open_positions = []
        self.volatility_limits = {}  # Store volatility limits per symbol
        self.correlation_matrix = pd.DataFrame()  # Store correlation between symbols
        self.market_regime = 'normal'  # Can be 'normal', 'volatile', 'trending'
        self.risk_multiplier = 1.0  # Adjust risk based on market conditions

    def can_open_position(
        self,
        symbol: str,
        strategy: str,
        account_balance: float,
        current_equity: float,
        market_data: pd.DataFrame
    ) -> Tuple[bool, str]:
        """Enhanced check for opening new positions with detailed reasoning."""
        # Reset daily trade counter if it's a new day
        self._reset_daily_trades_if_needed()
        
        # Basic checks
        if self.daily_trades >= MAX_DAILY_TRADES:
            return False, "Daily trade limit reached"
        
        if len(self.open_positions) >= MAX_OPEN_POSITIONS:
            return False, "Maximum number of open positions reached"
        
        if any(pos['symbol'] == symbol for pos in self.open_positions):
            return False, f"Already have an open position for {symbol}"
        
        # Update peak balance and calculate drawdown
        self.peak_balance = max(self.peak_balance, current_equity)
        self.current_drawdown = (self.peak_balance - current_equity) / self.peak_balance
        
        if self.current_drawdown > MAX_DRAWDOWN:
            return False, f"Maximum drawdown exceeded: {self.current_drawdown:.2%}"
        
        # Advanced checks
        if not self._check_volatility(symbol, market_data):
            return False, "Current volatility exceeds limits"
        
        if not self._check_correlation(symbol):
            return False, "High correlation with existing positions"
        
        if not self._check_market_regime(symbol, market_data):
            return False, "Unfavorable market regime"
        
        return True, "Position can be opened"

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        account_balance: float,
        market_data: pd.DataFrame,
        risk_per_trade: float = RISK_PER_TRADE
    ) -> float:
        """Enhanced position sizing with dynamic risk adjustment."""
        # Calculate base risk amount
        base_risk_amount = account_balance * risk_per_trade
        
        # Adjust risk based on market regime
        adjusted_risk = base_risk_amount * self.risk_multiplier
        
        # Adjust for volatility
        volatility_factor = self._calculate_volatility_factor(symbol, market_data)
        adjusted_risk *= volatility_factor
        
        # Calculate price difference between entry and stop loss
        price_diff = abs(entry_price - stop_loss)
        
        if price_diff == 0:
            return 0.0
        
        # Calculate position size
        position_size = adjusted_risk / price_diff
        
        # Apply position size limits
        position_size = self._apply_position_limits(position_size, symbol)
        
        return round(position_size, 2)

    def calculate_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        market_data: pd.DataFrame,
        atr_multiplier: float = 2.0
    ) -> float:
        """Enhanced stop loss calculation using multiple methods."""
        # Calculate ATR
        atr = self._calculate_atr(market_data)
        
        # Calculate volatility-based stop
        volatility_stop = self._calculate_volatility_stop(symbol, market_data)
        
        # Calculate support/resistance levels
        support_level = self._find_nearest_support(entry_price, market_data)
        
        # Use the most conservative stop loss
        stops = [
            entry_price - (atr * atr_multiplier),  # ATR-based
            volatility_stop,  # Volatility-based
            support_level  # Support-based
        ]
        
        return round(max(stops), 5)

    def calculate_take_profit(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        market_data: pd.DataFrame,
        risk_reward_ratio: float = 2.0
    ) -> float:
        """Enhanced take profit calculation using multiple methods."""
        # Calculate base take profit
        sl_distance = abs(entry_price - stop_loss)
        base_tp = entry_price + (sl_distance * risk_reward_ratio)
        
        # Calculate resistance-based take profit
        resistance_level = self._find_nearest_resistance(entry_price, market_data)
        
        # Calculate volatility-based take profit
        volatility_tp = self._calculate_volatility_tp(symbol, market_data, entry_price)
        
        # Use the most conservative take profit
        take_profits = [
            base_tp,
            resistance_level,
            volatility_tp
        ]
        
        return round(min(take_profits), 5)

    def update_market_conditions(self, market_data: Dict[str, pd.DataFrame]) -> None:
        """Update market conditions and adjust risk parameters."""
        self._update_volatility_limits(market_data)
        self._update_correlation_matrix(market_data)
        self._update_market_regime(market_data)
        self._adjust_risk_multiplier()

    def _check_volatility(self, symbol: str, market_data: pd.DataFrame) -> bool:
        """Check if current volatility is within acceptable limits."""
        current_volatility = self._calculate_volatility(symbol, market_data)
        return current_volatility <= self.volatility_limits.get(symbol, float('inf'))

    def _check_correlation(self, symbol: str) -> bool:
        """Check correlation with existing positions."""
        if self.correlation_matrix.empty:
            return True
        
        for position in self.open_positions:
            if self.correlation_matrix.loc[symbol, position['symbol']] > 0.7:
                return False
        return True

    def _check_market_regime(self, symbol: str, market_data: pd.DataFrame) -> bool:
        """Check if current market regime is suitable for trading."""
        if self.market_regime == 'volatile':
            return False  # Don't trade in highly volatile markets
        return True

    def _calculate_volatility_factor(self, symbol: str, market_data: pd.DataFrame) -> float:
        """Calculate volatility adjustment factor for position sizing."""
        current_volatility = self._calculate_volatility(symbol, market_data)
        historical_volatility = self._calculate_historical_volatility(symbol, market_data)
        
        if historical_volatility == 0:
            return 1.0
        
        ratio = current_volatility / historical_volatility
        return max(0.5, min(1.0, 1.0 / ratio))

    def _apply_position_limits(self, position_size: float, symbol: str) -> float:
        """Apply position size limits based on symbol and account size."""
        # Get symbol info
        symbol_info = self._get_symbol_info(symbol)
        
        # Apply minimum and maximum position size limits
        position_size = max(symbol_info['min_lot'], position_size)
        position_size = min(symbol_info['max_lot'], position_size)
        
        return position_size

    def _calculate_atr(self, market_data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        high = market_data['high']
        low = market_data['low']
        close = market_data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean().iloc[-1]

    def _calculate_volatility(self, symbol: str, market_data: pd.DataFrame) -> float:
        """Calculate current volatility."""
        returns = market_data['close'].pct_change()
        return returns.std() * np.sqrt(252)  # Annualized volatility

    def _calculate_historical_volatility(self, symbol: str, market_data: pd.DataFrame) -> float:
        """Calculate historical volatility."""
        returns = market_data['close'].pct_change()
        return returns.std() * np.sqrt(252)  # Annualized volatility

    def _find_nearest_support(self, price: float, market_data: pd.DataFrame) -> float:
        """Find nearest support level."""
        # Simple implementation - can be enhanced with more sophisticated methods
        return market_data['low'].rolling(20).min().iloc[-1]

    def _find_nearest_resistance(self, price: float, market_data: pd.DataFrame) -> float:
        """Find nearest resistance level."""
        # Simple implementation - can be enhanced with more sophisticated methods
        return market_data['high'].rolling(20).max().iloc[-1]

    def _update_volatility_limits(self, market_data: Dict[str, pd.DataFrame]) -> None:
        """Update volatility limits for each symbol."""
        for symbol, data in market_data.items():
            self.volatility_limits[symbol] = self._calculate_historical_volatility(symbol, data) * 1.5

    def _update_correlation_matrix(self, market_data: Dict[str, pd.DataFrame]) -> None:
        """Update correlation matrix between symbols."""
        returns = pd.DataFrame()
        for symbol, data in market_data.items():
            returns[symbol] = data['close'].pct_change()
        
        self.correlation_matrix = returns.corr()

    def _update_market_regime(self, market_data: Dict[str, pd.DataFrame]) -> None:
        """Update market regime based on volatility and trend."""
        # Simple implementation - can be enhanced with more sophisticated methods
        volatility = self._calculate_volatility('EURUSD', market_data['EURUSD'])
        if volatility > 0.2:  # 20% annualized volatility
            self.market_regime = 'volatile'
        else:
            self.market_regime = 'normal'

    def _adjust_risk_multiplier(self) -> None:
        """Adjust risk multiplier based on market conditions."""
        if self.market_regime == 'volatile':
            self.risk_multiplier = 0.5
        else:
            self.risk_multiplier = 1.0

    def _get_symbol_info(self, symbol: str) -> Dict:
        """Get symbol information including lot size limits."""
        # This should be implemented to get actual symbol information from MT5
        return {
            'min_lot': 0.01,
            'max_lot': 100.0,
            'point': 0.00001,
            'digits': 5
        }

    def update_position(
        self,
        position_id: int,
        current_price: float,
        current_equity: float
    ) -> Dict:
        """Update position information and check for risk management rules."""
        # Find the position
        position = next(
            (pos for pos in self.open_positions if pos['id'] == position_id),
            None
        )
        
        if position is None:
            return {}
        
        # Update position information
        position['current_price'] = current_price
        position['unrealized_pnl'] = self._calculate_unrealized_pnl(position)
        
        # Check for trailing stop loss
        if position.get('trailing_stop', False):
            self._update_trailing_stop(position)
        
        # Update drawdown
        self.peak_balance = max(self.peak_balance, current_equity)
        self.current_drawdown = (self.peak_balance - current_equity) / self.peak_balance
        
        return position

    def add_position(
        self,
        position_id: int,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        position_size: float,
        strategy: str
    ) -> None:
        """Add a new position to the risk manager."""
        position = {
            'id': position_id,
            'symbol': symbol,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'strategy': strategy,
            'entry_time': datetime.now(),
            'trailing_stop': False,
            'trailing_stop_price': None
        }
        
        self.open_positions.append(position)
        self.daily_trades += 1

    def remove_position(self, position_id: int) -> None:
        """Remove a position from the risk manager."""
        self.open_positions = [
            pos for pos in self.open_positions if pos['id'] != position_id
        ]

    def get_position_risk(self, position_id: int) -> Dict:
        """Get risk metrics for a specific position."""
        position = next(
            (pos for pos in self.open_positions if pos['id'] == position_id),
            None
        )
        
        if position is None:
            return {}
        
        return {
            'position_size': position['position_size'],
            'risk_amount': abs(position['entry_price'] - position['stop_loss']) * position['position_size'],
            'risk_reward_ratio': abs(position['take_profit'] - position['entry_price']) / abs(position['entry_price'] - position['stop_loss']),
            'unrealized_pnl': position.get('unrealized_pnl', 0)
        }

    def get_portfolio_risk(self) -> Dict:
        """Get overall portfolio risk metrics."""
        total_risk = sum(
            abs(pos['entry_price'] - pos['stop_loss']) * pos['position_size']
            for pos in self.open_positions
        )
        
        total_exposure = sum(
            pos['position_size'] * pos['entry_price']
            for pos in self.open_positions
        )
        
        return {
            'total_positions': len(self.open_positions),
            'total_risk': total_risk,
            'total_exposure': total_exposure,
            'current_drawdown': self.current_drawdown,
            'daily_trades': self.daily_trades
        }

    def _reset_daily_trades_if_needed(self) -> None:
        """Reset daily trade counter if it's a new day."""
        if datetime.now().date() > self.last_trade_reset.date():
            self.daily_trades = 0
            self.last_trade_reset = datetime.now()

    def _calculate_unrealized_pnl(self, position: Dict) -> float:
        """Calculate unrealized P&L for a position."""
        price_diff = position['current_price'] - position['entry_price']
        return price_diff * position['position_size']

    def _update_trailing_stop(self, position: Dict) -> None:
        """Update trailing stop loss for a position."""
        if position['trailing_stop_price'] is None:
            position['trailing_stop_price'] = position['stop_loss']
        
        # Update trailing stop if price moved in favorable direction
        if position['current_price'] > position['entry_price']:
            new_stop = position['current_price'] - (position['entry_price'] - position['stop_loss'])
            if new_stop > position['trailing_stop_price']:
                position['trailing_stop_price'] = new_stop
                position['stop_loss'] = new_stop 