import numpy as np
import pandas as pd
from typing import Dict, List, Optional
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

    def can_open_position(
        self,
        symbol: str,
        strategy: str,
        account_balance: float,
        current_equity: float
    ) -> bool:
        """Check if a new position can be opened based on risk management rules."""
        # Reset daily trade counter if it's a new day
        self._reset_daily_trades_if_needed()
        
        # Check if we've reached the daily trade limit
        if self.daily_trades >= MAX_DAILY_TRADES:
            self.logger.warning("Daily trade limit reached")
            return False
        
        # Check if we've reached the maximum number of open positions
        if len(self.open_positions) >= MAX_OPEN_POSITIONS:
            self.logger.warning("Maximum number of open positions reached")
            return False
        
        # Check if we're already trading this symbol
        if any(pos['symbol'] == symbol for pos in self.open_positions):
            self.logger.warning(f"Already have an open position for {symbol}")
            return False
        
        # Update peak balance and calculate drawdown
        self.peak_balance = max(self.peak_balance, current_equity)
        self.current_drawdown = (self.peak_balance - current_equity) / self.peak_balance
        
        # Check if we've exceeded maximum drawdown
        if self.current_drawdown > MAX_DRAWDOWN:
            self.logger.warning(f"Maximum drawdown exceeded: {self.current_drawdown:.2%}")
            return False
        
        return True

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        account_balance: float,
        risk_per_trade: float = RISK_PER_TRADE
    ) -> float:
        """Calculate position size based on risk management rules."""
        # Calculate risk amount in account currency
        risk_amount = account_balance * risk_per_trade
        
        # Calculate price difference between entry and stop loss
        price_diff = abs(entry_price - stop_loss)
        
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
        entry_price: float,
        atr: float,
        atr_multiplier: float = 2.0
    ) -> float:
        """Calculate stop loss level using ATR."""
        return round(entry_price - (atr * atr_multiplier), 5)

    def calculate_take_profit(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        risk_reward_ratio: float = 2.0
    ) -> float:
        """Calculate take profit level based on risk-reward ratio."""
        # Calculate the distance to stop loss
        sl_distance = abs(entry_price - stop_loss)
        
        # Calculate take profit
        take_profit = entry_price + (sl_distance * risk_reward_ratio)
        
        return round(take_profit, 5)

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