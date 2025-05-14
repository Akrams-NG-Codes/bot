import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import MetaTrader5 as mt5

class RiskManager:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.initial_balance = config.get('initial_balance', 10000)
        self.max_risk_per_trade = config.get('max_risk_per_trade', 0.02)  # 2% per trade
        self.max_daily_loss = config.get('max_daily_loss', 0.05)  # 5% daily loss limit
        self.max_drawdown = config.get('max_drawdown', 0.15)  # 15% max drawdown
        self.max_correlation = config.get('max_correlation', 0.7)  # Maximum correlation between positions
        self.max_positions = config.get('max_positions', 5)  # Maximum number of concurrent positions
        self.position_sizes = {}  # Track current position sizes
        self.daily_pnl = 0.0
        self.peak_balance = self.initial_balance
        self.current_drawdown = 0.0
        self.last_reset_date = datetime.now().date()

    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float, 
                              direction: str) -> float:
        """Calculate position size based on risk parameters."""
        # Reset daily tracking if it's a new day
        self._check_daily_reset()
        
        # Check if we've hit daily loss limit
        if self.daily_pnl <= -self.initial_balance * self.max_daily_loss:
            self.logger.warning("Daily loss limit reached. No new positions allowed.")
            return 0.0
        
        # Check if we've hit max drawdown
        if self.current_drawdown >= self.max_drawdown:
            self.logger.warning("Maximum drawdown reached. No new positions allowed.")
            return 0.0
        
        # Check if we've hit max positions
        if len(self.position_sizes) >= self.max_positions:
            self.logger.warning("Maximum number of positions reached.")
            return 0.0
        
        # Calculate risk amount
        risk_amount = self.initial_balance * self.max_risk_per_trade
        
        # Calculate position size
        if direction == "BUY":
            risk_per_unit = entry_price - stop_loss
        else:  # SELL
            risk_per_unit = stop_loss - entry_price
        
        if risk_per_unit <= 0:
            self.logger.error("Invalid stop loss level")
            return 0.0
        
        position_size = risk_amount / risk_per_unit
        
        # Check correlation with existing positions
        if not self._check_correlation(symbol):
            self.logger.warning(f"Position rejected due to high correlation with existing positions")
            return 0.0
        
        return position_size

    def update_position(self, symbol: str, pnl: float):
        """Update position tracking and risk metrics."""
        self.daily_pnl += pnl
        
        # Update drawdown
        current_balance = self.initial_balance + self.daily_pnl
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
        self.current_drawdown = (self.peak_balance - current_balance) / self.peak_balance

    def close_position(self, symbol: str):
        """Remove position from tracking."""
        if symbol in self.position_sizes:
            del self.position_sizes[symbol]

    def _check_daily_reset(self):
        """Reset daily tracking if it's a new day."""
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.daily_pnl = 0.0
            self.last_reset_date = current_date

    def _check_correlation(self, symbol: str) -> bool:
        """Check correlation with existing positions."""
        if not self.position_sizes:
            return True
        
        # Get historical data for correlation calculation
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Get data for new symbol
        new_data = self._get_historical_data(symbol, start_date, end_date)
        if new_data is None:
            return True
        
        # Check correlation with each existing position
        for existing_symbol in self.position_sizes:
            existing_data = self._get_historical_data(existing_symbol, start_date, end_date)
            if existing_data is None:
                continue
            
            correlation = new_data['close'].corr(existing_data['close'])
            if abs(correlation) > self.max_correlation:
                return False
        
        return True

    def _get_historical_data(self, symbol: str, start_date: datetime, 
                           end_date: datetime) -> Optional[pd.DataFrame]:
        """Get historical data from MT5."""
        try:
            rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, 
                                       int(start_date.timestamp()),
                                       int(end_date.timestamp()))
            if rates is None:
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {str(e)}")
            return None

    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics."""
        return {
            'daily_pnl': self.daily_pnl,
            'daily_pnl_percent': self.daily_pnl / self.initial_balance,
            'current_drawdown': self.current_drawdown,
            'peak_balance': self.peak_balance,
            'current_balance': self.initial_balance + self.daily_pnl,
            'active_positions': len(self.position_sizes),
            'max_positions_remaining': self.max_positions - len(self.position_sizes)
        }

    def can_open_position(self) -> bool:
        """Check if new positions can be opened."""
        self._check_daily_reset()
        
        if self.daily_pnl <= -self.initial_balance * self.max_daily_loss:
            return False
        
        if self.current_drawdown >= self.max_drawdown:
            return False
        
        if len(self.position_sizes) >= self.max_positions:
            return False
        
        return True 