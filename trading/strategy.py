import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import MetaTrader5 as mt5
import json
from pathlib import Path

class BaseStrategy(ABC):
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.positions = {}
        self.trades = []
        self.performance = {}
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from market data."""
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: Dict) -> float:
        """Calculate position size based on risk parameters."""
        pass
    
    @abstractmethod
    def validate_signal(self, signal: Dict) -> bool:
        """Validate trading signal."""
        pass
    
    def execute_trade(self, signal: Dict):
        """Execute trading signal."""
        try:
            if not self.validate_signal(signal):
                return
            
            # Calculate position size
            size = self.calculate_position_size(signal)
            
            # Prepare order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": signal['symbol'],
                "volume": size,
                "type": mt5.ORDER_TYPE_BUY if signal['direction'] == 'long' else mt5.ORDER_TYPE_SELL,
                "price": signal['price'],
                "deviation": self.config.get('slippage', 10),
                "magic": self.config.get('magic_number', 123456),
                "comment": signal.get('comment', ''),
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Order failed: {result.comment}")
                return
            
            # Record trade
            trade = {
                'timestamp': datetime.now(),
                'symbol': signal['symbol'],
                'direction': signal['direction'],
                'size': size,
                'price': signal['price'],
                'order_id': result.order,
                'comment': signal.get('comment', '')
            }
            
            self.trades.append(trade)
            self.positions[signal['symbol']] = trade
            
            self.logger.info(f"Trade executed: {trade}")
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
    
    def update_positions(self):
        """Update open positions."""
        try:
            positions = mt5.positions_get()
            if positions is None:
                return
            
            for position in positions:
                symbol = position.symbol
                if symbol in self.positions:
                    self.positions[symbol].update({
                        'current_price': position.price_current,
                        'profit': position.profit,
                        'swap': position.swap,
                        'time': position.time
                    })
        except Exception as e:
            self.logger.error(f"Error updating positions: {str(e)}")
    
    def close_position(self, symbol: str):
        """Close position for symbol."""
        try:
            if symbol not in self.positions:
                return
            
            position = self.positions[symbol]
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": position['size'],
                "type": mt5.ORDER_TYPE_SELL if position['direction'] == 'long' else mt5.ORDER_TYPE_BUY,
                "position": position['order_id'],
                "price": mt5.symbol_info_tick(symbol).bid,
                "deviation": self.config.get('slippage', 10),
                "magic": self.config.get('magic_number', 123456),
                "comment": "Close position",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Close position failed: {result.comment}")
                return
            
            # Update trade record
            position['close_time'] = datetime.now()
            position['close_price'] = result.price
            position['status'] = 'closed'
            
            # Remove from active positions
            del self.positions[symbol]
            
            self.logger.info(f"Position closed: {position}")
            
        except Exception as e:
            self.logger.error(f"Error closing position: {str(e)}")
    
    def calculate_performance(self) -> Dict:
        """Calculate strategy performance metrics."""
        try:
            if not self.trades:
                return {}
            
            # Calculate basic metrics
            total_trades = len(self.trades)
            winning_trades = len([t for t in self.trades if t.get('profit', 0) > 0])
            losing_trades = len([t for t in self.trades if t.get('profit', 0) < 0])
            
            # Calculate profit metrics
            total_profit = sum(t.get('profit', 0) for t in self.trades)
            avg_profit = total_profit / total_trades if total_trades > 0 else 0
            
            # Calculate win rate
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Calculate risk metrics
            max_drawdown = self._calculate_max_drawdown()
            sharpe_ratio = self._calculate_sharpe_ratio()
            
            self.performance = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'avg_profit': avg_profit,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio
            }
            
            return self.performance
            
        except Exception as e:
            self.logger.error(f"Error calculating performance: {str(e)}")
            return {}
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        try:
            if not self.trades:
                return 0.0
            
            # Calculate cumulative returns
            returns = [t.get('profit', 0) for t in self.trades]
            cumulative = np.cumsum(returns)
            
            # Calculate running maximum
            running_max = np.maximum.accumulate(cumulative)
            
            # Calculate drawdowns
            drawdowns = running_max - cumulative
            
            return float(np.max(drawdowns))
        except Exception as e:
            self.logger.error(f"Error calculating max drawdown: {str(e)}")
            return 0.0
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio."""
        try:
            if not self.trades:
                return 0.0
            
            # Calculate daily returns
            returns = [t.get('profit', 0) for t in self.trades]
            
            # Calculate annualized Sharpe ratio
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0.0
            
            sharpe = (avg_return / std_return) * np.sqrt(252)  # Annualized
            return float(sharpe)
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0.0
    
    def save_trades(self, path: str):
        """Save trade history to file."""
        try:
            # Create directory if it doesn't exist
            Path(path).mkdir(parents=True, exist_ok=True)
            
            # Save trades
            file_path = Path(path) / 'trades.json'
            with open(file_path, 'w') as f:
                json.dump(self.trades, f, indent=2, default=str)
            
            self.logger.info(f"Trades saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving trades: {str(e)}")
    
    def load_trades(self, path: str):
        """Load trade history from file."""
        try:
            file_path = Path(path) / 'trades.json'
            if not file_path.exists():
                return
            
            with open(file_path, 'r') as f:
                self.trades = json.load(f)
            
            self.logger.info(f"Trades loaded from {file_path}")
        except Exception as e:
            self.logger.error(f"Error loading trades: {str(e)}")
    
    def get_open_positions(self) -> Dict:
        """Get current open positions."""
        return self.positions
    
    def get_trade_history(self) -> List[Dict]:
        """Get complete trade history."""
        return self.trades
    
    def get_performance(self) -> Dict:
        """Get current performance metrics."""
        return self.performance 