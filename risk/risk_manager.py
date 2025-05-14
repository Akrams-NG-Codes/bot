import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import json
from pathlib import Path

class RiskManager:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.risk_metrics = {}
        self.daily_risk = 0.0
        self.last_reset = datetime.now()
        
    def calculate_position_size(self, signal: Dict) -> float:
        """Calculate position size based on risk parameters."""
        try:
            method = self.config.get('position_sizing', {}).get('method', 'risk_based')
            
            if method == 'risk_based':
                return self._calculate_risk_based_size(signal)
            elif method == 'fixed':
                return self._calculate_fixed_size()
            elif method == 'kelly':
                return self._calculate_kelly_size(signal)
            else:
                self.logger.warning(f"Unknown position sizing method: {method}")
                return self._calculate_risk_based_size(signal)
                
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0.0
    
    def _calculate_risk_based_size(self, signal: Dict) -> float:
        """Calculate position size based on risk percentage."""
        try:
            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                return 0.0
            
            # Get risk parameters
            max_risk = self.config.get('risk', {}).get('max_risk_per_trade', 0.02)
            account_balance = account_info.balance
            
            # Calculate risk amount
            risk_amount = account_balance * max_risk
            
            # Calculate stop loss distance
            stop_loss = self._calculate_stop_loss(signal)
            if stop_loss is None:
                return 0.0
            
            # Calculate position size
            price = signal['price']
            stop_distance = abs(price - stop_loss)
            
            if stop_distance == 0:
                return 0.0
            
            # Get symbol info
            symbol_info = mt5.symbol_info(signal['symbol'])
            if symbol_info is None:
                return 0.0
            
            # Calculate position size in lots
            position_size = risk_amount / (stop_distance * symbol_info.trade_contract_size)
            
            # Apply size limits
            position_size = self._apply_size_limits(position_size)
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating risk-based size: {str(e)}")
            return 0.0
    
    def _calculate_fixed_size(self) -> float:
        """Calculate fixed position size."""
        try:
            fixed_size = self.config.get('position_sizing', {}).get('fixed_size', 0.1)
            return self._apply_size_limits(fixed_size)
        except Exception as e:
            self.logger.error(f"Error calculating fixed size: {str(e)}")
            return 0.0
    
    def _calculate_kelly_size(self, signal: Dict) -> float:
        """Calculate position size using Kelly Criterion."""
        try:
            # Get historical performance
            win_rate = self.risk_metrics.get('win_rate', 0.5)
            avg_win = self.risk_metrics.get('avg_win', 1.0)
            avg_loss = self.risk_metrics.get('avg_loss', 1.0)
            
            if avg_loss == 0:
                return 0.0
            
            # Calculate Kelly fraction
            kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            
            # Apply conservative Kelly (half Kelly)
            kelly = kelly * 0.5
            
            # Calculate position size
            account_info = mt5.account_info()
            if account_info is None:
                return 0.0
            
            position_size = kelly * account_info.balance
            
            # Convert to lots
            symbol_info = mt5.symbol_info(signal['symbol'])
            if symbol_info is None:
                return 0.0
            
            position_size = position_size / symbol_info.trade_contract_size
            
            return self._apply_size_limits(position_size)
            
        except Exception as e:
            self.logger.error(f"Error calculating Kelly size: {str(e)}")
            return 0.0
    
    def _apply_size_limits(self, size: float) -> float:
        """Apply position size limits."""
        try:
            # Get size limits
            max_size = self.config.get('position_sizing', {}).get('max_position_size', 1.0)
            min_size = self.config.get('position_sizing', {}).get('min_position_size', 0.01)
            increment = self.config.get('position_sizing', {}).get('size_increment', 0.01)
            
            # Apply limits
            size = min(max_size, max(min_size, size))
            
            # Round to increment
            size = round(size / increment) * increment
            
            return size
            
        except Exception as e:
            self.logger.error(f"Error applying size limits: {str(e)}")
            return 0.0
    
    def _calculate_stop_loss(self, signal: Dict) -> Optional[float]:
        """Calculate stop loss level."""
        try:
            if 'stop_loss' in signal:
                return signal['stop_loss']
            
            # Get ATR
            atr = self._calculate_atr(signal['symbol'])
            if atr is None:
                return None
            
            # Calculate stop loss
            atr_multiplier = self.config.get('risk', {}).get('stop_loss_atr_multiplier', 2.0)
            stop_distance = atr * atr_multiplier
            
            if signal['direction'] == 'long':
                return signal['price'] - stop_distance
            else:
                return signal['price'] + stop_distance
                
        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {str(e)}")
            return None
    
    def _calculate_atr(self, symbol: str, period: int = 14) -> Optional[float]:
        """Calculate Average True Range."""
        try:
            # Get historical data
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, period + 1)
            if rates is None:
                return None
            
            # Calculate True Range
            df = pd.DataFrame(rates)
            df['tr'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            
            # Calculate ATR
            atr = df['tr'].rolling(window=period).mean().iloc[-1]
            
            return atr
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {str(e)}")
            return None
    
    def update_risk_metrics(self, trade: Dict):
        """Update risk metrics with new trade."""
        try:
            # Reset daily risk if needed
            self._check_daily_reset()
            
            # Update daily risk
            self.daily_risk += abs(trade.get('profit', 0))
            
            # Update risk metrics
            if 'trades' not in self.risk_metrics:
                self.risk_metrics['trades'] = []
            
            self.risk_metrics['trades'].append(trade)
            
            # Calculate metrics
            self._calculate_risk_metrics()
            
            # Check risk limits
            self._check_risk_limits()
            
        except Exception as e:
            self.logger.error(f"Error updating risk metrics: {str(e)}")
    
    def _check_daily_reset(self):
        """Check and reset daily risk if needed."""
        try:
            now = datetime.now()
            if (now - self.last_reset).days >= 1:
                self.daily_risk = 0.0
                self.last_reset = now
        except Exception as e:
            self.logger.error(f"Error checking daily reset: {str(e)}")
    
    def _calculate_risk_metrics(self):
        """Calculate risk metrics from trade history."""
        try:
            trades = self.risk_metrics.get('trades', [])
            if not trades:
                return
            
            # Calculate basic metrics
            winning_trades = [t for t in trades if t.get('profit', 0) > 0]
            losing_trades = [t for t in trades if t.get('profit', 0) < 0]
            
            self.risk_metrics.update({
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': len(winning_trades) / len(trades) if trades else 0,
                'avg_win': np.mean([t.get('profit', 0) for t in winning_trades]) if winning_trades else 0,
                'avg_loss': np.mean([t.get('profit', 0) for t in losing_trades]) if losing_trades else 0,
                'profit_factor': abs(sum(t.get('profit', 0) for t in winning_trades) /
                                   sum(t.get('profit', 0) for t in losing_trades)) if losing_trades else float('inf'),
                'max_drawdown': self._calculate_max_drawdown(trades)
            })
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {str(e)}")
    
    def _calculate_max_drawdown(self, trades: List[Dict]) -> float:
        """Calculate maximum drawdown from trades."""
        try:
            if not trades:
                return 0.0
            
            # Calculate cumulative returns
            returns = [t.get('profit', 0) for t in trades]
            cumulative = np.cumsum(returns)
            
            # Calculate running maximum
            running_max = np.maximum.accumulate(cumulative)
            
            # Calculate drawdowns
            drawdowns = running_max - cumulative
            
            return float(np.max(drawdowns))
            
        except Exception as e:
            self.logger.error(f"Error calculating max drawdown: {str(e)}")
            return 0.0
    
    def _check_risk_limits(self):
        """Check if risk limits are exceeded."""
        try:
            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                return
            
            # Check daily risk
            max_daily_risk = self.config.get('risk', {}).get('max_daily_risk', 0.05)
            if self.daily_risk > account_info.balance * max_daily_risk:
                self._generate_alert('daily_risk_exceeded')
            
            # Check max drawdown
            max_drawdown = self.config.get('risk', {}).get('max_drawdown', 0.15)
            if self.risk_metrics.get('max_drawdown', 0) > account_info.balance * max_drawdown:
                self._generate_alert('max_drawdown_exceeded')
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {str(e)}")
    
    def _generate_alert(self, alert_type: str):
        """Generate risk alert."""
        try:
            alert = {
                'timestamp': datetime.now().isoformat(),
                'type': alert_type,
                'message': f"Risk limit exceeded: {alert_type}",
                'metrics': self.risk_metrics
            }
            
            self.logger.warning(f"Risk alert: {alert}")
            
        except Exception as e:
            self.logger.error(f"Error generating alert: {str(e)}")
    
    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics."""
        return self.risk_metrics
    
    def save_risk_metrics(self, path: str):
        """Save risk metrics to file."""
        try:
            # Create directory if it doesn't exist
            Path(path).mkdir(parents=True, exist_ok=True)
            
            # Save metrics
            file_path = Path(path) / 'risk_metrics.json'
            with open(file_path, 'w') as f:
                json.dump(self.risk_metrics, f, indent=2, default=str)
            
            self.logger.info(f"Risk metrics saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving risk metrics: {str(e)}")
    
    def load_risk_metrics(self, path: str):
        """Load risk metrics from file."""
        try:
            file_path = Path(path) / 'risk_metrics.json'
            if not file_path.exists():
                return
            
            with open(file_path, 'r') as f:
                self.risk_metrics = json.load(f)
            
            self.logger.info(f"Risk metrics loaded from {file_path}")
        except Exception as e:
            self.logger.error(f"Error loading risk metrics: {str(e)}") 