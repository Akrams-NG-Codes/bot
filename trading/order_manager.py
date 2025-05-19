import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path
import time

class OrderManager:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.orders = {}
        self.positions = {}
        self.last_order_time = {}
        self.initialize_mt5()
        
    def initialize_mt5(self):
        """Initialize MetaTrader 5 connection."""
        try:
            if not mt5.initialize():
                self.logger.error(f"Failed to initialize MT5: {mt5.last_error()}")
                return False
            
            # Set up symbol properties
            for symbol in self.config.get('symbols', []):
                if not mt5.symbol_select(symbol, True):
                    self.logger.error(f"Failed to select symbol {symbol}: {mt5.last_error()}")
                    return False
            
            self.logger.info("MT5 initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing MT5: {str(e)}")
            return False
    
    def place_order(self, order: Dict) -> Optional[int]:
        """Place a new order."""
        try:
            # Validate order
            if not self._validate_order(order):
                return None
            
            # Check trading conditions
            if not self._check_trading_conditions(order):
                return None
            
            # Prepare order request
            request = self._prepare_order_request(order)
            if request is None:
                return None
            
            # Send order
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Order failed: {result.comment}")
                return None
            
            # Store order information
            order_id = result.order
            self.orders[order_id] = {
                'order_id': order_id,
                'symbol': order['symbol'],
                'type': order['type'],
                'volume': order['volume'],
                'price': order['price'],
                'stop_loss': order.get('stop_loss'),
                'take_profit': order.get('take_profit'),
                'comment': order.get('comment', ''),
                'timestamp': datetime.now().isoformat(),
                'status': 'open'
            }
            
            self.last_order_time[order['symbol']] = datetime.now()
            
            self.logger.info(f"Order placed successfully: {order_id}")
            return order_id
            
        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
            return None
    
    def _validate_order(self, order: Dict) -> bool:
        """Validate order parameters."""
        try:
            required_fields = ['symbol', 'type', 'volume', 'price']
            if not all(field in order for field in required_fields):
                self.logger.error("Missing required order fields")
                return False
            
            # Check symbol
            if not mt5.symbol_select(order['symbol'], True):
                self.logger.error(f"Invalid symbol: {order['symbol']}")
                return False
            
            # Check volume
            symbol_info = mt5.symbol_info(order['symbol'])
            if symbol_info is None:
                return False
            
            min_volume = symbol_info.volume_min
            max_volume = symbol_info.volume_max
            
            if not (min_volume <= order['volume'] <= max_volume):
                self.logger.error(f"Invalid volume: {order['volume']}")
                return False
            
            # Check price
            if order['price'] <= 0:
                self.logger.error(f"Invalid price: {order['price']}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating order: {str(e)}")
            return False
    
    def _check_trading_conditions(self, order: Dict) -> bool:
        """Check if trading conditions are met."""
        try:
            symbol = order['symbol']
            
            # Check minimum time between orders
            if symbol in self.last_order_time:
                min_interval = self.config.get('min_order_interval', 60)  # seconds
                time_since_last = (datetime.now() - self.last_order_time[symbol]).total_seconds()
                if time_since_last < min_interval:
                    self.logger.warning(f"Minimum order interval not met: {time_since_last}s")
                    return False
            
            # Check maximum positions
            max_positions = self.config.get('max_positions', 5)
            if len(self.positions) >= max_positions:
                self.logger.warning("Maximum positions reached")
                return False
            
            # Check maximum positions per symbol
            max_per_symbol = self.config.get('max_positions_per_symbol', 1)
            symbol_positions = sum(1 for p in self.positions.values() if p['symbol'] == symbol)
            if symbol_positions >= max_per_symbol:
                self.logger.warning(f"Maximum positions for {symbol} reached")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking trading conditions: {str(e)}")
            return False
    
    def _prepare_order_request(self, order: Dict) -> Optional[Dict]:
        """Prepare order request for MT5."""
        try:
            symbol = order['symbol']
            order_type = order['type']
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
            
            # Prepare request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": order['volume'],
                "type": mt5.ORDER_TYPE_BUY if order_type == 'buy' else mt5.ORDER_TYPE_SELL,
                "price": tick.ask if order_type == 'buy' else tick.bid,
                "deviation": self.config.get('slippage', 10),
                "magic": self.config.get('magic_number', 123456),
                "comment": order.get('comment', ''),
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Add stop loss and take profit
            if 'stop_loss' in order:
                request['sl'] = order['stop_loss']
            if 'take_profit' in order:
                request['tp'] = order['take_profit']
            
            return request
            
        except Exception as e:
            self.logger.error(f"Error preparing order request: {str(e)}")
            return None
    
    def modify_order(self, order_id: int, modifications: Dict) -> bool:
        """Modify an existing order."""
        try:
            if order_id not in self.orders:
                self.logger.error(f"Order not found: {order_id}")
                return False
            
            order = self.orders[order_id]
            
            # Prepare modification request
            request = {
                "action": mt5.TRADE_ACTION_MODIFY,
                "position": order_id,
                "symbol": order['symbol'],
                "sl": modifications.get('stop_loss', order['stop_loss']),
                "tp": modifications.get('take_profit', order['take_profit']),
            }
            
            # Send modification request
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Order modification failed: {result.comment}")
                return False
            
            # Update order information
            self.orders[order_id].update(modifications)
            
            self.logger.info(f"Order modified successfully: {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error modifying order: {str(e)}")
            return False
    
    def close_order(self, order_id: int) -> bool:
        """Close an existing order."""
        try:
            if order_id not in self.orders:
                self.logger.error(f"Order not found: {order_id}")
                return False
            
            order = self.orders[order_id]
            
            # Prepare close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": order_id,
                "symbol": order['symbol'],
                "volume": order['volume'],
                "type": mt5.ORDER_TYPE_SELL if order['type'] == 'buy' else mt5.ORDER_TYPE_BUY,
                "deviation": self.config.get('slippage', 10),
                "magic": self.config.get('magic_number', 123456),
                "comment": "Close order",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send close request
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Order close failed: {result.comment}")
                return False
            
            # Update order status
            self.orders[order_id]['status'] = 'closed'
            self.orders[order_id]['close_time'] = datetime.now().isoformat()
            
            self.logger.info(f"Order closed successfully: {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error closing order: {str(e)}")
            return False
    
    def update_positions(self):
        """Update current positions."""
        try:
            positions = mt5.positions_get()
            if positions is None:
                return
            
            for position in positions:
                position_id = position.ticket
                self.positions[position_id] = {
                    'position_id': position_id,
                    'symbol': position.symbol,
                    'type': 'buy' if position.type == mt5.POSITION_TYPE_BUY else 'sell',
                    'volume': position.volume,
                    'price': position.price_open,
                    'stop_loss': position.sl,
                    'take_profit': position.tp,
                    'profit': position.profit,
                    'swap': position.swap,
                    'timestamp': datetime.fromtimestamp(position.time).isoformat()
                }
            
        except Exception as e:
            self.logger.error(f"Error updating positions: {str(e)}")
    
    def get_order(self, order_id: int) -> Optional[Dict]:
        """Get order information."""
        return self.orders.get(order_id)
    
    def get_position(self, position_id: int) -> Optional[Dict]:
        """Get position information."""
        return self.positions.get(position_id)
    
    def get_all_orders(self) -> Dict:
        """Get all orders."""
        return self.orders
    
    def get_all_positions(self) -> Dict:
        """Get all positions."""
        return self.positions
    
    def save_orders(self, path: str):
        """Save orders to file."""
        try:
            # Create directory if it doesn't exist
            Path(path).mkdir(parents=True, exist_ok=True)
            
            # Save orders
            file_path = Path(path) / 'orders.json'
            with open(file_path, 'w') as f:
                json.dump(self.orders, f, indent=2, default=str)
            
            self.logger.info(f"Orders saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving orders: {str(e)}")
    
    def load_orders(self, path: str):
        """Load orders from file."""
        try:
            file_path = Path(path) / 'orders.json'
            if not file_path.exists():
                return
            
            with open(file_path, 'r') as f:
                self.orders = json.load(f)
            
            self.logger.info(f"Orders loaded from {file_path}")
        except Exception as e:
            self.logger.error(f"Error loading orders: {str(e)}")
    
    def cleanup(self):
        """Clean up resources."""
        try:
            mt5.shutdown()
            self.logger.info("MT5 connection closed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}") 