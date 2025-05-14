import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Union
from config.config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_TIMEOUT

class MT5Connector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.connected = False
        self.initialize()

    def initialize(self) -> bool:
        """Initialize connection to MetaTrader 5."""
        if not mt5.initialize(
            login=MT5_LOGIN,
            password=MT5_PASSWORD,
            server=MT5_SERVER,
            timeout=MT5_TIMEOUT
        ):
            self.logger.error(f"Failed to initialize MT5: {mt5.last_error()}")
            return False
        
        self.connected = True
        self.logger.info("Successfully connected to MetaTrader 5")
        return True

    def shutdown(self) -> None:
        """Shutdown MT5 connection."""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            self.logger.info("MT5 connection closed")

    def get_account_info(self) -> Dict:
        """Get account information."""
        if not self.connected:
            return {}
        
        account_info = mt5.account_info()
        if account_info is None:
            self.logger.error(f"Failed to get account info: {mt5.last_error()}")
            return {}
        
        return {
            'balance': account_info.balance,
            'equity': account_info.equity,
            'margin': account_info.margin,
            'free_margin': account_info.margin_free,
            'leverage': account_info.leverage
        }

    def get_historical_data(
        self,
        symbol: str,
        timeframe: int,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Get historical price data."""
        if not self.connected:
            return pd.DataFrame()

        rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
        if rates is None:
            self.logger.error(f"Failed to get historical data: {mt5.last_error()}")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    def get_current_price(self, symbol: str) -> Dict[str, float]:
        """Get current price for a symbol."""
        if not self.connected:
            return {}

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            self.logger.error(f"Failed to get current price: {mt5.last_error()}")
            return {}

        return {
            'bid': tick.bid,
            'ask': tick.ask,
            'last': tick.last,
            'volume': tick.volume
        }

    def place_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: float,
        sl: float,
        tp: float,
        comment: str = ""
    ) -> Optional[int]:
        """Place a new order."""
        if not self.connected:
            return None

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 234000,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(f"Order failed: {result.comment}")
            return None

        self.logger.info(f"Order placed successfully: {result.order}")
        return result.order

    def modify_order(
        self,
        order_id: int,
        sl: float,
        tp: float
    ) -> bool:
        """Modify an existing order."""
        if not self.connected:
            return False

        request = {
            "action": mt5.TRADE_ACTION_MODIFY,
            "order": order_id,
            "sl": sl,
            "tp": tp
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(f"Order modification failed: {result.comment}")
            return False

        self.logger.info(f"Order modified successfully: {order_id}")
        return True

    def close_order(self, order_id: int) -> bool:
        """Close an existing order."""
        if not self.connected:
            return False

        position = mt5.positions_get(ticket=order_id)
        if not position:
            self.logger.error(f"Position not found: {order_id}")
            return False

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position[0].symbol,
            "volume": position[0].volume,
            "type": mt5.ORDER_TYPE_SELL if position[0].type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "position": order_id,
            "price": mt5.symbol_info_tick(position[0].symbol).bid,
            "deviation": 20,
            "magic": 234000,
            "comment": "Close position",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(f"Order close failed: {result.comment}")
            return False

        self.logger.info(f"Order closed successfully: {order_id}")
        return True

    def get_open_positions(self) -> List[Dict]:
        """Get all open positions."""
        if not self.connected:
            return []

        positions = mt5.positions_get()
        if positions is None:
            self.logger.error(f"Failed to get open positions: {mt5.last_error()}")
            return []

        return [{
            'ticket': pos.ticket,
            'symbol': pos.symbol,
            'type': 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL',
            'volume': pos.volume,
            'open_price': pos.price_open,
            'current_price': pos.price_current,
            'sl': pos.sl,
            'tp': pos.tp,
            'profit': pos.profit
        } for pos in positions]

    def get_symbol_info(self, symbol: str) -> Dict:
        """Get symbol information."""
        if not self.connected:
            return {}

        info = mt5.symbol_info(symbol)
        if info is None:
            self.logger.error(f"Failed to get symbol info: {mt5.last_error()}")
            return {}

        return {
            'name': info.name,
            'bid': info.bid,
            'ask': info.ask,
            'point': info.point,
            'digits': info.digits,
            'spread': info.spread,
            'trade_contract_size': info.trade_contract_size,
            'volume_min': info.volume_min,
            'volume_max': info.volume_max,
            'volume_step': info.volume_step
        } 