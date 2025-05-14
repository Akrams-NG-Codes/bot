import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import threading
import queue
import time
import json
from pathlib import Path
import sqlite3
import hashlib

class DataManager:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_cache = {}
        self.realtime_data = {}
        self.data_queue = queue.Queue()
        self.monitoring = False
        self.monitor_thread = None
        
        # Initialize MT5
        if not mt5.initialize():
            self.logger.error("Failed to initialize MT5")
            raise RuntimeError("MT5 initialization failed")
        
        # Create data directory
        self.data_dir = Path('data')
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for data storage."""
        try:
            self.conn = sqlite3.connect('data/market_data.db')
            self.cursor = self.conn.cursor()
            
            # Create tables
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS price_data (
                    symbol TEXT,
                    timestamp DATETIME,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    PRIMARY KEY (symbol, timestamp)
                )
            ''')
            
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS metadata (
                    symbol TEXT,
                    last_update DATETIME,
                    data_hash TEXT,
                    PRIMARY KEY (symbol)
                )
            ''')
            
            self.conn.commit()
        except Exception as e:
            self.logger.error(f"Error initializing database: {str(e)}")
            raise

    def start_data_monitoring(self):
        """Start real-time data monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.logger.info("Data monitoring started")

    def stop_data_monitoring(self):
        """Stop real-time data monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.info("Data monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop for real-time data."""
        while self.monitoring:
            try:
                # Update real-time data
                self._update_realtime_data()
                
                # Process data queue
                self._process_data_queue()
                
                # Save data periodically
                self._save_data()
                
                time.sleep(self.config.get('update_interval', 1))
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")

    def _update_realtime_data(self):
        """Update real-time market data."""
        try:
            symbols = self.config.get('symbols', [])
            for symbol in symbols:
                # Get latest tick data
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    continue
                
                # Update real-time data
                self.realtime_data[symbol] = {
                    'timestamp': datetime.now(),
                    'bid': tick.bid,
                    'ask': tick.ask,
                    'last': tick.last,
                    'volume': tick.volume
                }
                
                # Add to queue for processing
                self.data_queue.put({
                    'symbol': symbol,
                    'data': self.realtime_data[symbol]
                })
        except Exception as e:
            self.logger.error(f"Error updating real-time data: {str(e)}")

    def _process_data_queue(self):
        """Process queued data updates."""
        try:
            while True:
                try:
                    update = self.data_queue.get_nowait()
                    symbol = update['symbol']
                    data = update['data']
                    
                    # Update cache
                    if symbol not in self.data_cache:
                        self.data_cache[symbol] = []
                    self.data_cache[symbol].append(data)
                    
                    # Keep cache size within limits
                    max_cache_size = self.config.get('max_cache_size', 1000)
                    if len(self.data_cache[symbol]) > max_cache_size:
                        self.data_cache[symbol] = self.data_cache[symbol][-max_cache_size:]
                except queue.Empty:
                    break
        except Exception as e:
            self.logger.error(f"Error processing data queue: {str(e)}")

    def _save_data(self):
        """Save data to database."""
        try:
            for symbol, data in self.data_cache.items():
                if not data:
                    continue
                
                # Prepare data for database
                records = []
                for entry in data:
                    records.append((
                        symbol,
                        entry['timestamp'],
                        entry.get('open', entry['last']),
                        entry.get('high', entry['last']),
                        entry.get('low', entry['last']),
                        entry['last'],
                        entry['volume']
                    ))
                
                # Insert data
                self.cursor.executemany('''
                    INSERT OR REPLACE INTO price_data
                    (symbol, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', records)
                
                # Update metadata
                data_hash = self._calculate_data_hash(data)
                self.cursor.execute('''
                    INSERT OR REPLACE INTO metadata
                    (symbol, last_update, data_hash)
                    VALUES (?, ?, ?)
                ''', (symbol, datetime.now(), data_hash))
                
                self.conn.commit()
                
                # Clear processed data
                self.data_cache[symbol] = []
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")

    def _calculate_data_hash(self, data: List[Dict]) -> str:
        """Calculate hash of data for change detection."""
        try:
            data_str = json.dumps(data, sort_keys=True)
            return hashlib.md5(data_str.encode()).hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating data hash: {str(e)}")
            return ''

    def get_historical_data(self, symbol: str,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          timeframe: str = '1m') -> pd.DataFrame:
        """Get historical price data."""
        try:
            if not start_time:
                start_time = datetime.now() - timedelta(days=30)
            if not end_time:
                end_time = datetime.now()
            
            # Query database
            self.cursor.execute('''
                SELECT timestamp, open, high, low, close, volume
                FROM price_data
                WHERE symbol = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            ''', (symbol, start_time, end_time))
            
            # Convert to DataFrame
            data = pd.DataFrame(
                self.cursor.fetchall(),
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            if data.empty:
                return pd.DataFrame()
            
            data.set_index('timestamp', inplace=True)
            return data
        except Exception as e:
            self.logger.error(f"Error getting historical data: {str(e)}")
            return pd.DataFrame()

    def get_realtime_data(self, symbol: str) -> Dict:
        """Get latest real-time data for symbol."""
        return self.realtime_data.get(symbol, {})

    def get_data_metadata(self, symbol: str) -> Dict:
        """Get metadata for symbol's data."""
        try:
            self.cursor.execute('''
                SELECT last_update, data_hash
                FROM metadata
                WHERE symbol = ?
            ''', (symbol,))
            
            result = self.cursor.fetchone()
            if result:
                return {
                    'last_update': result[0],
                    'data_hash': result[1]
                }
            return {}
        except Exception as e:
            self.logger.error(f"Error getting data metadata: {str(e)}")
            return {}

    def validate_data(self, symbol: str) -> bool:
        """Validate data integrity for symbol."""
        try:
            # Get current data
            data = self.get_historical_data(symbol)
            if data.empty:
                return False
            
            # Calculate current hash
            current_hash = self._calculate_data_hash(data.to_dict('records'))
            
            # Get stored hash
            metadata = self.get_data_metadata(symbol)
            stored_hash = metadata.get('data_hash', '')
            
            return current_hash == stored_hash
        except Exception as e:
            self.logger.error(f"Error validating data: {str(e)}")
            return False

    def cleanup_old_data(self, days: int = 30):
        """Clean up old data from database."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            self.cursor.execute('''
                DELETE FROM price_data
                WHERE timestamp < ?
            ''', (cutoff_date,))
            
            self.conn.commit()
            self.logger.info(f"Cleaned up data older than {days} days")
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {str(e)}")

    def __del__(self):
        """Cleanup on object destruction."""
        try:
            if hasattr(self, 'conn'):
                self.conn.close()
            mt5.shutdown()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}") 