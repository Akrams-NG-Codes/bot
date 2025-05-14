import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from .mt5_connector import MT5Connector
from .risk_manager import RiskManager
from strategies.rsi_strategy import RSIStrategy
from ml.market_classifier import MarketClassifier
from ml.strategy_selector import StrategySelector
from config.config import (
    SYMBOLS,
    TIMEFRAMES,
    ML_CONFIG,
    EMAIL_CONFIG,
    TELEGRAM_CONFIG
)
from utils.logger import setup_logger
from utils.notifier import Notifier

class AkramsNG:
    def __init__(self):
        # Setup logging
        self.logger = setup_logger('akrams_ng')
        
        # Initialize components
        self.mt5 = MT5Connector()
        self.risk_manager = RiskManager()
        self.market_classifier = MarketClassifier(ML_CONFIG)
        self.strategy_selector = StrategySelector(ML_CONFIG)
        self.notifier = Notifier(EMAIL_CONFIG, TELEGRAM_CONFIG)
        
        # Initialize strategies
        self.strategies = {
            'RSI': RSIStrategy(),
            # Add other strategies here
        }
        
        # Trading state
        self.is_running = False
        self.trading_thread = None
        self.last_analysis_time = {}
        self.analysis_interval = 60  # seconds
        
        # Performance tracking
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0
        }

    def start(self):
        """Start the trading bot."""
        if self.is_running:
            self.logger.warning("Bot is already running")
            return
        
        self.is_running = True
        self.trading_thread = threading.Thread(target=self._trading_loop)
        self.trading_thread.start()
        
        self.logger.info("Trading bot started")
        self.notifier.send_notification(
            "Trading Bot Started",
            "Akrams-NG trading bot has been started successfully."
        )

    def stop(self):
        """Stop the trading bot."""
        if not self.is_running:
            self.logger.warning("Bot is not running")
            return
        
        self.is_running = False
        if self.trading_thread:
            self.trading_thread.join()
        
        self.logger.info("Trading bot stopped")
        self.notifier.send_notification(
            "Trading Bot Stopped",
            "Akrams-NG trading bot has been stopped."
        )

    def _trading_loop(self):
        """Main trading loop."""
        while self.is_running:
            try:
                # Get account information
                account_info = self.mt5.get_account_info()
                if not account_info:
                    self.logger.error("Failed to get account information")
                    time.sleep(5)
                    continue
                
                # Analyze each symbol
                for symbol in SYMBOLS:
                    self._analyze_symbol(symbol, account_info)
                
                # Sleep for a short interval
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {str(e)}")
                self.notifier.send_notification(
                    "Trading Bot Error",
                    f"An error occurred in the trading loop: {str(e)}"
                )
                time.sleep(5)

    def _analyze_symbol(self, symbol: str, account_info: Dict):
        """Analyze a symbol and execute trades if conditions are met."""
        # Check if it's time to analyze this symbol
        current_time = datetime.now()
        if (symbol in self.last_analysis_time and
            (current_time - self.last_analysis_time[symbol]).total_seconds() < self.analysis_interval):
            return
        
        self.last_analysis_time[symbol] = current_time
        
        try:
            # Get market data
            market_data = self._get_market_data(symbol)
            if market_data.empty:
                return
            
            # Classify market conditions
            market_condition = self.market_classifier.predict(market_data)
            
            # Select best strategy
            selected_strategy = self.strategy_selector.select_strategy(market_data)
            if not selected_strategy:
                return
            
            # Get strategy instance
            strategy = self.strategies.get(selected_strategy)
            if not strategy:
                return
            
            # Calculate indicators and generate signals
            market_data = strategy.calculate_indicators(market_data)
            signals = strategy.generate_signals(market_data)
            
            # Get latest signal
            latest_signal = signals.iloc[-1]['signal'] if not signals.empty else None
            
            # Check if we can open a new position
            if latest_signal and latest_signal != 'NEUTRAL':
                if self.risk_manager.can_open_position(
                    symbol,
                    selected_strategy,
                    account_info['balance'],
                    account_info['equity']
                ):
                    self._execute_trade(symbol, latest_signal, strategy, account_info)
            
            # Update existing positions
            self._update_positions(symbol, market_data)
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {str(e)}")

    def _get_market_data(self, symbol: str) -> pd.DataFrame:
        """Get market data for analysis."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Get 30 days of data
        
        data = self.mt5.get_historical_data(
            symbol,
            TIMEFRAMES['H1'],  # Use 1-hour timeframe for analysis
            start_date,
            end_date
        )
        
        return data

    def _execute_trade(
        self,
        symbol: str,
        signal: str,
        strategy: RSIStrategy,
        account_info: Dict
    ):
        """Execute a trade based on the signal."""
        try:
            # Get current price
            price_info = self.mt5.get_current_price(symbol)
            if not price_info:
                return
            
            entry_price = price_info['ask'] if signal == 'BUY' else price_info['bid']
            
            # Calculate stop loss and take profit
            stop_loss = strategy.calculate_stop_loss(symbol, signal, entry_price)
            take_profit = strategy.calculate_take_profit(symbol, signal, entry_price, stop_loss)
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                symbol,
                entry_price,
                stop_loss,
                account_info['balance']
            )
            
            if position_size <= 0:
                return
            
            # Place the order
            order_id = self.mt5.place_order(
                symbol=symbol,
                order_type=signal,
                volume=position_size,
                price=entry_price,
                sl=stop_loss,
                tp=take_profit,
                comment=f"{strategy.name} Strategy"
            )
            
            if order_id:
                # Add position to risk manager
                self.risk_manager.add_position(
                    position_id=order_id,
                    symbol=symbol,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    position_size=position_size,
                    strategy=strategy.name
                )
                
                # Send notification
                self.notifier.send_notification(
                    "New Trade Executed",
                    f"Symbol: {symbol}\n"
                    f"Type: {signal}\n"
                    f"Entry: {entry_price}\n"
                    f"Stop Loss: {stop_loss}\n"
                    f"Take Profit: {take_profit}\n"
                    f"Size: {position_size}\n"
                    f"Strategy: {strategy.name}"
                )
                
        except Exception as e:
            self.logger.error(f"Error executing trade for {symbol}: {str(e)}")

    def _update_positions(self, symbol: str, market_data: pd.DataFrame):
        """Update and manage existing positions."""
        if market_data.empty:
            return
        
        current_price = market_data.iloc[-1]['close']
        
        # Get open positions for this symbol
        positions = self.mt5.get_open_positions()
        symbol_positions = [pos for pos in positions if pos['symbol'] == symbol]
        
        for position in symbol_positions:
            # Update position in risk manager
            updated_position = self.risk_manager.update_position(
                position['ticket'],
                current_price,
                self.mt5.get_account_info()['equity']
            )
            
            if not updated_position:
                continue
            
            # Check if stop loss or take profit was hit
            if (current_price <= position['sl'] or current_price >= position['tp']):
                self.mt5.close_order(position['ticket'])
                self.risk_manager.remove_position(position['ticket'])
                
                # Update performance metrics
                self._update_performance_metrics(position)

    def _update_performance_metrics(self, position: Dict):
        """Update performance metrics after a trade is closed."""
        self.performance_metrics['total_trades'] += 1
        
        if position['profit'] > 0:
            self.performance_metrics['winning_trades'] += 1
        else:
            self.performance_metrics['losing_trades'] += 1
        
        self.performance_metrics['total_profit'] += position['profit']
        self.performance_metrics['win_rate'] = (
            self.performance_metrics['winning_trades'] /
            self.performance_metrics['total_trades']
        )
        
        # Update max drawdown
        current_drawdown = self.risk_manager.current_drawdown
        self.performance_metrics['max_drawdown'] = max(
            self.performance_metrics['max_drawdown'],
            current_drawdown
        )

    def get_performance_report(self) -> Dict:
        """Get current performance report."""
        return {
            'performance_metrics': self.performance_metrics,
            'portfolio_risk': self.risk_manager.get_portfolio_risk(),
            'open_positions': self.mt5.get_open_positions(),
            'account_info': self.mt5.get_account_info()
        }

    def retrain_models(self):
        """Retrain machine learning models."""
        try:
            # Get historical data for training
            training_data = {}
            for symbol in SYMBOLS:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=90)  # Use 90 days of data
                
                data = self.mt5.get_historical_data(
                    symbol,
                    TIMEFRAMES['H1'],
                    start_date,
                    end_date
                )
                
                if not data.empty:
                    training_data[symbol] = data
            
            # Train market classifier
            self.market_classifier.train(pd.concat(training_data.values()))
            
            # Train strategy selector
            self.strategy_selector.train(
                pd.concat(training_data.values()),
                self._get_strategy_performance()
            )
            
            self.logger.info("Models retrained successfully")
            self.notifier.send_notification(
                "Models Retrained",
                "Machine learning models have been retrained successfully."
            )
            
        except Exception as e:
            self.logger.error(f"Error retraining models: {str(e)}")
            self.notifier.send_notification(
                "Model Retraining Error",
                f"An error occurred while retraining models: {str(e)}"
            )

    def _get_strategy_performance(self) -> Dict[str, pd.DataFrame]:
        """Get historical performance data for each strategy."""
        performance_data = {}
        
        for strategy_name, strategy in self.strategies.items():
            # Get strategy performance metrics
            metrics = strategy.get_performance_metrics()
            
            # Convert to DataFrame
            performance_data[strategy_name] = pd.DataFrame({
                'profit': [metrics['total_profit']],
                'win_rate': [metrics['win_rate']],
                'risk_reward_ratio': [metrics['risk_reward_ratio']]
            })
        
        return performance_data 