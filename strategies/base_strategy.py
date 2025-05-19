from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class BaseStrategy(ABC):
    def __init__(self, name: str, params: Dict):
        self.name = name
        self.params = params
        self.logger = logging.getLogger(f"strategy.{name}")
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'risk_reward_ratio': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'current_consecutive_wins': 0,
            'current_consecutive_losses': 0,
            'market_regime_accuracy': 0.0,
            'strategy_confidence': 0.0
        }
        self.market_regime = 'unknown'
        self.strategy_confidence = 0.0
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=3)
        self.feature_importance = {}

    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the strategy."""
        pass

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on indicators."""
        pass

    @abstractmethod
    def calculate_position_size(
        self,
        symbol: str,
        signal: str,
        price: float,
        stop_loss: float,
        account_balance: float
    ) -> float:
        """Calculate position size based on risk management rules."""
        pass

    @abstractmethod
    def calculate_stop_loss(
        self,
        symbol: str,
        signal: str,
        price: float
    ) -> float:
        """Calculate stop loss level for the trade."""
        pass

    @abstractmethod
    def calculate_take_profit(
        self,
        symbol: str,
        signal: str,
        price: float,
        stop_loss: float
    ) -> float:
        """Calculate take profit level for the trade."""
        pass

    def analyze_market_regime(self, data: pd.DataFrame) -> str:
        """Analyze current market regime using multiple indicators."""
        # Calculate volatility
        volatility = self._calculate_volatility(data)
        
        # Calculate trend strength
        trend_strength = self._calculate_trend_strength(data)
        
        # Calculate market efficiency
        market_efficiency = self._calculate_market_efficiency(data)
        
        # Determine market regime
        if volatility > self.params.get('high_volatility_threshold', 0.2):
            return 'volatile'
        elif trend_strength > self.params.get('strong_trend_threshold', 0.7):
            return 'trending'
        elif market_efficiency > self.params.get('efficient_market_threshold', 0.6):
            return 'efficient'
        else:
            return 'ranging'

    def calculate_strategy_confidence(self, data: pd.DataFrame) -> float:
        """Calculate strategy confidence based on multiple factors."""
        # Calculate signal strength
        signal_strength = self._calculate_signal_strength(data)
        
        # Calculate indicator agreement
        indicator_agreement = self._calculate_indicator_agreement(data)
        
        # Calculate market alignment
        market_alignment = self._calculate_market_alignment(data)
        
        # Calculate historical accuracy
        historical_accuracy = self._calculate_historical_accuracy()
        
        # Combine factors with weights
        confidence = (
            signal_strength * 0.3 +
            indicator_agreement * 0.2 +
            market_alignment * 0.2 +
            historical_accuracy * 0.3
        )
        
        return min(max(confidence, 0.0), 1.0)

    def optimize_parameters(self, data: pd.DataFrame) -> Dict:
        """Optimize strategy parameters using historical data."""
        # Implement parameter optimization logic
        # This is a placeholder for the actual optimization code
        return self.params

    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        required_params = self.get_required_parameters()
        for param in required_params:
            if param not in self.params:
                self.logger.error(f"Missing required parameter: {param}")
                return False
        return True

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhanced data preprocessing."""
        # Remove any missing values
        data = data.dropna()
        
        # Ensure data is sorted by time
        if 'time' in data.columns:
            data = data.sort_values('time')
        
        # Remove outliers
        data = self._remove_outliers(data)
        
        # Normalize data
        data = self._normalize_data(data)
        
        return data

    def postprocess_signals(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Enhanced signal postprocessing."""
        # Remove duplicate signals
        signals = signals.drop_duplicates(subset=['time'])
        
        # Apply signal filters
        signals = self._apply_signal_filters(signals)
        
        # Add confidence scores
        signals['confidence'] = signals.apply(
            lambda x: self.calculate_strategy_confidence(
                pd.DataFrame([x])
            ),
            axis=1
        )
        
        # Add market regime
        signals['market_regime'] = signals.apply(
            lambda x: self.analyze_market_regime(
                pd.DataFrame([x])
            ),
            axis=1
        )
        
        return signals

    def update_performance_metrics(self, trade_result: Dict) -> None:
        """Enhanced performance metrics update."""
        # Update basic metrics
        self.performance_metrics['total_trades'] += 1
        
        if trade_result['profit'] > 0:
            self.performance_metrics['winning_trades'] += 1
            self.performance_metrics['total_profit'] += trade_result['profit']
            self.performance_metrics['current_consecutive_wins'] += 1
            self.performance_metrics['current_consecutive_losses'] = 0
        else:
            self.performance_metrics['losing_trades'] += 1
            self.performance_metrics['total_loss'] += abs(trade_result['profit'])
            self.performance_metrics['current_consecutive_losses'] += 1
            self.performance_metrics['current_consecutive_wins'] = 0

        # Update consecutive wins/losses
        self.performance_metrics['max_consecutive_wins'] = max(
            self.performance_metrics['max_consecutive_wins'],
            self.performance_metrics['current_consecutive_wins']
        )
        self.performance_metrics['max_consecutive_losses'] = max(
            self.performance_metrics['max_consecutive_losses'],
            self.performance_metrics['current_consecutive_losses']
        )

        # Update win rate and profit factor
        self._update_win_rate_and_profit_factor()
        
        # Update risk-adjusted returns
        self._update_risk_adjusted_returns(trade_result)
        
        # Update market regime accuracy
        self._update_market_regime_accuracy(trade_result)
        
        # Update strategy confidence
        self.performance_metrics['strategy_confidence'] = self.strategy_confidence

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate price volatility."""
        returns = data['close'].pct_change()
        return returns.std() * np.sqrt(252)  # Annualized volatility

    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength using ADX or similar indicator."""
        # Implement trend strength calculation
        return 0.5  # Placeholder

    def _calculate_market_efficiency(self, data: pd.DataFrame) -> float:
        """Calculate market efficiency ratio."""
        # Implement market efficiency calculation
        return 0.5  # Placeholder

    def _calculate_signal_strength(self, data: pd.DataFrame) -> float:
        """Calculate the strength of trading signals."""
        # Implement signal strength calculation
        return 0.5  # Placeholder

    def _calculate_indicator_agreement(self, data: pd.DataFrame) -> float:
        """Calculate agreement between different indicators."""
        # Implement indicator agreement calculation
        return 0.5  # Placeholder

    def _calculate_market_alignment(self, data: pd.DataFrame) -> float:
        """Calculate alignment with market conditions."""
        # Implement market alignment calculation
        return 0.5  # Placeholder

    def _calculate_historical_accuracy(self) -> float:
        """Calculate historical accuracy of the strategy."""
        if self.performance_metrics['total_trades'] == 0:
            return 0.5
        return self.performance_metrics['win_rate']

    def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers from the data."""
        for column in ['open', 'high', 'low', 'close']:
            if column in data.columns:
                z_scores = np.abs(stats.zscore(data[column]))
                data = data[z_scores < 3]  # Remove points more than 3 standard deviations
        return data

    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize the data using StandardScaler."""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = self.scaler.fit_transform(data[numeric_columns])
        return data

    def _apply_signal_filters(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Apply filters to trading signals."""
        # Implement signal filtering logic
        return signals

    def _update_win_rate_and_profit_factor(self) -> None:
        """Update win rate and profit factor metrics."""
        total_trades = self.performance_metrics['total_trades']
        if total_trades > 0:
            self.performance_metrics['win_rate'] = (
                self.performance_metrics['winning_trades'] / total_trades
            )
        
        if self.performance_metrics['total_loss'] > 0:
            self.performance_metrics['profit_factor'] = (
                self.performance_metrics['total_profit'] / 
                self.performance_metrics['total_loss']
            )

    def _update_risk_adjusted_returns(self, trade_result: Dict) -> None:
        """Update risk-adjusted return metrics."""
        # Calculate Sharpe Ratio
        returns = trade_result.get('returns', [])
        if returns:
            excess_returns = np.array(returns) - 0.02/252  # Assuming 2% risk-free rate
            if len(excess_returns) > 1:
                self.performance_metrics['sharpe_ratio'] = (
                    np.mean(excess_returns) / np.std(excess_returns)
                ) * np.sqrt(252)
        
        # Calculate Sortino Ratio
        if returns:
            excess_returns = np.array(returns) - 0.02/252
            downside_returns = excess_returns[excess_returns < 0]
            if len(downside_returns) > 1:
                self.performance_metrics['sortino_ratio'] = (
                    np.mean(excess_returns) / np.std(downside_returns)
                ) * np.sqrt(252)

    def _update_market_regime_accuracy(self, trade_result: Dict) -> None:
        """Update market regime prediction accuracy."""
        predicted_regime = trade_result.get('predicted_regime')
        actual_regime = trade_result.get('actual_regime')
        
        if predicted_regime and actual_regime:
            if predicted_regime == actual_regime:
                self.performance_metrics['market_regime_accuracy'] = (
                    self.performance_metrics['market_regime_accuracy'] * 0.9 + 0.1
                )
            else:
                self.performance_metrics['market_regime_accuracy'] *= 0.9

    def get_required_parameters(self) -> List[str]:
        """Get list of required parameters for the strategy."""
        return []

    def get_strategy_info(self) -> Dict:
        """Get enhanced strategy information."""
        return {
            'name': self.name,
            'parameters': self.params,
            'performance': self.performance_metrics,
            'market_regime': self.market_regime,
            'strategy_confidence': self.strategy_confidence,
            'feature_importance': self.feature_importance
        }

    def reset_performance_metrics(self) -> None:
        """Reset all performance metrics to initial values."""
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'risk_reward_ratio': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'current_consecutive_wins': 0,
            'current_consecutive_losses': 0,
            'market_regime_accuracy': 0.0,
            'strategy_confidence': 0.0
        } 