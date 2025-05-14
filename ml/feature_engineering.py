import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime, timedelta
import ta
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import MetaTrader5 as mt5

class FeatureEngineer:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.scalers = {}
        self.pca = None
        self.feature_importance = {}
        
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive feature set from price data."""
        try:
            # Technical indicators
            data = self._add_technical_indicators(data)
            
            # Price action features
            data = self._add_price_action_features(data)
            
            # Volatility features
            data = self._add_volatility_features(data)
            
            # Volume features
            data = self._add_volume_features(data)
            
            # Market regime features
            data = self._add_market_regime_features(data)
            
            # Time-based features
            data = self._add_time_features(data)
            
            # Feature interactions
            data = self._add_feature_interactions(data)
            
            # Clean and prepare features
            data = self._prepare_features(data)
            
            return data
        except Exception as e:
            self.logger.error(f"Error generating features: {str(e)}")
            return pd.DataFrame()

    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataset."""
        try:
            # Trend indicators
            data['sma_20'] = ta.trend.SMAIndicator(data['close'], window=20).sma_indicator()
            data['sma_50'] = ta.trend.SMAIndicator(data['close'], window=50).sma_indicator()
            data['ema_20'] = ta.trend.EMAIndicator(data['close'], window=20).ema_indicator()
            data['macd'] = ta.trend.MACD(data['close']).macd()
            data['macd_signal'] = ta.trend.MACD(data['close']).macd_signal()
            data['adx'] = ta.trend.ADXIndicator(data['high'], data['low'], data['close']).adx()
            
            # Momentum indicators
            data['rsi'] = ta.momentum.RSIIndicator(data['close']).rsi()
            data['stoch'] = ta.momentum.StochasticOscillator(data['high'], data['low'], data['close']).stoch()
            data['williams_r'] = ta.momentum.WilliamsRIndicator(data['high'], data['low'], data['close']).williams_r()
            
            # Volatility indicators
            data['bb_upper'] = ta.volatility.BollingerBands(data['close']).bollinger_hband()
            data['bb_lower'] = ta.volatility.BollingerBands(data['close']).bollinger_lband()
            data['atr'] = ta.volatility.AverageTrueRange(data['high'], data['low'], data['close']).average_true_range()
            
            # Volume indicators
            data['obv'] = ta.volume.OnBalanceVolumeIndicator(data['close'], data['volume']).on_balance_volume()
            data['mfi'] = ta.volume.MFIIndicator(data['high'], data['low'], data['close'], data['volume']).money_flow_index()
            
            return data
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {str(e)}")
            return data

    def _add_price_action_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add price action features."""
        try:
            # Price changes
            data['returns'] = data['close'].pct_change()
            data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
            
            # Price ranges
            data['daily_range'] = data['high'] - data['low']
            data['daily_range_pct'] = data['daily_range'] / data['close']
            
            # Candlestick patterns
            data['body_size'] = abs(data['close'] - data['open'])
            data['upper_shadow'] = data['high'] - data[['open', 'close']].max(axis=1)
            data['lower_shadow'] = data[['open', 'close']].min(axis=1) - data['low']
            
            # Support/Resistance levels
            data['support_level'] = data['low'].rolling(window=20).min()
            data['resistance_level'] = data['high'].rolling(window=20).max()
            
            return data
        except Exception as e:
            self.logger.error(f"Error adding price action features: {str(e)}")
            return data

    def _add_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features."""
        try:
            # Historical volatility
            data['volatility_20'] = data['returns'].rolling(window=20).std() * np.sqrt(252)
            data['volatility_50'] = data['returns'].rolling(window=50).std() * np.sqrt(252)
            
            # Parkinson volatility
            data['parkinson_vol'] = np.sqrt(
                (1 / (4 * np.log(2))) * 
                ((np.log(data['high'] / data['low']) ** 2).rolling(window=20).mean() * 252)
            )
            
            # Garman-Klass volatility
            data['garman_klass_vol'] = np.sqrt(
                (0.5 * np.log(data['high'] / data['low']) ** 2 - 
                 (2 * np.log(2) - 1) * 
                 (np.log(data['close'] / data['open']) ** 2)).rolling(window=20).mean() * 252
            )
            
            return data
        except Exception as e:
            self.logger.error(f"Error adding volatility features: {str(e)}")
            return data

    def _add_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        try:
            # Volume changes
            data['volume_change'] = data['volume'].pct_change()
            data['volume_ma_20'] = data['volume'].rolling(window=20).mean()
            
            # Volume-price relationship
            data['volume_price_trend'] = data['volume'] * data['returns']
            data['volume_volatility'] = data['volume'].rolling(window=20).std()
            
            # Volume profile
            data['volume_ratio'] = data['volume'] / data['volume_ma_20']
            data['volume_trend'] = data['volume'].rolling(window=20).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0]
            )
            
            return data
        except Exception as e:
            self.logger.error(f"Error adding volume features: {str(e)}")
            return data

    def _add_market_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add market regime features."""
        try:
            # Trend strength
            data['trend_strength'] = abs(data['sma_20'] - data['sma_50']) / data['sma_50']
            
            # Market regime indicators
            data['regime_volatility'] = data['volatility_20'] / data['volatility_50']
            data['regime_trend'] = (data['close'] - data['sma_50']) / data['sma_50']
            
            # Market conditions
            data['market_condition'] = np.where(
                (data['trend_strength'] > 0.02) & (data['adx'] > 25),
                'trending',
                np.where(
                    data['volatility_20'] > data['volatility_20'].rolling(window=50).mean(),
                    'volatile',
                    'ranging'
                )
            )
            
            return data
        except Exception as e:
            self.logger.error(f"Error adding market regime features: {str(e)}")
            return data

    def _add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        try:
            # Time features
            data['hour'] = data.index.hour
            data['day_of_week'] = data.index.dayofweek
            data['month'] = data.index.month
            data['is_weekend'] = data.index.dayofweek.isin([5, 6]).astype(int)
            
            # Time-based volatility
            data['hourly_volatility'] = data.groupby('hour')['returns'].transform('std')
            data['daily_volatility'] = data.groupby('day_of_week')['returns'].transform('std')
            
            # Time-based volume
            data['hourly_volume'] = data.groupby('hour')['volume'].transform('mean')
            data['daily_volume'] = data.groupby('day_of_week')['volume'].transform('mean')
            
            return data
        except Exception as e:
            self.logger.error(f"Error adding time features: {str(e)}")
            return data

    def _add_feature_interactions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add feature interaction terms."""
        try:
            # Price-Volume interactions
            data['price_volume_corr'] = data['returns'].rolling(window=20).corr(data['volume_change'])
            
            # Volatility-Trend interactions
            data['volatility_trend'] = data['volatility_20'] * data['trend_strength']
            
            # Technical indicator interactions
            data['rsi_macd'] = data['rsi'] * data['macd']
            data['bb_rsi'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower']) * data['rsi']
            
            return data
        except Exception as e:
            self.logger.error(f"Error adding feature interactions: {str(e)}")
            return data

    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model input."""
        try:
            # Handle missing values
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            # Remove infinite values
            data = data.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Scale features
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col not in self.scalers:
                    self.scalers[col] = RobustScaler()
                    data[col] = self.scalers[col].fit_transform(data[col].values.reshape(-1, 1))
                else:
                    data[col] = self.scalers[col].transform(data[col].values.reshape(-1, 1))
            
            # Apply PCA if configured
            if self.config.get('use_pca', False):
                if self.pca is None:
                    self.pca = PCA(n_components=self.config.get('pca_components', 10))
                    data_pca = self.pca.fit_transform(data[numeric_columns])
                else:
                    data_pca = self.pca.transform(data[numeric_columns])
                
                # Add PCA components to data
                for i in range(data_pca.shape[1]):
                    data[f'pca_{i+1}'] = data_pca[:, i]
            
            return data
        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}")
            return data

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return self.feature_importance

    def update_feature_importance(self, importance_scores: Dict[str, float]):
        """Update feature importance scores."""
        self.feature_importance = importance_scores

    def get_feature_list(self) -> List[str]:
        """Get list of all features."""
        return list(self.feature_importance.keys())

    def get_top_features(self, n: int = 10) -> List[str]:
        """Get top N most important features."""
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [f[0] for f in sorted_features[:n]] 