import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import MetaTrader5 as mt5
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import ta

class MarketAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.regime_period = config.get('regime_period', 20)  # Days for regime detection
        self.volatility_period = config.get('volatility_period', 14)  # Days for volatility calculation
        self.correlation_period = config.get('correlation_period', 30)  # Days for correlation analysis
        self.regime_threshold = config.get('regime_threshold', 0.7)  # Threshold for regime change
        self.market_data = {}  # Cache for market data

    def analyze_market(self, symbol: str) -> Dict:
        """Perform comprehensive market analysis."""
        # Get market data
        data = self._get_market_data(symbol)
        if data is None:
            return {}
        
        # Detect market regime
        regime = self._detect_market_regime(data)
        
        # Calculate volatility
        volatility = self._calculate_volatility(data)
        
        # Calculate trend strength
        trend = self._calculate_trend_strength(data)
        
        # Calculate support/resistance levels
        levels = self._calculate_support_resistance(data)
        
        return {
            'regime': regime,
            'volatility': volatility,
            'trend': trend,
            'support_resistance': levels,
            'timestamp': datetime.now()
        }

    def _get_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get market data from MT5."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.regime_period)
            
            rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, 
                                       int(start_date.timestamp()),
                                       int(end_date.timestamp()))
            if rates is None:
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Calculate technical indicators
            df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
            df['macd'] = ta.trend.MACD(df['close']).macd()
            df['bb_upper'] = ta.volatility.BollingerBands(df['close']).bollinger_hband()
            df['bb_lower'] = ta.volatility.BollingerBands(df['close']).bollinger_lband()
            
            return df
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {str(e)}")
            return None

    def _detect_market_regime(self, data: pd.DataFrame) -> str:
        """Detect current market regime using clustering."""
        try:
            # Prepare features for regime detection
            features = pd.DataFrame({
                'returns': data['close'].pct_change(),
                'volatility': data['close'].pct_change().rolling(20).std(),
                'rsi': data['rsi'],
                'macd': data['macd']
            }).dropna()
            
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(scaled_features)
            
            # Map clusters to regimes
            regime_map = {
                0: 'ranging',
                1: 'trending',
                2: 'volatile'
            }
            
            # Get current regime
            current_regime = regime_map[clusters[-1]]
            
            return current_regime
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {str(e)}")
            return 'unknown'

    def _calculate_volatility(self, data: pd.DataFrame) -> Dict:
        """Calculate various volatility metrics."""
        try:
            returns = data['close'].pct_change().dropna()
            
            # Calculate different volatility measures
            historical_vol = returns.std() * np.sqrt(252)  # Annualized
            parkinson_vol = np.sqrt(
                (1 / (4 * np.log(2))) * 
                ((np.log(data['high'] / data['low']) ** 2).mean() * 252)
            )
            garman_klass_vol = np.sqrt(
                (0.5 * np.log(data['high'] / data['low']) ** 2 - 
                 (2 * np.log(2) - 1) * 
                 (np.log(data['close'] / data['open']) ** 2)).mean() * 252
            )
            
            return {
                'historical_volatility': historical_vol,
                'parkinson_volatility': parkinson_vol,
                'garman_klass_volatility': garman_klass_vol,
                'volatility_rank': self._calculate_volatility_rank(data)
            }
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {str(e)}")
            return {}

    def _calculate_volatility_rank(self, data: pd.DataFrame) -> float:
        """Calculate volatility rank (0-100) compared to historical data."""
        try:
            # Get longer historical data for ranking
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)  # 1 year of data
            
            historical_rates = mt5.copy_rates_range(data.name, mt5.TIMEFRAME_H1,
                                                  int(start_date.timestamp()),
                                                  int(end_date.timestamp()))
            
            if historical_rates is None:
                return 50.0  # Default to middle rank
            
            historical_data = pd.DataFrame(historical_rates)
            historical_vol = historical_data['close'].pct_change().rolling(20).std()
            
            current_vol = data['close'].pct_change().rolling(20).std().iloc[-1]
            rank = (current_vol > historical_vol).mean() * 100
            
            return rank
        except Exception as e:
            self.logger.error(f"Error calculating volatility rank: {str(e)}")
            return 50.0

    def _calculate_trend_strength(self, data: pd.DataFrame) -> Dict:
        """Calculate trend strength indicators."""
        try:
            # Calculate ADX
            adx = ta.trend.ADXIndicator(data['high'], data['low'], data['close'])
            adx_value = adx.adx().iloc[-1]
            
            # Calculate trend direction
            sma_20 = ta.trend.SMAIndicator(data['close'], window=20).sma_indicator()
            sma_50 = ta.trend.SMAIndicator(data['close'], window=50).sma_indicator()
            
            trend_direction = 'up' if sma_20.iloc[-1] > sma_50.iloc[-1] else 'down'
            
            return {
                'adx': adx_value,
                'trend_direction': trend_direction,
                'trend_strength': 'strong' if adx_value > 25 else 'weak'
            }
        except Exception as e:
            self.logger.error(f"Error calculating trend strength: {str(e)}")
            return {}

    def _calculate_support_resistance(self, data: pd.DataFrame) -> Dict:
        """Calculate support and resistance levels."""
        try:
            # Use recent price action to identify levels
            recent_data = data.tail(100)  # Use last 100 periods
            
            # Find local minima and maxima
            window = 5
            local_min = recent_data['low'].rolling(window=window, center=True).min()
            local_max = recent_data['high'].rolling(window=window, center=True).max()
            
            # Cluster the levels
            levels = pd.concat([
                local_min[local_min == recent_data['low']],
                local_max[local_max == recent_data['high']]
            ]).dropna()
            
            # Use K-means to identify significant levels
            if len(levels) > 0:
                kmeans = KMeans(n_clusters=min(5, len(levels)), random_state=42)
                clusters = kmeans.fit_predict(levels.values.reshape(-1, 1))
                
                # Get cluster centers
                significant_levels = kmeans.cluster_centers_.flatten()
                
                # Separate support and resistance
                current_price = data['close'].iloc[-1]
                support_levels = significant_levels[significant_levels < current_price]
                resistance_levels = significant_levels[significant_levels > current_price]
                
                return {
                    'support_levels': sorted(support_levels, reverse=True),
                    'resistance_levels': sorted(resistance_levels)
                }
            
            return {'support_levels': [], 'resistance_levels': []}
        except Exception as e:
            self.logger.error(f"Error calculating support/resistance: {str(e)}")
            return {'support_levels': [], 'resistance_levels': []}

    def get_correlation_matrix(self, symbols: List[str]) -> pd.DataFrame:
        """Calculate correlation matrix for multiple symbols."""
        try:
            # Get data for all symbols
            data_dict = {}
            for symbol in symbols:
                data = self._get_market_data(symbol)
                if data is not None:
                    data_dict[symbol] = data['close'].pct_change()
            
            # Create correlation matrix
            returns_df = pd.DataFrame(data_dict)
            correlation_matrix = returns_df.corr()
            
            return correlation_matrix
        except Exception as e:
            self.logger.error(f"Error calculating correlation matrix: {str(e)}")
            return pd.DataFrame() 