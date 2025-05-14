import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional
import logging
import joblib
from datetime import datetime
import os

class MarketClassifier:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        self.model = None
        self.last_training_time = None
        self.model_path = config['model_path']
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_path, exist_ok=True)

    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for market classification."""
        # Calculate technical indicators
        features = pd.DataFrame()
        
        # Price action features
        features['returns'] = data['close'].pct_change()
        features['volatility'] = features['returns'].rolling(window=20).std()
        features['range'] = (data['high'] - data['low']) / data['close']
        
        # Trend features
        features['sma_20'] = data['close'].rolling(window=20).mean() / data['close']
        features['sma_50'] = data['close'].rolling(window=50).mean() / data['close']
        features['sma_200'] = data['close'].rolling(window=200).mean() / data['close']
        
        # Volume features
        if 'volume' in data.columns:
            features['volume_ma'] = data['volume'].rolling(window=20).mean() / data['volume']
            features['volume_std'] = data['volume'].rolling(window=20).std() / data['volume']
        
        # Momentum features
        features['rsi'] = self._calculate_rsi(data['close'])
        features['macd'] = self._calculate_macd(data['close'])
        
        # Volatility features
        features['atr'] = self._calculate_atr(data)
        features['bollinger_width'] = self._calculate_bollinger_width(data)
        
        # Remove any missing values
        features = features.dropna()
        
        return features

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD indicator."""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        return exp1 - exp2

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def _calculate_bollinger_width(self, data: pd.DataFrame, period: int = 20, std: int = 2) -> pd.Series:
        """Calculate Bollinger Bands width."""
        sma = data['close'].rolling(window=period).mean()
        std_dev = data['close'].rolling(window=period).std()
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        return (upper_band - lower_band) / sma

    def train(self, data: pd.DataFrame, method: str = 'kmeans') -> None:
        """Train the market classifier."""
        # Prepare features
        features = self.prepare_features(data)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Apply PCA
        pca_features = self.pca.fit_transform(scaled_features)
        
        # Train clustering model
        if method == 'kmeans':
            self.model = KMeans(
                n_clusters=self.config['n_clusters'],
                random_state=42
            )
        else:  # DBSCAN
            self.model = DBSCAN(
                eps=0.5,
                min_samples=self.config['min_samples']
            )
        
        self.model.fit(pca_features)
        self.last_training_time = datetime.now()
        
        # Save model
        self._save_model()

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict market conditions for new data."""
        if self.model is None:
            self.logger.error("Model not trained")
            return np.array([])
        
        # Prepare features
        features = self.prepare_features(data)
        
        # Scale features
        scaled_features = self.scaler.transform(features)
        
        # Apply PCA
        pca_features = self.pca.transform(scaled_features)
        
        # Predict clusters
        return self.model.predict(pca_features)

    def get_cluster_characteristics(self, data: pd.DataFrame) -> Dict:
        """Get characteristics of each market condition cluster."""
        if self.model is None:
            return {}
        
        predictions = self.predict(data)
        features = self.prepare_features(data)
        
        characteristics = {}
        for cluster in range(self.config['n_clusters']):
            cluster_data = features[predictions == cluster]
            
            characteristics[cluster] = {
                'volatility': cluster_data['volatility'].mean(),
                'trend_strength': abs(cluster_data['sma_20'] - cluster_data['sma_200']).mean(),
                'momentum': cluster_data['rsi'].mean(),
                'volume_profile': cluster_data['volume_ma'].mean() if 'volume_ma' in cluster_data.columns else 0,
                'sample_size': len(cluster_data)
            }
        
        return characteristics

    def _save_model(self) -> None:
        """Save the trained model and scaler."""
        if self.model is None:
            return
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'pca': self.pca,
            'last_training_time': self.last_training_time
        }
        
        joblib.dump(model_data, os.path.join(self.model_path, 'market_classifier.joblib'))

    def load_model(self) -> bool:
        """Load the trained model and scaler."""
        try:
            model_data = joblib.load(os.path.join(self.model_path, 'market_classifier.joblib'))
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.pca = model_data['pca']
            self.last_training_time = model_data['last_training_time']
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            return False

    def should_retrain(self) -> bool:
        """Check if the model should be retrained based on time interval."""
        if self.last_training_time is None:
            return True
        
        hours_since_training = (datetime.now() - self.last_training_time).total_seconds() / 3600
        return hours_since_training >= self.config['retrain_interval'] 