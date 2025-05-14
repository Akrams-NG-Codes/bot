import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging
from datetime import datetime
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from config.config import ML_CONFIG

class StrategySelector:
    def __init__(self, config: Dict = None):
        self.config = config or ML_CONFIG
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.scaler = StandardScaler()
        self.last_training_time = None
        self.model_path = self.config['model_path']
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_path, exist_ok=True)

    def prepare_training_data(
        self,
        market_data: pd.DataFrame,
        strategy_performance: Dict[str, pd.DataFrame]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for strategy selection model."""
        features = []
        labels = []
        
        # Get market conditions for each time period
        market_conditions = self._get_market_conditions(market_data)
        
        # For each time period, determine the best performing strategy
        for timestamp in market_conditions.index:
            if timestamp not in strategy_performance:
                continue
                
            # Get performance metrics for each strategy at this timestamp
            strategy_metrics = {}
            for strategy_name, performance in strategy_performance.items():
                if timestamp in performance.index:
                    strategy_metrics[strategy_name] = {
                        'profit': performance.loc[timestamp, 'profit'],
                        'win_rate': performance.loc[timestamp, 'win_rate'],
                        'risk_reward': performance.loc[timestamp, 'risk_reward_ratio']
                    }
            
            if not strategy_metrics:
                continue
            
            # Find the best performing strategy
            best_strategy = max(
                strategy_metrics.items(),
                key=lambda x: (
                    x[1]['profit'] * 0.4 +  # 40% weight on profit
                    x[1]['win_rate'] * 0.3 +  # 30% weight on win rate
                    x[1]['risk_reward'] * 0.3  # 30% weight on risk/reward
                )
            )[0]
            
            # Add features and label
            features.append(market_conditions.loc[timestamp].values)
            labels.append(best_strategy)
        
        return np.array(features), np.array(labels)

    def _get_market_conditions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract market condition features."""
        conditions = pd.DataFrame()
        
        # Trend features
        conditions['trend'] = (data['close'] - data['close'].shift(20)) / data['close'].shift(20)
        conditions['trend_strength'] = abs(conditions['trend'])
        
        # Volatility features
        conditions['volatility'] = data['close'].pct_change().rolling(window=20).std()
        conditions['range'] = (data['high'] - data['low']) / data['close']
        
        # Volume features
        if 'volume' in data.columns:
            conditions['volume_trend'] = data['volume'].pct_change(20)
            conditions['volume_volatility'] = data['volume'].rolling(window=20).std() / data['volume'].rolling(window=20).mean()
        
        # Momentum features
        conditions['rsi'] = self._calculate_rsi(data['close'])
        conditions['macd'] = self._calculate_macd(data['close'])
        
        # Remove any missing values
        conditions = conditions.dropna()
        
        return conditions

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

    def train(
        self,
        market_data: pd.DataFrame,
        strategy_performance: Dict[str, pd.DataFrame]
    ) -> None:
        """Train the strategy selection model."""
        # Prepare training data
        X, y = self.prepare_training_data(market_data, strategy_performance)
        
        if len(X) == 0 or len(y) == 0:
            self.logger.error("No training data available")
            return
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model.fit(X_scaled, y)
        
        self.last_training_time = datetime.now()
        
        # Save model
        self._save_model()

    def select_strategy(self, market_data: pd.DataFrame) -> str:
        """Select the best strategy for current market conditions."""
        if self.model is None:
            self.logger.error("Model not trained")
            return None
        
        # Get current market conditions
        conditions = self._get_market_conditions(market_data)
        if conditions.empty:
            return None
        
        # Get latest market conditions
        latest_conditions = conditions.iloc[-1].values.reshape(1, -1)
        
        # Scale features
        latest_conditions_scaled = self.scaler.transform(latest_conditions)
        
        # Predict best strategy
        return self.model.predict(latest_conditions_scaled)[0]

    def get_strategy_probabilities(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Get probability distribution over strategies for current market conditions."""
        if self.model is None:
            return {}
        
        # Get current market conditions
        conditions = self._get_market_conditions(market_data)
        if conditions.empty:
            return {}
        
        # Get latest market conditions
        latest_conditions = conditions.iloc[-1].values.reshape(1, -1)
        
        # Scale features
        latest_conditions_scaled = self.scaler.transform(latest_conditions)
        
        # Get strategy probabilities
        probabilities = self.model.predict_proba(latest_conditions_scaled)[0]
        
        return dict(zip(self.model.classes_, probabilities))

    def _save_model(self) -> None:
        """Save the trained model and scaler."""
        if self.model is None:
            return
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'last_training_time': self.last_training_time
        }
        
        joblib.dump(model_data, os.path.join(self.model_path, 'strategy_selector.joblib'))

    def load_model(self) -> bool:
        """Load the trained model and scaler."""
        try:
            model_data = joblib.load(os.path.join(self.model_path, 'strategy_selector.joblib'))
            self.model = model_data['model']
            self.scaler = model_data['scaler']
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

    def evaluate_performance(
        self,
        market_data: pd.DataFrame,
        strategy_performance: Dict[str, pd.DataFrame]
    ) -> Dict:
        """Evaluate the performance of strategy selection."""
        if self.model is None:
            return {}
        
        # Prepare test data
        X, y_true = self.prepare_training_data(market_data, strategy_performance)
        
        if len(X) == 0 or len(y_true) == 0:
            return {}
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get predictions
        y_pred = self.model.predict(X_scaled)
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y_true)
        
        # Calculate strategy-wise accuracy
        strategy_accuracy = {}
        for strategy in np.unique(y_true):
            mask = y_true == strategy
            strategy_accuracy[strategy] = np.mean(y_pred[mask] == y_true[mask])
        
        return {
            'overall_accuracy': accuracy,
            'strategy_accuracy': strategy_accuracy
        } 