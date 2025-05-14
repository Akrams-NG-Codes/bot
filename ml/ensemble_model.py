import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime, timedelta
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.model_selection import TimeSeriesSplit
import joblib
import json
from pathlib import Path

class EnsembleModel:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.ensemble = None
        self.models = {}
        self.weights = config.get('ensemble', {}).get('weights', {})
        self.metrics = {}
        
    def build_ensemble(self, models: Dict):
        """Build ensemble from trained models."""
        try:
            self.models = models
            
            if not self.config.get('ensemble', {}).get('enabled', True):
                self.logger.info("Ensemble disabled, using best model only")
                return
            
            method = self.config.get('ensemble', {}).get('method', 'voting')
            
            if method == 'voting':
                self._build_voting_ensemble()
            elif method == 'stacking':
                self._build_stacking_ensemble()
            else:
                self.logger.warning(f"Unknown ensemble method: {method}, using voting")
                self._build_voting_ensemble()
                
        except Exception as e:
            self.logger.error(f"Error building ensemble: {str(e)}")

    def _build_voting_ensemble(self):
        """Build voting ensemble classifier."""
        try:
            estimators = []
            for name, model in self.models.items():
                if name in self.weights:
                    estimators.append((name, model))
            
            self.ensemble = VotingClassifier(
                estimators=estimators,
                voting='soft',
                weights=[self.weights[name] for name, _ in estimators]
            )
            
            self.logger.info("Voting ensemble built successfully")
        except Exception as e:
            self.logger.error(f"Error building voting ensemble: {str(e)}")

    def _build_stacking_ensemble(self):
        """Build stacking ensemble classifier."""
        try:
            estimators = []
            for name, model in self.models.items():
                if name in self.weights:
                    estimators.append((name, model))
            
            self.ensemble = StackingClassifier(
                estimators=estimators,
                final_estimator=self.models.get('random_forest'),
                cv=TimeSeriesSplit(n_splits=5)
            )
            
            self.logger.info("Stacking ensemble built successfully")
        except Exception as e:
            self.logger.error(f"Error building stacking ensemble: {str(e)}")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit ensemble model."""
        try:
            if self.ensemble is None:
                self.logger.warning("No ensemble built, using best model")
                return
            
            self.ensemble.fit(X, y)
            self.logger.info("Ensemble model fitted successfully")
        except Exception as e:
            self.logger.error(f"Error fitting ensemble: {str(e)}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using ensemble."""
        try:
            if self.ensemble is None:
                self.logger.warning("No ensemble built, using best model")
                return self.models.get('best_model', None).predict(X)
            
            return self.ensemble.predict(X)
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            return np.array([])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities using ensemble."""
        try:
            if self.ensemble is None:
                self.logger.warning("No ensemble built, using best model")
                return self.models.get('best_model', None).predict_proba(X)
            
            return self.ensemble.predict_proba(X)
        except Exception as e:
            self.logger.error(f"Error getting prediction probabilities: {str(e)}")
            return np.array([])

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Evaluate ensemble performance."""
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            y_pred = self.predict(X)
            
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred),
                'recall': recall_score(y, y_pred),
                'f1': f1_score(y, y_pred)
            }
            
            self.metrics = metrics
            return metrics
        except Exception as e:
            self.logger.error(f"Error evaluating ensemble: {str(e)}")
            return {}

    def get_model_weights(self) -> Dict[str, float]:
        """Get current model weights."""
        return self.weights

    def update_weights(self, new_weights: Dict[str, float]):
        """Update model weights."""
        try:
            self.weights.update(new_weights)
            if self.ensemble is not None:
                self._build_voting_ensemble()  # Rebuild ensemble with new weights
            self.logger.info("Model weights updated successfully")
        except Exception as e:
            self.logger.error(f"Error updating weights: {str(e)}")

    def save_ensemble(self, path: str):
        """Save ensemble model and configuration."""
        try:
            if self.ensemble is None:
                raise ValueError("No ensemble model to save")
            
            # Create directory if it doesn't exist
            Path(path).mkdir(parents=True, exist_ok=True)
            
            # Save ensemble
            ensemble_path = Path(path) / 'ensemble.joblib'
            joblib.dump(self.ensemble, ensemble_path)
            
            # Save configuration
            config_path = Path(path) / 'ensemble_config.json'
            config = {
                'weights': self.weights,
                'metrics': self.metrics,
                'method': self.config.get('ensemble', {}).get('method', 'voting')
            }
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.logger.info(f"Ensemble saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving ensemble: {str(e)}")

    def load_ensemble(self, path: str):
        """Load saved ensemble model and configuration."""
        try:
            # Load ensemble
            ensemble_path = Path(path) / 'ensemble.joblib'
            self.ensemble = joblib.load(ensemble_path)
            
            # Load configuration
            config_path = Path(path) / 'ensemble_config.json'
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            self.weights = config['weights']
            self.metrics = config['metrics']
            
            self.logger.info(f"Ensemble loaded from {path}")
        except Exception as e:
            self.logger.error(f"Error loading ensemble: {str(e)}")

    def get_metrics(self) -> Dict:
        """Get ensemble performance metrics."""
        return self.metrics 