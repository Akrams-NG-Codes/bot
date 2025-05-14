import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
import joblib
import json
from pathlib import Path

class ModelTrainer:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.model_metrics = {}
        self.best_model = None
        self.feature_importance = {}
        
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train multiple models and select the best performer."""
        try:
            # Initialize models
            self._initialize_models()
            
            # Prepare cross-validation
            tscv = TimeSeriesSplit(n_splits=self.config.get('cv_splits', 5))
            
            # Train and evaluate each model
            for name, model in self.models.items():
                self.logger.info(f"Training {name}...")
                
                # Perform grid search if configured
                if self.config.get('use_grid_search', False):
                    model = self._perform_grid_search(model, X, y, tscv)
                    self.models[name] = model
                
                # Train model
                model.fit(X, y)
                
                # Evaluate model
                metrics = self._evaluate_model(model, X, y, tscv)
                self.model_metrics[name] = metrics
                
                # Update feature importance
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = dict(zip(X.columns, model.feature_importances_))
            
            # Select best model
            self._select_best_model()
            
            return self.model_metrics
        except Exception as e:
            self.logger.error(f"Error training models: {str(e)}")
            return {}

    def _initialize_models(self):
        """Initialize model configurations."""
        try:
            # Random Forest
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=self.config.get('rf_n_estimators', 100),
                max_depth=self.config.get('rf_max_depth', 10),
                min_samples_split=self.config.get('rf_min_samples_split', 2),
                random_state=42
            )
            
            # Gradient Boosting
            self.models['gradient_boosting'] = GradientBoostingClassifier(
                n_estimators=self.config.get('gb_n_estimators', 100),
                learning_rate=self.config.get('gb_learning_rate', 0.1),
                max_depth=self.config.get('gb_max_depth', 3),
                random_state=42
            )
            
            # XGBoost
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=self.config.get('xgb_n_estimators', 100),
                learning_rate=self.config.get('xgb_learning_rate', 0.1),
                max_depth=self.config.get('xgb_max_depth', 3),
                random_state=42
            )
            
            # LightGBM
            self.models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=self.config.get('lgb_n_estimators', 100),
                learning_rate=self.config.get('lgb_learning_rate', 0.1),
                max_depth=self.config.get('lgb_max_depth', 3),
                random_state=42
            )
            
            # Neural Network
            self.models['neural_network'] = MLPClassifier(
                hidden_layer_sizes=self.config.get('nn_hidden_layers', (100, 50)),
                activation=self.config.get('nn_activation', 'relu'),
                learning_rate_init=self.config.get('nn_learning_rate', 0.001),
                max_iter=self.config.get('nn_max_iter', 1000),
                random_state=42
            )
            
            # SVM
            self.models['svm'] = SVC(
                kernel=self.config.get('svm_kernel', 'rbf'),
                C=self.config.get('svm_c', 1.0),
                probability=True,
                random_state=42
            )
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")

    def _perform_grid_search(self, model, X: pd.DataFrame, y: pd.Series, 
                           tscv: TimeSeriesSplit) -> object:
        """Perform grid search for hyperparameter optimization."""
        try:
            param_grid = self.config.get('param_grids', {}).get(type(model).__name__, {})
            if not param_grid:
                return model
            
            grid_search = GridSearchCV(
                model,
                param_grid,
                cv=tscv,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X, y)
            self.logger.info(f"Best parameters: {grid_search.best_params_}")
            
            return grid_search.best_estimator_
        except Exception as e:
            self.logger.error(f"Error performing grid search: {str(e)}")
            return model

    def _evaluate_model(self, model: object, X: pd.DataFrame, y: pd.Series,
                       tscv: TimeSeriesSplit) -> Dict:
        """Evaluate model performance using cross-validation."""
        try:
            metrics = {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1': []
            }
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Train on this fold
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                metrics['accuracy'].append(accuracy_score(y_test, y_pred))
                metrics['precision'].append(precision_score(y_test, y_pred))
                metrics['recall'].append(recall_score(y_test, y_pred))
                metrics['f1'].append(f1_score(y_test, y_pred))
            
            # Calculate average metrics
            return {
                metric: np.mean(scores) for metric, scores in metrics.items()
            }
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            return {}

    def _select_best_model(self):
        """Select the best performing model based on F1 score."""
        try:
            best_score = -1
            best_model_name = None
            
            for name, metrics in self.model_metrics.items():
                if metrics['f1'] > best_score:
                    best_score = metrics['f1']
                    best_model_name = name
            
            if best_model_name:
                self.best_model = self.models[best_model_name]
                self.logger.info(f"Selected {best_model_name} as best model with F1 score: {best_score:.4f}")
        except Exception as e:
            self.logger.error(f"Error selecting best model: {str(e)}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the best model."""
        try:
            if self.best_model is None:
                raise ValueError("No model has been trained yet")
            
            return self.best_model.predict(X)
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            return np.array([])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities using the best model."""
        try:
            if self.best_model is None:
                raise ValueError("No model has been trained yet")
            
            return self.best_model.predict_proba(X)
        except Exception as e:
            self.logger.error(f"Error getting prediction probabilities: {str(e)}")
            return np.array([])

    def save_model(self, path: str):
        """Save the best model and its metrics."""
        try:
            if self.best_model is None:
                raise ValueError("No model has been trained yet")
            
            # Create directory if it doesn't exist
            Path(path).mkdir(parents=True, exist_ok=True)
            
            # Save model
            model_path = Path(path) / 'model.joblib'
            joblib.dump(self.best_model, model_path)
            
            # Save metrics
            metrics_path = Path(path) / 'metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(self.model_metrics, f, indent=2)
            
            # Save feature importance
            importance_path = Path(path) / 'feature_importance.json'
            with open(importance_path, 'w') as f:
                json.dump(self.feature_importance, f, indent=2)
            
            self.logger.info(f"Model saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")

    def load_model(self, path: str):
        """Load a saved model and its metrics."""
        try:
            # Load model
            model_path = Path(path) / 'model.joblib'
            self.best_model = joblib.load(model_path)
            
            # Load metrics
            metrics_path = Path(path) / 'metrics.json'
            with open(metrics_path, 'r') as f:
                self.model_metrics = json.load(f)
            
            # Load feature importance
            importance_path = Path(path) / 'feature_importance.json'
            with open(importance_path, 'r') as f:
                self.feature_importance = json.load(f)
            
            self.logger.info(f"Model loaded from {path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")

    def get_model_metrics(self) -> Dict:
        """Get performance metrics for all models."""
        return self.model_metrics

    def get_feature_importance(self) -> Dict:
        """Get feature importance scores for all models."""
        return self.feature_importance 