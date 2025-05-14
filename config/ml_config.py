ML_CONFIG = {
    # Feature engineering settings
    'feature_engineering': {
        'use_pca': True,
        'pca_components': 10,
        'feature_groups': {
            'technical': True,
            'price_action': True,
            'volatility': True,
            'volume': True,
            'market_regime': True,
            'time': True,
            'interactions': True
        },
        'scaling': {
            'method': 'robust',  # 'standard' or 'robust'
            'handle_outliers': True
        }
    },
    
    # Model training settings
    'model_training': {
        'cv_splits': 5,
        'use_grid_search': True,
        'test_size': 0.2,
        'random_state': 42,
        'models': {
            'random_forest': True,
            'gradient_boosting': True,
            'xgboost': True,
            'lightgbm': True,
            'neural_network': True,
            'svm': True
        }
    },
    
    # Model hyperparameters
    'param_grids': {
        'RandomForestClassifier': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10]
        },
        'GradientBoostingClassifier': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        'XGBClassifier': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        },
        'LGBMClassifier': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'num_leaves': [31, 63, 127]
        },
        'MLPClassifier': {
            'hidden_layer_sizes': [(50,), (100,), (100, 50)],
            'activation': ['relu', 'tanh'],
            'learning_rate_init': [0.001, 0.01, 0.1]
        },
        'SVC': {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['rbf', 'poly'],
            'gamma': ['scale', 'auto']
        }
    },
    
    # Model evaluation settings
    'evaluation': {
        'metrics': [
            'accuracy',
            'precision',
            'recall',
            'f1',
            'roc_auc',
            'confusion_matrix'
        ],
        'threshold': 0.5,
        'cross_validation': {
            'method': 'time_series',
            'n_splits': 5
        }
    },
    
    # Feature selection settings
    'feature_selection': {
        'method': 'importance',  # 'importance', 'recursive', 'lasso'
        'n_features': 20,
        'importance_threshold': 0.01
    },
    
    # Model persistence settings
    'persistence': {
        'save_path': 'models',
        'save_format': 'joblib',
        'save_metrics': True,
        'save_feature_importance': True
    },
    
    # Ensemble settings
    'ensemble': {
        'enabled': True,
        'method': 'voting',  # 'voting', 'stacking', 'bagging'
        'weights': {
            'random_forest': 1.0,
            'gradient_boosting': 1.0,
            'xgboost': 1.0,
            'lightgbm': 1.0,
            'neural_network': 1.0,
            'svm': 1.0
        }
    },
    
    # Model monitoring settings
    'monitoring': {
        'performance_threshold': 0.6,
        'retraining_interval': 24,  # hours
        'drift_detection': {
            'enabled': True,
            'window_size': 1000,
            'threshold': 0.1
        }
    },
    
    # Data preprocessing settings
    'preprocessing': {
        'handle_missing': True,
        'handle_outliers': True,
        'normalize': True,
        'encode_categorical': True,
        'window_size': 20
    },
    
    # Logging settings
    'logging': {
        'level': 'INFO',
        'save_predictions': True,
        'save_metrics': True,
        'log_path': 'logs/ml'
    }
} 