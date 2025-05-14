import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json
from pathlib import Path
import threading
import queue
import time

class ModelMonitor:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metrics_history = []
        self.drift_history = []
        self.alerts = queue.Queue()
        self.monitoring = False
        self.monitor_thread = None
        self.last_retraining = datetime.now()
        
        # Create directories
        self.metrics_dir = Path('metrics')
        self.metrics_dir.mkdir(exist_ok=True)
        
        # Initialize drift detection
        self.drift_config = config.get('monitoring', {}).get('drift_detection', {})
        self.window_size = self.drift_config.get('window_size', 1000)
        self.drift_threshold = self.drift_config.get('threshold', 0.1)
        
    def start_monitoring(self):
        """Start model monitoring in a separate thread."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.logger.info("Model monitoring started")

    def stop_monitoring(self):
        """Stop model monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.info("Model monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Check if retraining is needed
                self._check_retraining()
                
                # Save current metrics
                self._save_metrics()
                
                # Clean up old metrics
                self._cleanup_old_metrics()
                
                time.sleep(self.config.get('monitoring_interval', 60))
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")

    def update_metrics(self, metrics: Dict):
        """Update model performance metrics."""
        try:
            metrics['timestamp'] = datetime.now().isoformat()
            self.metrics_history.append(metrics)
            
            # Check for performance degradation
            self._check_performance(metrics)
            
            # Check for data drift
            self._check_drift(metrics)
            
            # Keep history within limits
            max_history = self.config.get('max_metrics_history', 1000)
            if len(self.metrics_history) > max_history:
                self.metrics_history = self.metrics_history[-max_history:]
        except Exception as e:
            self.logger.error(f"Error updating metrics: {str(e)}")

    def _check_performance(self, metrics: Dict):
        """Check for performance degradation."""
        try:
            threshold = self.config.get('monitoring', {}).get('performance_threshold', 0.6)
            
            if metrics.get('accuracy', 1.0) < threshold:
                self._generate_alert(
                    'performance_degradation',
                    f"Model accuracy ({metrics['accuracy']:.4f}) below threshold ({threshold})"
                )
        except Exception as e:
            self.logger.error(f"Error checking performance: {str(e)}")

    def _check_drift(self, metrics: Dict):
        """Check for data drift."""
        try:
            if not self.drift_config.get('enabled', True):
                return
            
            if len(self.metrics_history) < self.window_size:
                return
            
            # Calculate drift score
            recent_metrics = self.metrics_history[-self.window_size:]
            drift_score = self._calculate_drift_score(recent_metrics)
            
            # Record drift
            drift_record = {
                'timestamp': datetime.now().isoformat(),
                'drift_score': drift_score
            }
            self.drift_history.append(drift_record)
            
            # Check for significant drift
            if drift_score > self.drift_threshold:
                self._generate_alert(
                    'data_drift',
                    f"Significant data drift detected (score: {drift_score:.4f})"
                )
        except Exception as e:
            self.logger.error(f"Error checking drift: {str(e)}")

    def _calculate_drift_score(self, metrics: List[Dict]) -> float:
        """Calculate drift score from recent metrics."""
        try:
            # Calculate average performance
            avg_performance = np.mean([m.get('accuracy', 0) for m in metrics])
            
            # Calculate performance trend
            performance_trend = np.polyfit(
                range(len(metrics)),
                [m.get('accuracy', 0) for m in metrics],
                1
            )[0]
            
            # Calculate volatility
            performance_volatility = np.std([m.get('accuracy', 0) for m in metrics])
            
            # Combine factors into drift score
            drift_score = (
                abs(performance_trend) * 0.4 +
                performance_volatility * 0.3 +
                (1 - avg_performance) * 0.3
            )
            
            return drift_score
        except Exception as e:
            self.logger.error(f"Error calculating drift score: {str(e)}")
            return 0.0

    def _check_retraining(self):
        """Check if model retraining is needed."""
        try:
            retraining_interval = self.config.get('monitoring', {}).get('retraining_interval', 24)
            hours_since_retraining = (datetime.now() - self.last_retraining).total_seconds() / 3600
            
            if hours_since_retraining >= retraining_interval:
                self._generate_alert(
                    'retraining_needed',
                    f"Model retraining due (last training: {self.last_retraining})"
                )
        except Exception as e:
            self.logger.error(f"Error checking retraining: {str(e)}")

    def _generate_alert(self, alert_type: str, message: str):
        """Generate and queue an alert."""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'message': message,
            'severity': self._get_alert_severity(alert_type)
        }
        self.alerts.put(alert)
        self.logger.warning(f"Alert: {message}")

    def _get_alert_severity(self, alert_type: str) -> str:
        """Get alert severity based on type."""
        severity_map = {
            'performance_degradation': 'warning',
            'data_drift': 'warning',
            'retraining_needed': 'info'
        }
        return severity_map.get(alert_type, 'info')

    def _save_metrics(self):
        """Save current metrics to file."""
        try:
            if not self.metrics_history:
                return
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = self.metrics_dir / f'metrics_{timestamp}.json'
            
            with open(file_path, 'w') as f:
                json.dump(self.metrics_history[-1], f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving metrics: {str(e)}")

    def _cleanup_old_metrics(self):
        """Clean up old metric files."""
        try:
            retention_days = self.config.get('metrics_retention_days', 7)
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            for file_path in self.metrics_dir.glob('metrics_*.json'):
                try:
                    file_date = datetime.strptime(file_path.stem.split('_')[1],
                                                '%Y%m%d')
                    if file_date < cutoff_date:
                        file_path.unlink()
                except Exception as e:
                    self.logger.error(f"Error processing file {file_path}: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error cleaning up metrics: {str(e)}")

    def get_current_metrics(self) -> Dict:
        """Get current model metrics."""
        if not self.metrics_history:
            return {}
        return self.metrics_history[-1]

    def get_metrics_history(self, 
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None) -> List[Dict]:
        """Get metrics history within time range."""
        if not start_time:
            start_time = datetime.now() - timedelta(days=1)
        if not end_time:
            end_time = datetime.now()
        
        return [
            m for m in self.metrics_history
            if start_time <= datetime.fromisoformat(m['timestamp']) <= end_time
        ]

    def get_drift_history(self,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> List[Dict]:
        """Get drift history within time range."""
        if not start_time:
            start_time = datetime.now() - timedelta(days=1)
        if not end_time:
            end_time = datetime.now()
        
        return [
            d for d in self.drift_history
            if start_time <= datetime.fromisoformat(d['timestamp']) <= end_time
        ]

    def get_alerts(self) -> List[Dict]:
        """Get current alerts."""
        alerts = []
        while True:
            try:
                alert = self.alerts.get_nowait()
                alerts.append(alert)
            except queue.Empty:
                break
        return alerts

    def update_retraining_time(self):
        """Update last retraining timestamp."""
        self.last_retraining = datetime.now()
        self.logger.info(f"Retraining timestamp updated: {self.last_retraining}") 