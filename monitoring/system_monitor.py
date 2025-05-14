import psutil
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import threading
import queue
import time
import os
import json
from pathlib import Path

class SystemMonitor:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metrics_queue = queue.Queue()
        self.alert_queue = queue.Queue()
        self.monitoring = False
        self.monitor_thread = None
        self.metrics_history = []
        self.alerts_history = []
        self.start_time = datetime.now()
        
        # Create metrics directory if it doesn't exist
        self.metrics_dir = Path('metrics')
        self.metrics_dir.mkdir(exist_ok=True)

    def start_monitoring(self):
        """Start system monitoring in a separate thread."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.logger.info("System monitoring started")

    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.info("System monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_queue.put(metrics)
                self._check_thresholds(metrics)
                self._save_metrics(metrics)
                time.sleep(self.config.get('monitoring_interval', 1))
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")

    def _collect_metrics(self) -> Dict:
        """Collect system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            # Network metrics
            net_io = psutil.net_io_counters()
            
            # Process metrics
            process = psutil.Process(os.getpid())
            process_memory = process.memory_info()
            process_cpu = process.cpu_percent()
            
            # System uptime
            uptime = datetime.now() - self.start_time
            
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count,
                    'frequency': {
                        'current': cpu_freq.current,
                        'min': cpu_freq.min,
                        'max': cpu_freq.max
                    }
                },
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'percent': memory.percent,
                    'used': memory.used,
                    'free': memory.free
                },
                'swap': {
                    'total': swap.total,
                    'used': swap.used,
                    'free': swap.free,
                    'percent': swap.percent
                },
                'disk': {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'percent': disk.percent,
                    'read_bytes': disk_io.read_bytes,
                    'write_bytes': disk_io.write_bytes
                },
                'network': {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv,
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv
                },
                'process': {
                    'memory_rss': process_memory.rss,
                    'memory_vms': process_memory.vms,
                    'cpu_percent': process_cpu,
                    'threads': process.num_threads(),
                    'open_files': len(process.open_files()),
                    'connections': len(process.connections())
                },
                'uptime': str(uptime)
            }
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {str(e)}")
            return {}

    def _check_thresholds(self, metrics: Dict):
        """Check metrics against thresholds and generate alerts."""
        try:
            thresholds = self.config.get('thresholds', {})
            
            # CPU threshold check
            if metrics['cpu']['percent'] > thresholds.get('cpu_percent', 80):
                self._generate_alert('high_cpu', 
                                   f"CPU usage is {metrics['cpu']['percent']}%")
            
            # Memory threshold check
            if metrics['memory']['percent'] > thresholds.get('memory_percent', 80):
                self._generate_alert('high_memory',
                                   f"Memory usage is {metrics['memory']['percent']}%")
            
            # Disk threshold check
            if metrics['disk']['percent'] > thresholds.get('disk_percent', 80):
                self._generate_alert('high_disk',
                                   f"Disk usage is {metrics['disk']['percent']}%")
            
            # Process memory threshold check
            if metrics['process']['memory_rss'] > thresholds.get('process_memory', 1e9):
                self._generate_alert('high_process_memory',
                                   f"Process memory usage is {metrics['process']['memory_rss'] / 1e6:.2f} MB")
            
            # Network threshold check
            if metrics['network']['bytes_sent'] > thresholds.get('network_bytes', 1e9):
                self._generate_alert('high_network_usage',
                                   f"High network usage: {metrics['network']['bytes_sent'] / 1e6:.2f} MB sent")
        except Exception as e:
            self.logger.error(f"Error checking thresholds: {str(e)}")

    def _generate_alert(self, alert_type: str, message: str):
        """Generate and queue an alert."""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'message': message,
            'severity': self._get_alert_severity(alert_type)
        }
        self.alert_queue.put(alert)
        self.alerts_history.append(alert)
        self.logger.warning(f"Alert: {message}")

    def _get_alert_severity(self, alert_type: str) -> str:
        """Get alert severity based on type."""
        severity_map = {
            'high_cpu': 'warning',
            'high_memory': 'warning',
            'high_disk': 'critical',
            'high_process_memory': 'warning',
            'high_network_usage': 'info'
        }
        return severity_map.get(alert_type, 'info')

    def _save_metrics(self, metrics: Dict):
        """Save metrics to file."""
        try:
            # Add to history
            self.metrics_history.append(metrics)
            
            # Keep history within limits
            max_history = self.config.get('max_metrics_history', 1000)
            if len(self.metrics_history) > max_history:
                self.metrics_history = self.metrics_history[-max_history:]
            
            # Save to file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = self.metrics_dir / f'metrics_{timestamp}.json'
            
            with open(file_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Clean up old files
            self._cleanup_old_metrics()
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
        """Get current system metrics."""
        try:
            return self.metrics_queue.get_nowait()
        except queue.Empty:
            return {}

    def get_alerts(self) -> List[Dict]:
        """Get current alerts."""
        alerts = []
        while True:
            try:
                alert = self.alert_queue.get_nowait()
                alerts.append(alert)
            except queue.Empty:
                break
        return alerts

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

    def get_alerts_history(self,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None) -> List[Dict]:
        """Get alerts history within time range."""
        if not start_time:
            start_time = datetime.now() - timedelta(days=1)
        if not end_time:
            end_time = datetime.now()
        
        return [
            a for a in self.alerts_history
            if start_time <= datetime.fromisoformat(a['timestamp']) <= end_time
        ] 