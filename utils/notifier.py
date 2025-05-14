import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import logging
from typing import Dict, Optional
from config.config import EMAIL_CONFIG, TELEGRAM_CONFIG

class Notifier:
    def __init__(self, email_config: Dict, telegram_config: Dict):
        self.logger = logging.getLogger(__name__)
        self.email_config = email_config
        self.telegram_config = telegram_config

    def send_notification(self, subject: str, message: str, notification_type: str = 'all'):
        """Send notification through specified channels."""
        if notification_type in ['email', 'all']:
            self._send_email(subject, message)
        
        if notification_type in ['telegram', 'all']:
            self._send_telegram(subject, message)

    def _send_email(self, subject: str, message: str):
        """Send email notification."""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_config['smtp_username']
            msg['To'] = self.email_config['recipient_email']
            msg['Subject'] = f"Akrams-NG: {subject}"
            
            # Add message body
            msg.attach(MIMEText(message, 'plain'))
            
            # Connect to SMTP server
            server = smtplib.SMTP(
                self.email_config['smtp_server'],
                self.email_config['smtp_port']
            )
            server.starttls()
            
            # Login
            server.login(
                self.email_config['smtp_username'],
                self.email_config['smtp_password']
            )
            
            # Send email
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email notification sent: {subject}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {str(e)}")

    def _send_telegram(self, subject: str, message: str):
        """Send Telegram notification."""
        try:
            # Format message
            formatted_message = f"*{subject}*\n\n{message}"
            
            # Prepare API request
            url = f"https://api.telegram.org/bot{self.telegram_config['bot_token']}/sendMessage"
            data = {
                'chat_id': self.telegram_config['chat_id'],
                'text': formatted_message,
                'parse_mode': 'Markdown'
            }
            
            # Send message
            response = requests.post(url, data=data)
            response.raise_for_status()
            
            self.logger.info(f"Telegram notification sent: {subject}")
            
        except Exception as e:
            self.logger.error(f"Failed to send Telegram notification: {str(e)}")

    def send_trade_notification(self, trade_info: Dict):
        """Send trade notification with formatted trade information."""
        subject = "New Trade Executed"
        message = (
            f"Symbol: {trade_info['symbol']}\n"
            f"Type: {trade_info['type']}\n"
            f"Entry Price: {trade_info['entry_price']}\n"
            f"Stop Loss: {trade_info['stop_loss']}\n"
            f"Take Profit: {trade_info['take_profit']}\n"
            f"Position Size: {trade_info['position_size']}\n"
            f"Strategy: {trade_info['strategy']}"
        )
        
        self.send_notification(subject, message)

    def send_error_notification(self, error: Exception, context: str = ""):
        """Send error notification with formatted error information."""
        subject = "Trading Bot Error"
        message = (
            f"Context: {context}\n"
            f"Error: {str(error)}\n"
            f"Type: {type(error).__name__}"
        )
        
        self.send_notification(subject, message)

    def send_performance_notification(self, performance_metrics: Dict):
        """Send performance notification with formatted metrics."""
        subject = "Performance Update"
        message = (
            f"Total Trades: {performance_metrics['total_trades']}\n"
            f"Win Rate: {performance_metrics['win_rate']:.2%}\n"
            f"Total Profit: {performance_metrics['total_profit']:.2f}\n"
            f"Max Drawdown: {performance_metrics['max_drawdown']:.2%}"
        )
        
        self.send_notification(subject, message)

    def send_system_status_notification(self, status: Dict):
        """Send system status notification."""
        subject = "System Status Update"
        message = (
            f"Status: {'Running' if status['is_running'] else 'Stopped'}\n"
            f"Open Positions: {status['open_positions']}\n"
            f"Daily Trades: {status['daily_trades']}\n"
            f"Current Drawdown: {status['current_drawdown']:.2%}"
        )
        
        self.send_notification(subject, message) 