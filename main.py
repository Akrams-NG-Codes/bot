import os
import sys
import logging
import schedule
import time
from datetime import datetime
from dotenv import load_dotenv

from core.bot import AkramsNG
from utils.logger import setup_logger
from config.config import (
    LOG_CONFIG,
    ML_CONFIG,
    EMAIL_CONFIG,
    TELEGRAM_CONFIG
)

def setup_environment():
    """Setup environment variables and logging."""
    # Load environment variables
    load_dotenv()
    
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs(ML_CONFIG['model_path'], exist_ok=True)
    
    # Setup logging
    logger = setup_logger('main', LOG_CONFIG)
    return logger

def schedule_tasks(bot: AkramsNG):
    """Schedule periodic tasks."""
    # Schedule model retraining
    schedule.every(ML_CONFIG['retrain_interval']).hours.do(bot.retrain_models)
    
    # Schedule performance reports
    schedule.every().day.at("00:00").do(
        lambda: bot.notifier.send_performance_notification(bot.get_performance_report()['performance_metrics'])
    )
    
    # Schedule system status updates
    schedule.every(6).hours.do(
        lambda: bot.notifier.send_system_status_notification({
            'is_running': bot.is_running,
            'open_positions': len(bot.mt5.get_open_positions()),
            'daily_trades': bot.risk_manager.daily_trades,
            'current_drawdown': bot.risk_manager.current_drawdown
        })
    )

def main():
    """Main entry point for the trading bot."""
    # Setup environment
    logger = setup_environment()
    logger.info("Starting Akrams-NG Trading Bot")
    
    try:
        # Initialize bot
        bot = AkramsNG()
        
        # Schedule periodic tasks
        schedule_tasks(bot)
        
        # Start the bot
        bot.start()
        
        # Main loop
        while True:
            try:
                # Run scheduled tasks
                schedule.run_pending()
                
                # Sleep for a short interval
                time.sleep(1)
                
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                bot.notifier.send_error_notification(e, "Main loop error")
                time.sleep(5)
        
        # Stop the bot
        bot.stop()
        logger.info("Akrams-NG Trading Bot stopped")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 