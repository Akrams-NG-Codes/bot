import logging
import logging.config
from config.config import LOG_CONFIG

def setup_logger(name, config):
    """Setup and return a logger instance."""
    logging.config.dictConfig(config)
    return logging.getLogger(name) 