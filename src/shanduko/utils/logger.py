# src/shanduko/utils/logger.py
import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name: str, log_file: Path = None, level=logging.INFO):
    """Set up logger with console and file handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger

# Example usage in model_training.py:
# logger = setup_logger(
#     'model_training',
#     Path('logs/training_{}.log'.format(datetime.now().strftime('%Y%m%d_%H%M%S')))
# )