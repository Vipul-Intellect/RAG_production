import logging
from pathlib import Path


# Create logs directory if it doesn't exist
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "rag.log"


# Create logger
logger = logging.getLogger("rag_app")
logger.setLevel(logging.INFO) #Logger will record INFO ,WARNING,ERROR,CRITICAL


# Prevent duplicate logs
logger.propagate = False


# Log format
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)


# Console handler
console_handler = logging.StreamHandler() # console kis vs code terminal where the logs will be printed but stored in log folder
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)


# File handler
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)


# Add handlers only once
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)