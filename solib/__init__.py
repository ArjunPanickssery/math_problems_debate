import os
from datetime import datetime
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler

load_dotenv()

# LOG_LEVEL_MAP = {
#     "DEBUG": logging.DEBUG,
#     "INFO": logging.INFO,
#     "WARNING": logging.WARNING,
#     "ERROR": logging.ERROR,
#     "CRITICAL": logging.CRITICAL,
# }
# LOG_LEVEL = LOG_LEVEL_MAP[os.getenv("LOG_LEVEL", "INFO")]

LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
LOG_LEVEL_CONSOLE = os.getenv("LOG_LEVEL_CONSOLE", "WARNING")
LOG_LEVEL_FILE = os.getenv("LOG_LEVEL_FILE", "DEBUG")

#LOGGER = logging.get#LOGGER(__name__)
#LOGGER.setLevel(LOG_LEVEL)

FORMATTER = logging.Formatter(
    "{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
)

CONSOLE_HANDLER = logging.StreamHandler()
CONSOLE_HANDLER.setLevel(LOG_LEVEL_CONSOLE)
CONSOLE_HANDLER.setFormatter(FORMATTER)
#LOGGER.addHandler(CONSOLE_HANDLER)

FILE_HANDLER = RotatingFileHandler(
    f".logs/{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.log",
    maxBytes=1024 * 1024,
    backupCount=3,
)
FILE_HANDLER.setLevel(LOG_LEVEL_FILE)
FILE_HANDLER.setFormatter(FORMATTER)
#LOGGER.addHandler(FILE_HANDLER)
