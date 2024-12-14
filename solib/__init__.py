import os
from datetime import datetime
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler

load_dotenv()

# Define custom STATUS level (35 is between WARNING-30 and ERROR-40)
STATUS_LEVEL = 35
logging.addLevelName(STATUS_LEVEL, 'STATUS')

# Add status method to logger
def status(self, message, *args, **kwargs):
    if self.isEnabledFor(STATUS_LEVEL):
        self._log(STATUS_LEVEL, message, args, **kwargs)

logging.Logger.status = status

LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
LOG_LEVEL_CONSOLE = os.getenv("LOG_LEVEL_CONSOLE", "WARNING")
LOG_LEVEL_FILE = os.getenv("LOG_LEVEL_FILE", "DEBUG")

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(LOG_LEVEL)

FORMATTER = logging.Formatter(
    "{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
)

CONSOLE_HANDLER = logging.StreamHandler()
CONSOLE_HANDLER.setLevel(LOG_LEVEL_CONSOLE)
CONSOLE_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(CONSOLE_HANDLER)

os.makedirs(".logs", exist_ok=True)

FILE_HANDLER = RotatingFileHandler(
    f".logs/{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.log",
    maxBytes=1024 * 1024,
    backupCount=3,
)
FILE_HANDLER.setLevel(LOG_LEVEL_FILE)
FILE_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(FILE_HANDLER)
