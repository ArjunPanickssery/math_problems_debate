import asyncio
import logging
import os
from costly import Costlog
from dotenv import load_dotenv


load_dotenv()
GLOBAL_COST_LOG = Costlog(mode="jsonl", discard_extras=True)
SIMULATE = os.getenv("SIMULATE", "False").lower() == "true"
DISABLE_COSTLY = os.getenv("DISABLE_COSTLY", "False").lower() == "true"
USE_TQDM = os.getenv("USE_TQDM", "True").lower() == "true"
MAX_CONCURRENT_QUERIES = int(os.getenv("MAX_CONCURRENT_QUERIES", 10))
LOGGER = logging.getLogger(__name__)


def reset_global_semaphore():
    global GLOBAL_LLM_SEMAPHORE
    GLOBAL_LLM_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_QUERIES)
    LOGGER.info(
        f"Resetting global semaphore, max concurrent queries: {MAX_CONCURRENT_QUERIES}"
    )

reset_global_semaphore()