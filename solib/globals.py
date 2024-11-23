import os
from costly import Costlog
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader

load_dotenv()
GLOBAL_COST_LOG = Costlog(mode="jsonl", discard_extras=True)
SIMULATE = os.getenv("SIMULATE", "False").lower() == "true"
DISABLE_COSTLY = os.getenv("DISABLE_COSTLY", "False").lower() == "true"
USE_TQDM = os.getenv("USE_TQDM", "True").lower() == "true"
MAX_CONCURRENT_QUERIES = int(os.getenv("MAX_CONCURRENT_QUERIES", 10))
if SIMULATE:
    MAX_CONCURRENT_QUERIES = 10_000
MAX_WORDS = int(os.getenv("MAX_WORDS", 100))
# LOGGER = logging.get#LOGGER(__name__)

# Setup Jinja environment
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "prompts")
jinja_env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
