import logging
import os
from dotenv import load_dotenv
from costly import Costlog
from jinja2 import Environment, FileSystemLoader

LOGGER = logging.getLogger(__name__)

load_dotenv()

GLOBAL_COST_LOG = Costlog(mode="jsonl", discard_extras=True)
SIMULATE = os.getenv("SIMULATE", "False").lower() == "true"
DISABLE_COSTLY = os.getenv("DISABLE_COSTLY", "False").lower() == "true"
CACHING = os.getenv("CACHING", "False").lower() == "true"
MAX_CONCURRENT_QUERIES = int(os.getenv("MAX_CONCURRENT_QUERIES", 100))
NUM_LOGITS = 5
MAX_WORDS = 100
DEFAULT_MODEL=os.getenv("DEFAULT_MODEL", "claude-3-haiku-20240307")
DEFAULT_BON_MODEL=os.getenv("DEFAULT_BON_MODEL", "claude-3-haiku-20240307") # default Best-of-N ranker
RUNLOCAL = os.getenv("RUNLOCAL", "False").lower() == "true"
USE_TQDM = os.getenv("USE_TQDM", "True").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

TOOL_CALL_START_TAG = "<tool_call>"
TOOL_CALL_END_TAG = "</tool_call>"
TOOL_RESULT_START_TAG = "<tool_result>"
TOOL_RESULT_END_TAG = "</tool_result>"
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")
jinja_env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
jinja_env.globals.update(
    {
        "TOOL_CALL_START_TAG": TOOL_CALL_START_TAG,
        "TOOL_CALL_END_TAG": TOOL_CALL_END_TAG,
        "TOOL_RESULT_START_TAG": TOOL_RESULT_START_TAG,
        "TOOL_RESULT_END_TAG": TOOL_RESULT_END_TAG,
        "MAX_WORDS": MAX_WORDS,
    }
)
# helper function for logging
# returns source code of jinja template
jinja_env.get_source = lambda template: (
    jinja_env.loader.get_source(jinja_env, template)[0]
)
TOOL_CALL_TEMPLATE = jinja_env.get_template("tool_use/tool_call.jinja")
TOOL_RESULT_TEMPLATE = jinja_env.get_template("tool_use/tool_result.jinja")
