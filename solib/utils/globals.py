import logging
import os
import asyncio
import time
import traceback
from aiolimiter import AsyncLimiter
from pydantic import BaseModel
from dotenv import load_dotenv
from costly import Costlog
from jinja2 import Environment, FileSystemLoader
from litellm.types.utils import ModelResponse, Choices, Message
from solib.utils import parse_time_interval, estimate_tokens #, coerce
from solib.datatypes import Prob
from costly.simulators.llm_simulator_faker import LLM_Simulator_Faker

LOGGER = logging.getLogger(__name__)

load_dotenv()

GLOBAL_COST_LOG = Costlog(mode="jsonl", discard_extras=True)
SIMULATE = os.getenv("SIMULATE", "False").lower() == "true"
DISABLE_COSTLY = os.getenv("DISABLE_COSTLY", "False").lower() == "true"
CACHING = os.getenv("CACHING", "False").lower() == "true"
MAX_CONCURRENT_QUERIES = int(os.getenv("MAX_CONCURRENT_QUERIES", 100))
CHECK_OPENROUTER_EVERY = int(os.getenv("CHECK_OPENROUTER_EVERY", 100))
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
HF_TOKEN = os.getenv("HF_TOKEN")

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

# initialize rate limits
RATE_LIMITS = (
    {
        model: {"rpm": 500, "tpm": 8e4} # 4e5 but let's be even gentler
        for model in [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ]
    }
    | {
        "gpt-4o": {"rpm": 500, "tpm": 3e4},
        "gpt-4-turbo": {"rpm": 500, "tpm": 3e4},
        "gpt-4": {"rpm": 500, "tpm": 1e4},
        "gpt-4o-mini": {"rpm": 500, "tpm": 2e5},
        "gpt-3.5-turbo": {"rpm": 500, "tpm": 2e5},
        "o1-mini": {"rpm": 500, "tpm": 2e5},
        "o1-preview": {"rpm": 500, "tpm": 3e4},
    }
    | {
        "gemini/gemini-1.5-pro": {"rpm": 1000, "tpm": 4e6},
        "gemini/gemini-1.5-flash": {"rpm": 2000, "tpm": 4e6},
        "gemini/gemini-1.5-flash-8b": {"rpm": 4000, "tpm": 4e6},
        "gemini/gemini-2.0-flash-exp": {
            "rpm": 10,
            "tpm": 1e3,
            "rpd": 1500,
        },  # requests per day is not enforced
    }
    | {
        "openrouter/deepseek/deepseek-chat": {"rpm": 500, "tpm": 2e5},
        "openrouter/gpt-4o-mini-2024-07-18": {"rpm": 500, "tpm": 2e5},
        "openrouter/gpt-4o-mini": {"rpm": 500, "tpm": 2e5},
    }
)
# | {
#     "deepseek/deepseek-chat": {"rpm": None, "tpm": None},
#     "deepseek/deepseek-reasoner": {"rpm": None, "tpm": None},
# }

### RateLimiter and Resource from
### https://github.com/safety-research/safety-tooling

class RateLimiter:
    def __init__(self, rpm):
        self.rpm = rpm
        self.allowable_requests = rpm
        self.last_refill_time = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self, context: str = None):
        async with self.lock:
            now = time.time()
            time_passed = now - self.last_refill_time
            self.allowable_requests = min(self.rpm, self.allowable_requests + time_passed * (self.rpm / 60))
            self.last_refill_time = now

            if self.allowable_requests < 1:
                wait_time = (1 - self.allowable_requests) / (self.rpm / 60)
                LOGGER.info(f"Waiting {wait_time} seconds. Context: {context}")
                await asyncio.sleep(wait_time)
                self.allowable_requests = 1

            self.allowable_requests -= 1


class Resource:
    """
    A resource that is consumed over time and replenished at a constant rate.
    """

    def __init__(self, refresh_rate, total=0, throughput=0):
        self.refresh_rate = refresh_rate
        self.total = total
        self.throughput = throughput
        self.last_update_time = time.time()
        self.start_time = time.time()
        self.value = self.refresh_rate

    def _replenish(self):
        """
        Updates the value of the resource based on the time since the last update.
        """
        curr_time = time.time()
        self.value = min(
            self.refresh_rate,
            self.value + (curr_time - self.last_update_time) * self.refresh_rate / 60,
        )
        self.last_update_time = curr_time
        self.throughput = self.total / (curr_time - self.start_time) * 60

    def geq(self, amount: float) -> bool:
        self._replenish()
        return self.value >= amount

    def consume(self, amount: float):
        """
        Consumes the given amount of the resource.
        """
        assert self.geq(amount), f"Resource does not have enough capacity to consume {amount} units"
        self.value -= amount
        self.total += amount

RATE_LIMITERS = {}
SEMAPHORES = {}
LOCKS = {}
REQUEST_CAPACITIES = {}
TOKEN_CAPACITIES = {}

for model, rate_limit in RATE_LIMITS.items():
    RATE_LIMITERS[model] = RateLimiter(rate_limit["rpm"])
    SEMAPHORES[model] = asyncio.Semaphore(MAX_CONCURRENT_QUERIES)
    LOCKS[model] = asyncio.Lock()
    REQUEST_CAPACITIES[model] = Resource(rate_limit["rpm"])
    TOKEN_CAPACITIES[model] = Resource(rate_limit["tpm"])



class LLM_Simulator(LLM_Simulator_Faker):
    @classmethod
    def _fake_custom(cls, t: type):
        assert issubclass(t, Prob)
        import random

        return t(prob=random.random())

    @classmethod
    def simulate_llm_call(
        cls,
        input_string: str = None,
        input_tokens: int = None,
        messages: list[dict[str, str]] = None,
        model: str = None,
        response_model: type = str,
        cost_log: Costlog = None,
        description: list[str] = None,
    ) -> ModelResponse:
        output: str | BaseModel = super().simulate_llm_call(
            input_string,
            input_tokens,
            messages,
            model,
            response_model,
            cost_log,
            description,
        )
        if isinstance(output, BaseModel):
            output = output.model_dump_json()
        return ModelResponse(
            id="SIMULATED", 
            created=0, 
            model=model, 
            object='chat.completion', 
            system_fingerprint='PLANTED_FINGERPRINTS',
            choices = [
                Choices(
                    finish_reason='tool_calls',
                    index=0,
                    message=Message(
                        content=output,
                        role='assistant',
                        tool_calls=None, # TODO
                    )
                )
            ]
        )

COSTLY_PARAMS = {
    "simulator": LLM_Simulator.simulate_llm_call,
    "response_model": "response_format",
    "disable_costly": DISABLE_COSTLY,
}