import logging
import os
import asyncio
import time
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

class RateLimiter:

    def __init__(self):
        self.initialize_limiters()
        self.openrouter_calls_without_rate_limit_check: int = 0
        self.token_usage_window = 60  # 1 minute window for TPM

    def initialize_limiters(self):
        self.OPENROUTER_LIMITER = AsyncLimiter(500) # will be updated later
        self.openrouter_token_usage: list[tuple[int, int]] = [] # List of (timestamp, tokens) tuples
        self.OPENROUTER_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_QUERIES)

        for model in self.stuff:
            if model.startswith("openrouter/"):
                self.stuff[model]["rate_limiter"] = self.OPENROUTER_LIMITER
                self.stuff[model]["token_usage"] = self.openrouter_token_usage
                self.stuff[model]["semaphore"] = self.OPENROUTER_SEMAPHORE
            else:
                self.stuff[model]["rate_limiter"] = AsyncLimiter(self.stuff[model]["rpm"])
                self.stuff[model]["token_usage"] = []
                self.stuff[model]["semaphore"] = asyncio.Semaphore(MAX_CONCURRENT_QUERIES)

    def get_current_token_usage(self, model: str) -> int:
        """Get token usage in the past minute, and clean up old entries"""

        now = time.time()
        window_start = now - self.token_usage_window
        
        # Clean old entries and sum current window
        self.stuff[model]["token_usage"] = [
            (ts, tokens) for ts, tokens in self.stuff[model]["token_usage"] 
            if ts > window_start
        ]
        return sum(tokens for _, tokens in self.stuff[model]["token_usage"])

    def add_token_usage(self, model: str, tokens: int):
        """Log token usage for a call."""

        now = time.time()
        if model.startswith("openrouter/"):
            self.openrouter_token_usage.append((now, tokens))
            for _model in self.openrouter_models:
                assert self.stuff[_model]["token_usage"][-1] == (now, tokens)
        else:
            self.stuff[model]["token_usage"].append((now, tokens))

    def wait_for_tpm(self, model: str, tokens: int) -> float:
        """Returns how long we need to wait for tpm to free up."""

        rate_limit_info = self.stuff[model]
        tpm = rate_limit_info.get("tpm")
        if tpm is None:
            return 0

        current_usage = self.get_current_token_usage(model)
        if current_usage + tokens > tpm:
            # Calculate how long to wait for enough tokens to free up
            oldest_timestamp = min(
                (ts for ts, _ in rate_limit_info["token_usage"]),
                default=time.time()
            )
            wait_time = self.token_usage_window - (time.time() - oldest_timestamp)
            return max(0, wait_time)
        return 0
            
    # def override_defaults(self):
    #     # allow setting rate limits in .env, e.g. if you're on a different tier
    #     for model in self.stuff:
    #         self.stuff[model]["rpm"] = coerce(
    #             os.getenv(f"{model.upper()}_RPM", self.stuff[model].get("rpm", None)), int
    #         )
    #         self.stuff[model]["tpm"] = coerce(
    #             os.getenv(f"{model.upper()}_TPM", self.stuff[model].get("tpm", None)), int
    #         )
    
    @property
    def openrouter_models(self) -> list[str]:
        return [model for model in self.stuff if model.startswith("openrouter/")]

    def update_openrouter_ratelimit(self) -> bool:
        """https://openrouter.ai/docs/limits"""
        try:
            import requests
            url = "https://openrouter.ai/api/v1/auth/key"
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}"
            }
            
            response = requests.get(url, headers=headers)
            result = response.json()['data']['rate_limit'] # {'requests': 750, 'interval': '10s'}
            num = result['requests']
            denom = parse_time_interval(result['interval'])
            rpm = min(0.85 * 60 * num / denom, 1000)
            checked_openrouter_successfully = True
            
        except Exception as e:
            import traceback
            LOGGER.error(f"Error getting OpenRouter rate limits:\n{e}\n{traceback.format_exc()}")
            rpm = 500 # play safe until we get real rate limit
            checked_openrouter_successfully = False
        
        for model in self.openrouter_models:
            LOGGER.info(f"Setting rpm for {model} to {rpm}")
            self.stuff[model]["rpm"] = rpm
        
        return checked_openrouter_successfully
    
    def get(self, model: str) -> dict:
        """Returns a dict like
        {"rpm": 4000, "tpm": 4e5, "rate_limiter": AsyncLimiter, "token_usage": list[tuple[int, int]]}

        Use this function to get, instead of RateLimiter.stuff[model], to ensure that update_openrouter_ratelimit is
        regularly called.
        """
        if model.startswith("openrouter/"):
            # check and update OpenRouter rate limit every 100 or so calls
            checked_openrouter_this_turn: bool = False
            if self.openrouter_calls_without_rate_limit_check > CHECK_OPENROUTER_EVERY:
                checked_openrouter_this_turn = self.update_openrouter_ratelimit()
            if checked_openrouter_this_turn:
                self.openrouter_calls_without_rate_limit_check = 0                
            else: 
                self.openrouter_calls_without_rate_limit_check += 1
        return self.stuff.get(model, None)
    
    # def set_last_request(self, model: str, last_request: float):
    #     if model.startswith("openrouter/"):
    #         self.OPENROUTER_LIMITER["last_request"] = last_request
    #         for _model in self.openrouter_models:
    #             assert self.stuff[_model]["rate_limiter"]["last_request"] == last_request
    #     else:
    #         self.stuff[model]["rate_limiter"]["last_request"] = last_request


    async def enforce_limits(self, model: str, messages: list[dict[str, str]]):
        """
        Enforces RPM, TPM, and max concurrency limits for a given model.
        """
        if model not in self.stuff:
            return

        limiter = self.stuff[model]

        # Enforce RPM limit
        async with limiter["rate_limiter"]:
            # Enforce TPM limit
            estimated_tokens = estimate_tokens(messages)
            tpm_wait = self.wait_for_tpm(model, estimated_tokens)
            await asyncio.sleep(tpm_wait)

            # Enforce max concurrency
            async with limiter["semaphore"]:
                # Yield control back to the caller
                return

    # initialize rate limits
    stuff = (
        {
            model: {"rpm": 500, "tpm": 3e5} # 4e5 but let's be even gentler
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
                "rpd": 1500,
            },  # requests per day is not enforced
        }
        | {
            "openrouter/deepseek/deepseek-chat": {"rpm": 500}, # in general 60*($$ in your OpenRouter account)
            "openrouter/gpt-4o-mini-2024-07-18": {"rpm": 500},
            "openrouter/gpt-4o-mini": {"rpm": 500},
        }
    )
    # | {
    #     "deepseek/deepseek-chat": {"rpm": None, "tpm": None},
    #     "deepseek/deepseek-reasoner": {"rpm": None, "tpm": None},
# }

RATE_LIMITER = RateLimiter()

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