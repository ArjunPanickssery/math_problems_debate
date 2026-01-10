import logging
import os
from costly.simulators.llm_simulator_faker import LLM_Simulator_Faker
from costly.estimators.llm_api_estimation import LLM_API_Estimation
from dotenv import load_dotenv
from costly import Costlog
from jinja2 import Environment, FileSystemLoader

from solib.datatypes import Prob
from litellm.types.utils import Choices, Message, ModelResponse
from pydantic import BaseModel


# from solib.utils.rate_limits.rate_limits import RateLimiter

LOGGER = logging.getLogger(__name__)

load_dotenv()

GLOBAL_COST_LOG = Costlog(mode="jsonl", discard_extras=True)
SIMULATE = os.getenv("SIMULATE", "False").lower() == "true"
DISABLE_COSTLY = os.getenv("DISABLE_COSTLY", "False").lower() == "true"
CACHING = os.getenv("CACHING", "True").lower() == "true"
NUM_LOGITS = 5
MAX_WORDS = 100
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "openrouter/openai/gpt-oss-20b")
DEFAULT_BON_MODEL = os.getenv(
    "DEFAULT_BON_MODEL", "openrouter/openai/gpt-oss-20b"
)  # default Best-of-N ranker
RUNLOCAL = os.getenv("RUNLOCAL", "False").lower() == "true"
USE_TQDM = os.getenv("USE_TQDM", "True").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
ENABLE_PROMPT_HISTORY = os.getenv("ENABLE_PROMPT_HISTORY", "False").lower() == "true"
RATE_LIMITER_FRAC_RATE_LIMIT = float(os.getenv("RATE_LIMITER_FRAC_RATE_LIMIT", "0.9"))

# Argument alignment verification settings
VERIFY_ALIGNMENT = os.getenv("VERIFY_ALIGNMENT", "False").lower() == "true"
VERIFY_ALIGNMENT_N_TRIES = int(os.getenv("VERIFY_ALIGNMENT_N_TRIES", "3"))
VERIFY_ALIGNMENT_MODEL = os.getenv("VERIFY_ALIGNMENT_MODEL", "openrouter/openai/gpt-oss-20b")

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



class Custom_Estimator(LLM_API_Estimation):
    PRICES = LLM_API_Estimation.PRICES | {
        "openrouter/nvidia/nemotron-3-nano-30b-a3b": {
            "input_cost_per_token": 6e-8,      # $0.06 per million tokens
            "output_cost_per_token": 2.4e-7,   # $0.24 per million tokens
        },
        "openrouter/openai/gpt-oss-20b": {
            "input_cost_per_token": 2e-8,      # $0.06 per million tokens
            "output_cost_per_token": 1e-7,   # $0.24 per million tokens
        },
        "openrouter/x-ai/grok-4.1-fast": {
            "input_cost_per_token": 2e-7,
            "output_cost_per_token": 5e-7,
        }
    }    

class LLM_Simulator(LLM_Simulator_Faker):
    ESTIMATOR = Custom_Estimator

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
        fast: bool = False,
    ) -> ModelResponse:
        output: str | BaseModel = super().simulate_llm_call(
            input_string,
            input_tokens,
            messages,
            model,
            response_model,
            cost_log,
            description,
            fast=fast,
        )
        if isinstance(output, BaseModel):
            output = output.model_dump_json()
        return ModelResponse(
            id="SIMULATED",
            created=0,
            model=model,
            object="chat.completion",
            system_fingerprint="PLANTED_FINGERPRINTS",
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    index=0,
                    message=Message(
                        content=output,
                        role="assistant",
                        tool_calls=None,  # TODO
                    ),
                )
            ],
        )

COSTLY_PARAMS = {
    "simulator": LLM_Simulator.simulate_llm_call,
    "estimator": Custom_Estimator.get_cost_real,
    "response_model": "response_format",
    "disable_costly": DISABLE_COSTLY,
    # "fast": True,
}
