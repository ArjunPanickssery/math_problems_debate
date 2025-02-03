import logging
import os
import asyncio
from pydantic import BaseModel
from dotenv import load_dotenv
from costly import Costlog
from jinja2 import Environment, FileSystemLoader
from litellm.types.utils import ModelResponse, Choices, Message
from solib.utils import coerce
from solib.datatypes import Prob
from costly.simulators.llm_simulator_faker import LLM_Simulator_Faker

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

RATE_LIMITERS = (
    {
        model: {"rpm": 4000, "tpm": 4e5}
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

for k in RATE_LIMITERS:
    # allow setting rate limits in .env, e.g. if you're on a different tier
    RATE_LIMITERS[k]["rpm"] = coerce(
        os.getenv(f"{k.upper()}_RPM", RATE_LIMITERS[k].get("rpm", None)), int
    )
    RATE_LIMITERS[k]["tpm"] = coerce(
        os.getenv(f"{k.upper()}_TPM", RATE_LIMITERS[k].get("tpm", None)), int
    )
    RATE_LIMITERS[k]["semaphore"] = asyncio.Semaphore(MAX_CONCURRENT_QUERIES)
    RATE_LIMITERS[k]["last_request"] = 0

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