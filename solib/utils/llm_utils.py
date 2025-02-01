import os
import uuid
import logging
import asyncio
import json
import time
import math
import litellm
from litellm import completion, acompletion
from litellm.caching.caching import Cache
from typing import Any, TYPE_CHECKING
from collections import defaultdict
from dotenv import load_dotenv
from pydantic import BaseModel
from costly import Costlog, costly, CostlyResponse
from jinja2 import Environment, FileSystemLoader
from solib.utils import coerce

if TYPE_CHECKING:
    from litellm.types.utils import (
        ModelResponse,
        ChatCompletionMessageToolCall,
        Function,
    )

LOGGER = logging.getLogger(__name__)

load_dotenv()

GLOBAL_COST_LOG = Costlog(mode="jsonl", discard_extras=True)
SIMULATE = os.getenv("SIMULATE", "False").lower() == "true"
DISABLE_COSTLY = os.getenv("DISABLE_COSTLY", "False").lower() == "true"
CACHING = os.getenv("CACHING", "False").lower() == "true"
MAX_CONCURRENT_QUERIES = int(os.getenv("MAX_CONCURRENT_QUERIES", 100))
NUM_LOGITS = 5
MAX_WORDS = 100
RUNHF = os.getenv("RUNHF", "False").lower() == "true"
USE_TQDM = os.getenv("USE_TQDM", "True").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")


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

# add tools to prompt for models that don't natively support it 
# -- idk if this works though
litellm.add_function_to_prompt = True
litellm.drop_params = True  # make LLM calls ignore extra params
litellm.cache = Cache(type="disk", disk_cache_dir=".cache")
# see full list at https://github.com/BerriAI/litellm/blob/main/litellm/__init__.py

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


def format_prompt(
    prompt: str, system_prompt: str = None, words_in_mouth: str = None
) -> list[dict[str, str]]:
    system_prompt = system_prompt or "You are a helpful assistant."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    if words_in_mouth:  # litellm assistant prefill
        messages.append({"role": "assistant", "content": words_in_mouth})
    return messages


def should_use_words_in_mouth(model: str) -> bool:
    """Models that should use the words_in_mouth. All other models will drop it."""
    return model.startswith("ollama/") or model.startswith("ollama_chat/")


def supports_async(model: str) -> bool:
    """Models that support async completion."""
    return not model.startswith("ollama/") and not model.startswith("ollama_chat/")

@costly()
async def acompletion_ratelimited(
    model: str,
    messages: list[dict[str, str]],
    caching: bool =CACHING,
    cost_log: Costlog = GLOBAL_COST_LOG,
    simulate: bool = SIMULATE,
    **kwargs,
) -> "ModelResponse":
    """
    Wrapper around acompletion that respects the defined global Semaphores and rate limits,
    and also uses costly.
    """
    rate_limiter = RATE_LIMITERS.get(model, None)
    max_retries = kwargs.pop("max_retries", 3)
    call_id = uuid.uuid4().hex
    LOGGER.info(
        f"Getting response [async]; params:\n"
        f"model: {model}\n"
        f"messages: {messages}\n"
        f"kwargs: {kwargs}\n"
        f"caching: {caching}\n"
        f"call_id: {call_id}\n"
    )
    if rate_limiter is None:
        LOGGER.warning(
            f"{call_id}: Rate limiter not found for model {model}, running without rate limits."
        )
        response = await acompletion(model, messages, max_retries=max_retries, caching=caching, **kwargs)
        LOGGER.debug(f"{call_id}: Response from {model}: {response}")
        return response
    async with rate_limiter["semaphore"]:
        now = time.time()
        elapsed = now - rate_limiter["last_request"]
        rpm = rate_limiter["rpm"]
        if rpm is not None and elapsed < 60 / rpm:
            LOGGER.info(
                f"{call_id}: Sleeping for {60 / rpm - elapsed} seconds to avoid overloading {model}."
            )
            await asyncio.sleep(60 / rpm - elapsed)
        LOGGER.info(f"{call_id}: Now actually making the request to {model}.")
        response = await acompletion(
            model=model,
            messages=messages,
            max_retries=max_retries,
            caching=caching,
            **kwargs,
        )
        LOGGER.debug(f"{call_id}: Response from {model}: {response}")
        rate_limiter["last_request"] = time.time()
        return CostlyResponse(
            output=response,
            cost_info={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            },
        )

@costly()
def completion_ratelimited(
    model: str,
    messages: list[dict[str, str]],
    caching: bool = CACHING,
    cost_log: Costlog = GLOBAL_COST_LOG,
    simulate: bool = SIMULATE,
    **kwargs,
) -> "ModelResponse":
    """
    Wrapper around completion that respects the defined global Semaphores and rate limits,
    and also uses costly.
    """
    rate_limiter = RATE_LIMITERS.get(model, None)
    max_retries = kwargs.pop("max_retries", 3)
    call_id = uuid.uuid4().hex
    LOGGER.info(
        f"Getting response [sync]; params:\n"
        f"model: {model}\n"
        f"messages: {messages}\n"
        f"kwargs: {kwargs}\n"
        f"caching: {caching}\n"
        f"call_id: {call_id}\n"
    )
    if rate_limiter is None:
        LOGGER.warning(
            f"{call_id}: Rate limiter not found for model {model}, running without rate limits."
        )
        response = completion(model, messages, max_retries=max_retries, caching=caching, **kwargs)
        LOGGER.debug(f"{call_id}: Response from {model}: {response}")
        return response
    now = time.time()
    elapsed = now - rate_limiter["last_request"]
    rpm = rate_limiter["rpm"]
    if rpm is not None and elapsed < 60 / rpm:
        LOGGER.info(
            f"{call_id}: Sleeping for {60 / rpm - elapsed} seconds to avoid overloading {model}."
        )
        time.sleep(60 / rpm - elapsed)
    LOGGER.info(f"{call_id}: Now actually making the request to {model}.")
    response = completion(
        model=model,
        messages=messages,
        max_retries=max_retries,
        caching=caching,
        **kwargs,
    )
    LOGGER.debug(f"{call_id}: Response from {model}: {response}")
    rate_limiter["last_request"] = time.time()
    return CostlyResponse(
        output=response,
        cost_info={
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        },
    )


async def acompletion_toolloop(
    model: str, messages: list[dict[str, str]], tools: list[callable], **kwargs
) -> list[dict[str, str]]:
    """
    Run a chat model in a tool use loop. The loop terminates when the model outputs a message that
    does not contain any tool calls. If a tool call is detected, the result is computed, and appended
    to the message history.

    Uses acompletion_ratelimited.
    """
    toolloop_id = uuid.uuid4().hex
    for tool in tools:
        if not callable(tool):
            raise ValueError("1) what")
        if not hasattr(tool, "json") or not isinstance(tool.json, dict):
            json_spec: dict = litellm.utils.function_to_dict(tool)
            LOGGER.warning(
                "A tool is a function with an attribute json of type dict. "
                f"Assuming the following json spec:\n\n {json_spec}"
            )
            tool.json = {"type": "function", "function": json_spec}
    start_len = len(messages)
    tools_ = [tool.json for tool in tools]
    tool_map = {tool.json["function"]["name"]: tool for tool in tools}
    LOGGER.debug(
        f"Starting tool loop with tools {[t.json['function']['name'] for t in tools]}."
    )

    i = 0
    while True:
        LOGGER.info(f"Tool loop {toolloop_id} iteration {i} starting ...")
        response = await acompletion_ratelimited(
            model=model,
            messages=messages,
            tools=tools_,
            **kwargs,
        )

        response_text: str = response.choices[0].message.content
        response_tool_calls: list[ChatCompletionMessageToolCall] = response.choices[
            0
        ].message.tool_calls
        response_msg = {
            "role": "assistant",
            "content": response_text,
            "tool_calls": response_tool_calls,
        }
        messages.append(response_msg)

        LOGGER.info(
            f"Tool loop {toolloop_id} iteration {i} response:\n"
            f"{response_text}\n"
            f"tool calls:\n"
            f"{response_tool_calls}"
        )

        if response_tool_calls:
            for t in response_tool_calls:
                t: ChatCompletionMessageToolCall
                f: Function = t.function
                f_name: str = f.name
                f_args: str = f.arguments
                tool: callable = tool_map[f_name]
                tool_kwargs: dict = json.loads(f_args)
                tool_result: Any = tool(**tool_kwargs)
                tool_result_msg = {
                    "tool_call_id": response_tool_calls[0].id,
                    "role": "tool",
                    "content": str(tool_result),
                }
                messages.append(tool_result_msg)
                LOGGER.debug(
                    f"{len(messages)} messages in tool loop {toolloop_id} so far..."
                )

        else:
            if i == 0:
                LOGGER.warning(
                    f"Tool loop {toolloop_id} did not result in any tool calls."
                )
            LOGGER.info(
                f"Tool loop {toolloop_id} terminated with {i} tool call iterations."
            )
            return messages[start_len:]


def completion_toolloop(
    model: str, messages: list[dict[str, str]], tools: list[callable], **kwargs
) -> list[dict[str, str]]:
    """
    Run a chat model in a tool use loop. The loop terminates when the model outputs a message that
    does not contain any tool calls. If a tool call is detected, the result is computed, and appended
    to the message history.

    Uses completion_ratelimited.
    """
    toolloop_id = uuid.uuid4().hex
    for tool in tools:
        if not callable(tool):
            raise ValueError("1) what")
        if not hasattr(tool, "json") or not isinstance(tool.json, dict):
            json_spec = litellm.utils.function_to_dict(tool)
            LOGGER.warning(
                "A tool is a function with an attribute json of type dict. "
                f"Assuming the following json spec:\n\n {json_spec}"
            )
            tool.json = {"type": "function", "function": json_spec}
    start_len = len(messages)
    tools_ = [tool.json for tool in tools]
    tool_map = {tool.json["function"]["name"]: tool for tool in tools}

    i = 0
    while True:
        LOGGER.info(f"Tool loop {toolloop_id} iteration {i} starting ...")
        response = completion_ratelimited(
            model=model,
            messages=messages,
            tools=tools_,
            **kwargs,
        )

        response_text = response.choices[0].message.content
        response_tool_calls = response.choices[0].message.tool_calls
        response_msg = {
            "role": "assistant",
            "content": response_text,
            "tool_calls": response_tool_calls,
        }
        messages.append(response_msg)

        LOGGER.info(
            f"Tool loop {toolloop_id} iteration {i} response:\n"
            f"{response_text}\n"
            f"tool calls:\n"
            f"{response_tool_calls}"
        )

        if response_tool_calls:
            for t in response_tool_calls:
                t: ChatCompletionMessageToolCall
                f: Function = t.function
                f_name: str = f.name
                f_args: str = f.arguments
                tool: callable = tool_map[f_name]
                tool_kwargs: dict = json.loads(f_args)
                tool_result: Any = tool(**tool_kwargs)
                tool_result_msg = {
                    "tool_call_id": response_tool_calls[0].id,
                    "role": "tool",
                    "content": str(tool_result),
                }
                messages.append(tool_result_msg)
                LOGGER.debug(
                    f"{len(messages)} messages in tool loop {toolloop_id} so far..."
                )
        else:
            if i == 0:
                LOGGER.warning(
                    f"Tool loop {toolloop_id} did not result in any tool calls."
                )
            LOGGER.info(
                f"Tool loop {toolloop_id} terminated with {i} tool call iterations."
            )
            return messages[start_len:]

        i += 1


def render_tool_call(name: str, args: dict[str, str]) -> str:
    """Render a tool call with the given name and arguments. This is used to represent tool calls
    happening in API-based models, as these models typically will not give you the actual tokens
    that are generated. Also, this way we can make transcripts of API-based models and
    hugging face models more similar."""
    return TOOL_CALL_TEMPLATE.render(name=name, arguments=args)


def render_tool_call_result(name: str, result: str, args: dict[str, str]) -> str:
    """Render a tool call result with the given name, result, and arguments. This is used to represent
    the outputs of tool calls in both API-based models and hugging face models."""
    return TOOL_RESULT_TEMPLATE.render(name=name, result=result, arguments=args)


def render_tool_call_conversation(messages: list[dict[str, str]]) -> str:
    """Given a list of BaseMessages, turn the conversation into one unified string representation that includes
    model responses, tool calls, and tool call results."""
    raw_response = ""
    tool_calls: dict[
        str, ChatCompletionMessageToolCall
    ] = {}  # map tool call ids to their associated messages
    for msg in messages:
        if msg["role"] == "assistant":
            if msg["content"]:  # sometimes msg["content"] is None
                raw_response += msg["content"]
            if (
                "tool_calls" in msg and msg["tool_calls"]
            ):  # sometimes msg["tool_calls"] is None
                for t in msg["tool_calls"]:
                    t: ChatCompletionMessageToolCall
                    tool_calls[t.id] = t
                    f: Function = t.function
                    f_name: str = f.name
                    f_args: str = f.arguments  # it's a str but it's fine
                    raw_response += render_tool_call(f_name, f_args)
        elif msg["role"] == "tool":
            t: ChatCompletionMessageToolCall = tool_calls[msg["tool_call_id"]]
            f: Function = t.function
            f_name: str = f.name
            f_args: str = f.arguments  # it's a str but it's fine
            raw_response += render_tool_call_result(f_name, msg["content"], f_args)
    return raw_response


async def acompletion_wrapper(
    model: str,
    tools: list[callable] | None = None,
    response_format: BaseModel | None = None,
    return_probs_for: list[str] | None = None,
    prompt: str | None = None,
    messages: list[dict[str, str]] | None = None,
    system_prompt=None,
    words_in_mouth=None,
    **kwargs,
) -> str | BaseModel | dict[str, float]:
    """
    Wrapper to handle diffrent needs.

    At most one of tools, response_format, and return_probs_for should be provided.

    Exactly one of prompt and messages should be provided.
    """

    assert (tools is None) + (response_format is None) + (
        return_probs_for is None
    ) >= 2, (
        "At most one of tools, response_format, and return_probs_for should be provided."
    )
    assert (prompt is None) + (messages is None) == 1, (
        "Exactly one of prompt and messages should be provided."
    )

    if prompt:
        LOGGER.info(f"Converting {prompt} to messages.")
        if not should_use_words_in_mouth(model):
            words_in_mouth = None
        messages = format_prompt(
            prompt, system_prompt=system_prompt, words_in_mouth=words_in_mouth
        )

    if tools:
        LOGGER.info(
            f"Getting response from {model} with tools {[t.__name__ for t in tools]}."
        )
        response: list[dict[str, str]] = await acompletion_toolloop(
            model=model, messages=messages, tools=tools, **kwargs
        )
        response_rendered: str = render_tool_call_conversation(response)
        return response_rendered

    if return_probs_for:
        LOGGER.info(f"Getting logprobs for tokens {return_probs_for} from {model}.")
        response: ModelResponse = await acompletion_ratelimited(
            model,
            messages,
            logprobs=True,
            top_logprobs=NUM_LOGITS,
            max_tokens=20,
            **kwargs,
        )
        # logprob_content is a list of dicts -- each dict contains the next chosen token and
        # a 'top_logprobs' key with the logprobs for all alternatives
        all_logprobs_sequence: list[dict] = response.choices[0].logprobs["content"]
        # we usually only care for the immediate next token
        all_logprobs = all_logprobs_sequence[0]["top_logprobs"]
        all_logprobs_dict = {x["token"]: x["logprob"] for x in all_logprobs}
        probs = defaultdict(float)  # {token: 0 for token in return_probs_for}
        for token in return_probs_for:
            if token in all_logprobs_dict:
                probs[token] = math.exp(all_logprobs_dict[token])
        total_probs = sum(probs.values())
        try:
            probs_relative: dict[str, float] = {
                token: probs[token] / total_probs for token in return_probs_for
            }
        except ZeroDivisionError:
            import pdb

            pdb.set_trace()
        return probs_relative

    if response_format:
        LOGGER.info(f"Getting structured response from {model}.")
        response: ModelResponse = await acompletion_ratelimited(
            model, messages, response_format=response_format, **kwargs
        )
        response_json: str = response.choices[0].message.content
        response_obj: BaseModel = response_format.model_validate_json(response_json)
        return response_obj

    LOGGER.info(f"Getting native text response from {model}.")
    response: ModelResponse = await acompletion_ratelimited(model, messages, **kwargs)
    response_text: str = response.choices[0].message.content
    return response_text


def completion_wrapper(
    model: str,
    tools: list[callable] | None = None,
    response_format: BaseModel | None = None,
    return_probs_for: list[str] | None = None,
    prompt: str | None = None,
    messages: list[dict[str, str]] | None = None,
    system_prompt=None,
    words_in_mouth=None,
    **kwargs,
) -> str | BaseModel | dict[str, float]:
    """
    Wrapper to handle diffrent needs.

    At most one of tools, response_format, and return_probs_for should be provided.

    Exactly one of prompt and messages should be provided.
    """

    assert (tools is None) + (response_format is None) + (
        return_probs_for is None
    ) >= 2, (
        "At most one of tools, response_format, and return_probs_for should be provided."
    )
    assert (prompt is None) + (messages is None) == 1, (
        "Exactly one of prompt and messages should be provided."
    )

    if prompt:
        LOGGER.info(f"Converting {prompt} to messages.")
        if not should_use_words_in_mouth(model):
            words_in_mouth = None
        messages = format_prompt(
            prompt, system_prompt=system_prompt, words_in_mouth=words_in_mouth
        )

    if tools:
        LOGGER.info(
            f"Getting response from {model} with tools {[t.__name__ for t in tools]}."
        )
        response: list[dict[str, str]] = completion_toolloop(
            model=model, messages=messages, tools=tools, **kwargs
        )
        response_rendered: str = render_tool_call_conversation(response)
        return response_rendered

    if return_probs_for:
        LOGGER.info(f"Getting logprobs for tokens {return_probs_for} from {model}.")
        response: ModelResponse = completion_ratelimited(
            model,
            messages,
            logprobs=True,
            top_logprobs=NUM_LOGITS,
            max_tokens=20,
            **kwargs,
        )
        # logprob_content is a list of dicts -- each dict contains the next chosen token and
        # a 'top_logprobs' key with the logprobs for all alternatives
        all_logprobs_sequence: list[dict] = response.choices[0].logprobs["content"]
        # we usually only care for the immediate next token
        all_logprobs = all_logprobs_sequence[0]["top_logprobs"]
        all_logprobs_dict = {x["token"]: x["logprob"] for x in all_logprobs}
        probs = defaultdict(float)
        for token in return_probs_for:
            if token in all_logprobs_dict:
                probs[token] = math.exp(all_logprobs_dict[token])
        total_probs = sum(probs.values())
        try:
            probs_relative: dict[str, float] = {
                token: probs[token] / total_probs for token in return_probs_for
            }
        except ZeroDivisionError:
            import pdb

            pdb.set_trace()
        return probs_relative

    if response_format:
        LOGGER.info(f"Getting structured response from {model}.")
        response: ModelResponse = completion_ratelimited(
            model, messages, response_format=response_format, **kwargs
        )
        response_json: str = response.choices[0].message.content
        response_obj: BaseModel = response_format.model_validate_json(response_json)
        return response_obj

    LOGGER.info(f"Getting native text response from {model}.")
    response: ModelResponse = completion_ratelimited(model, messages, **kwargs)
    response_text: str = response.choices[0].message.content
    return response_text


class LLM_Agent:
    """Kind of a legacy abstraction layer that doesn't add anything, but everything in the code is built around it."""

    def __init__(
        self, model: str = None, tools: list[callable] = None, sync_mode: bool = False
    ):
        self.model = (
            model or "claude-3-5-sonnet-20241022"
        )  # or "claude-3-5-sonnet-20241022"
        self.tools = tools
        self.sync_mode = sync_mode
        self.supports_async = supports_async(model)

    def get_response_sync(
        self,
        prompt: str = None,
        messages: list[dict[str, str]] = None,
        response_model: BaseModel | None = None,
        system_prompt: str = None,
        words_in_mouth: str = None,
        **kwargs,
    ):
        return completion_wrapper(
            model=self.model,
            tools=self.tools,
            response_format=response_model,
            prompt=prompt,
            messages=messages,
            system_prompt=system_prompt,
            words_in_mouth=words_in_mouth,
            **kwargs,
        )

    async def get_response_async(
        self,
        response_model: BaseModel | None = None,
        prompt: str = None,
        messages: list[dict[str, str]] = None,
        system_prompt: str = None,
        words_in_mouth: str = None,
        **kwargs,
    ):
        return await acompletion_wrapper(
            model=self.model,
            tools=self.tools,
            response_format=response_model,
            prompt=prompt,
            messages=messages,
            system_prompt=system_prompt,
            words_in_mouth=words_in_mouth,
            **kwargs,
        )

    async def get_response(self, *args, **kwargs):
        if self.supports_async and not self.sync_mode:
            return await self.get_response_async(*args, **kwargs)
        return self.get_response_sync(*args, **kwargs)

    async def get_probs(self, *args, **kwargs):
        if self.supports_async and not self.sync_mode:
            return await self.get_probs_async(*args, **kwargs)
        return self.get_probs_sync(*args, **kwargs)

    def get_probs_sync(
        self,
        return_probs_for: list[str],
        prompt: str = None,
        messages: list[dict[str, str]] = None,
        words_in_mouth: str = None,
        system_prompt: str = None,
        **kwargs,
    ):
        return completion_wrapper(
            model=self.model,
            return_probs_for=return_probs_for,
            prompt=prompt,
            messages=messages,
            system_prompt=system_prompt,
            words_in_mouth=words_in_mouth,
            **kwargs,
        )

    async def get_probs_async(
        self,
        return_probs_for: list[str],
        prompt: str = None,
        messages: list[dict[str, str]] = None,
        words_in_mouth: str = None,
        system_prompt: str = None,
        **kwargs,
    ):
        return await acompletion_wrapper(
            model=self.model,
            return_probs_for=return_probs_for,
            prompt=prompt,
            messages=messages,
            system_prompt=system_prompt,
            words_in_mouth=words_in_mouth,
            **kwargs,
        )
