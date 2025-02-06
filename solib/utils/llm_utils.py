import functools
import uuid
import logging
import traceback
import json
import math
import litellm
from litellm import acompletion
from litellm.types.utils import ModelResponse
from litellm.caching.caching import Cache
from typing import Any, TYPE_CHECKING
from collections import defaultdict
from dotenv import load_dotenv
from pydantic import BaseModel
from costly import Costlog, costly, CostlyResponse
from solib.utils.globals import *
from solib.utils.llm_hf_utils import get_hf_llm
from solib.utils.rate_limits.rate_limits import RATE_LIMITER

if TYPE_CHECKING:
    from litellm.types.utils import (
        # ModelResponse,
        ChatCompletionMessageToolCall,
        Function,
    )

LOGGER = logging.getLogger(__name__)

load_dotenv()

# see full list of config options at https://github.com/BerriAI/litellm/blob/main/litellm/__init__.py
litellm.add_function_to_prompt = (
    True  # add tools to prompt for nonnative models, idk if works though
)
litellm.drop_params = True  # make LLM calls ignore extra params
litellm.cache = Cache(type="disk")
# litellm.set_verbose = True


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


def is_local(model: str) -> bool:
    return (
        model.startswith("ollama/")
        or model.startswith("ollama_chat/")
        or model.startswith("localhf://")
    )


def is_localhf(model: str) -> bool:
    return model.startswith("localhf://")


def should_use_words_in_mouth(model: str) -> bool:
    """Models that should use the words_in_mouth. All other models will drop it."""
    return is_local(model)


def supports_async(model: str) -> bool:
    """Models that support async completion."""
    return not is_local(model)


def supports_tool_use(model: str) -> bool:
    """Returns whether a model supports tool use based on its name."""
    # Local models don't support tools
    if is_local(model) or is_localhf(model):
        return False

    # Claude models support tools
    if "claude" in model.lower():
        return True

    # OpenAI models generally support tools
    if any(x in model.lower() for x in ["gpt-4", "gpt-3.5"]):
        return True

    # Specific model checks
    model_lower = model.lower()
    if "deepseek" in model_lower:
        return False  # is currently buggy, don't use this
    if "gemini" in model_lower:
        return True
    if "mistral" in model_lower:
        return True
    if "llama" in model_lower and not is_local(model):
        return True

    # Default to False for unknown models
    return False


def supports_response_models(model: str) -> bool:
    """Returns whether a model supports response models based on its name."""
    # Local HF models and Deepseek don't support response models
    if is_localhf(model) or "deepseek" in model.lower():
        return False

    return True


@costly(**COSTLY_PARAMS)
async def acompletion_ratelimited(
    model: str,
    messages: list[dict[str, str]],
    caching: bool = CACHING,
    cost_log: Costlog = GLOBAL_COST_LOG,
    simulate: bool = SIMULATE,
    **kwargs,
) -> "ModelResponse":

    max_retries = kwargs.pop("max_retries", 10)
    call_id = uuid.uuid4().hex

    LOGGER.info(
        f"Getting response [async]; params:\n"
        f"model: {model}\n"
        f"messages: {messages}\n"
        f"kwargs: {kwargs}\n"
        f"caching: {caching}\n"
        f"call_id: {call_id}\n"
    )

    LOGGER.info(f"{call_id}: Making request to {model}")
    acompletion_ = functools.partial(
        acompletion, max_retries=1, caching=caching, **kwargs
    )
    try:
        response = await RATE_LIMITER.call(
            model_id=model,
            messages=messages,
            max_attempts=max_retries,
            call_function=acompletion_,
            is_valid=lambda x: x is not None,
        )
    except Exception as e:
        LOGGER.error(f"{call_id}: Error in completion: {e}")
        LOGGER.error(traceback.format_exc())
        LOGGER.error(f"Context: {call_id=}")
        raise e
    LOGGER.debug(f"{call_id}: Response from {model}: {response}")

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
            LOGGER.info(
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
    tool_calls: dict[str, ChatCompletionMessageToolCall] = (
        {}
    )  # map tool call ids to their associated messages
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
    ) >= 2, "At most one of tools, response_format, and return_probs_for should be provided."
    assert (prompt is None) + (
        messages is None
    ) == 1, "Exactly one of prompt and messages should be provided."

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
        if "max_tokens" in kwargs:
            max_tokens = kwargs.pop("max_tokens")
            LOGGER.info(
                f"{max_tokens=}. "
                "Using max_tokens with structured responses never ends well! "
                "Removing it."
            )

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


class LLM_Agent:

    def __init__(
        self,
        model: str = None,
        tools: list[callable] = None,
        hf_quantization_config=True,
    ):
        self.model = model or DEFAULT_MODEL  # or "claude-3-5-sonnet-20241022"
        self.tools = tools
        if is_localhf(self.model):
            assert (
                not self.tools
            ), "Tools are not currently supported for localhf:// models, use ollama instead."
            self.hf_quantization_config = hf_quantization_config
            self.client, self.generate_func, self.return_probs_func = get_hf_llm(
                self.model
            )

    async def get_response(
        self,
        response_model: BaseModel | None = None,
        prompt: str = None,
        messages: list[dict[str, str]] = None,
        system_prompt: str = None,
        words_in_mouth: str = None,
        **kwargs,
    ):
        if is_localhf(self.model):
            return await self.generate_func(
                prompt=prompt,
                messages=messages,
                system_message=system_prompt,
                words_in_mouth=words_in_mouth,
                **kwargs,
            )
        else:
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

    async def get_probs(
        self,
        return_probs_for: list[str],
        prompt: str = None,
        messages: list[dict[str, str]] = None,
        words_in_mouth: str = None,
        system_prompt: str = None,
        **kwargs,
    ):
        if is_localhf(self.model):
            return await self.return_probs_func(
                return_probs_for=return_probs_for,
                prompt=prompt,
                messages=messages,
                system_message=system_prompt,
                words_in_mouth=words_in_mouth,
                **kwargs,
            )
        else:
            return await acompletion_wrapper(
                model=self.model,
                return_probs_for=return_probs_for,
                prompt=prompt,
                messages=messages,
                system_prompt=system_prompt,
                words_in_mouth=words_in_mouth,
                **kwargs,
            )
