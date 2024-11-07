from collections import defaultdict
from functools import partial, wraps
import functools
import inspect
import os
import asyncio
import json
import math
import logging
from dotenv import load_dotenv
from typing import Callable, Iterable, Literal, Union, Coroutine, TYPE_CHECKING
from transformers import BitsAndBytesConfig
from pydantic import BaseModel
import cloudpickle
import io
from perscache import Cache
from perscache.cache import hash_it
from perscache.serializers import (
    JSONSerializer,
    Serializer,
    CloudPickleSerializer,
)
from perscache.storage import Storage
from costly import Costlog, CostlyResponse, costly
from costly.simulators.llm_simulator_faker import LLM_Simulator_Faker
import warnings

from langchain_core.messages import (
    HumanMessage,
    ToolMessage,
    SystemMessage,
    BaseMessage,
    AIMessage,
)
from langchain_core.runnables import ConfigurableField

import solib.tool_use.tool_rendering
from solib.utils import apply, apply_async  # noqa
from solib.datatypes import Prob
from solib.tool_use import tool_use
from solib.tool_use.tool_use import HuggingFaceToolCaller  # noqa

if TYPE_CHECKING:
    from transformers import AutoTokenizer

    # from langchain_openai import ChatOpenAI
    # from langchain_mistralai import ChatMistralAI
    # from langchain_anthropic import ChatAnthropic

LOGGER = logging.getLogger(__name__)


# for cache
class PydanticJSONSerializer(JSONSerializer):
    @staticmethod
    def default(obj):
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        elif isinstance(obj, BaseModel) and issubclass(
            type(obj), BaseModel
        ):  # check for subclass
            # If it's a Pydantic model class, return its name for serialization
            return f"{obj.__module__}.{obj.__class__.__name__}"
        elif isinstance(obj, BaseModel):
            return obj.model_dump()
        else:
            try:
                return dict(obj)
            except:  # noqa
                raise TypeError(
                    f"Object of type {obj.__class__.__name__} is not JSON serializable"
                )

    @classmethod
    def dumps(cls, data):
        return json.dumps(data, default=cls.default).encode("utf-8")

    @classmethod
    def loads(cls, data):
        return json.loads(data.decode("utf-8"))


class DisabledStorage(Storage):  # class that disables storage
    def read(self, path, deadline) -> bytes:
        raise FileNotFoundError

    def write(self, path, data: bytes) -> None:
        pass


class BetterCache(Cache):
    """A subclass of Cache that hashes types by their names."""

    @staticmethod
    def _get_hash(
        fn: Callable,
        args: tuple,
        kwargs: dict,
        serializer: Serializer,
        ignore: Iterable[str],
    ) -> str:
        # Get the argument dictionary by binding the function signature with args and kwargs
        arg_dict = inspect.signature(fn).bind(*args, **kwargs).arguments

        # Remove ignored arguments from the argument dictionary
        if ignore is not None:
            arg_dict = {k: v for k, v in arg_dict.items() if k not in ignore}

        # Convert types in the argument dictionary to their names
        for key, value in arg_dict.items():
            if isinstance(value, type):
                arg_dict[key] = (
                    value.__name__
                )  # Use type name instead of the actual type object

        # Include global variables in the cache hash because Python handles default
        # variables a bit differently than you might expect
        arg_dict["simulate"] = SIMULATE  # Add to the hash key
        arg_dict["cache_breaker"] = CACHE_BREAKER

        # Hash the function source, serializer type, and the argument dictionary
        return hash_it(inspect.getsource(fn), type(serializer).__name__, arg_dict)


class SafeCloudPickleSerializer(CloudPickleSerializer):
    # https://github.com/pydantic/pydantic/issues/8232#issuecomment-2189431721
    @classmethod
    def dumps(cls, obj):
        model_namespaces = {}

        with io.BytesIO() as f:
            pickler = cloudpickle.CloudPickler(f)

            for ModelClass in BaseModel.__subclasses__():
                model_namespaces[ModelClass] = ModelClass.__pydantic_parent_namespace__
                ModelClass.__pydantic_parent_namespace__ = None

            try:
                pickler.dump(obj)
                return f.getvalue()
            finally:
                for ModelClass, namespace in model_namespaces.items():
                    ModelClass.__pydantic_parent_namespace__ = namespace


load_dotenv()
cache = BetterCache(serializer=SafeCloudPickleSerializer())
storageless_cache = Cache(storage=DisabledStorage())
GLOBAL_COST_LOG = Costlog(mode="jsonl", discard_extras=True)
SIMULATE = os.getenv("SIMULATE", "False").lower() == "true"
DISABLE_COSTLY = os.getenv("DISABLE_COSTLY", "False").lower() == "true"
USE_TQDM = os.getenv("USE_TQDM", "True").lower() == "true"
MAX_CONCURRENT_QUERIES = int(os.getenv("MAX_CONCURRENT_QUERIES", 10))


def reset_global_semaphore():
    global GLOBAL_LLM_SEMAPHORE
    GLOBAL_LLM_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_QUERIES)
    LOGGER.info(
        f"Resetting global semaphore, max concurrent queries: {MAX_CONCURRENT_QUERIES}"
    )


def reset_cache_breaker():
    global CACHE_BREAKER
    CACHE_BREAKER = os.getenv("CACHE_BREAKER", "")


reset_global_semaphore()
reset_cache_breaker()


# HACK. I have no idea why this works but just manually adding 'self' to
# @cache(ignore=...) doesn't.
def method_cache(ignore=None):
    if ignore is None:
        ignore = []
    # Ensure 'self' is always ignored
    if "self" not in ignore:
        ignore = ["self"] + ignore

    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            return cache(ignore=ignore)(method)(self, *args, **kwargs)

        return wrapper

    return decorator


class LLM_Simulator(LLM_Simulator_Faker):
    @classmethod
    def _fake_custom(cls, t: type):
        assert issubclass(t, Prob)
        import random

        return t(prob=random.random())


# async def parallelized_call(
#     func: Coroutine,
#     data: list[str],
#     max_concurrent_queries: int = 100,
# ) -> list[any]:
#     """
#     Run async func in parallel on the given data.
#     func will usually be a partial which uses query_api or whatever in some way.

#     Example usage:
#         partial_eval_method = functools.partial(eval_method, model=model, **kwargs)
#         results = await parallelized_call(partial_eval_method, [format_post(d) for d in data])
#     """

#     if os.getenv("SINGLE_THREAD"):
#         LOGGER.info(f"Running {func} on {len(data)} datapoints sequentially")
#         return [await func(d) for d in data]

#     max_concurrent_queries = min(
#         max_concurrent_queries,
#         int(os.getenv("MAX_CONCURRENT_QUERIES", max_concurrent_queries)),
#     )

#     LOGGER.info(
#         f"Running {func} on {len(data)} datapoints with {max_concurrent_queries} concurrent queries"
#     )

#     # Create a local semaphore
#     local_semaphore = asyncio.Semaphore(max_concurrent_queries)

#     async def call_func(sem, func, datapoint):
#         async with sem:
#             return await func(datapoint)

#     LOGGER.info("Calling call_func")
#     tasks = [call_func(local_semaphore, func, d) for d in data]
#     return await asyncio.gather(*tasks)


# async def parallelized_call(
#     func: Coroutine,
#     data: list[str],
#     max_concurrent_queries: int = 100,
#     max_tokens_per_minute: int = 30000,  # Max tokens per minute
#     token_usage_per_call: int = 2000,  # Estimate token usage per call
# ) -> list[any]:
#     """
#     Run async func in parallel on the given data.
#     func will usually be a partial which uses query_api or whatever in some way.

#     Example usage:
#         partial_eval_method = functools.partial(eval_method, model=model, **kwargs)
#         results = await parallelized_call(partial_eval_method, [format_post(d) for d in data])
#     """

#     total_tokens_used = 0
#     token_lock = asyncio.Lock()  # Lock for managing token usage safely

#     if os.getenv("SINGLE_THREAD"):
#         LOGGER.info(f"Running {func} on {len(data)} datapoints sequentially")
#         return [await func(d) for d in data]

#     max_concurrent_queries = min(
#         max_concurrent_queries,
#         int(os.getenv("MAX_CONCURRENT_QUERIES", max_concurrent_queries)),
#     )

#     LOGGER.info(
#         f"Running {func} on {len(data)} datapoints with {max_concurrent_queries} concurrent queries"
#     )

#     # Create a local semaphore
#     local_semaphore = asyncio.Semaphore(max_concurrent_queries)

#     async def call_func(sem, func, datapoint):
#         nonlocal total_tokens_used
#         async with sem:
#             # Estimate the tokens for this call and update usage
#             async with token_lock:
#                 if total_tokens_used + token_usage_per_call > max_tokens_per_minute:
#                     # If token limit is about to exceed, wait
#                     await asyncio.sleep(60)  # Wait 60 seconds (1 minute)
#                     total_tokens_used = 0  # Reset token count after waiting

#                 total_tokens_used += token_usage_per_call

#             return await func(datapoint)

#     LOGGER.info("Calling call_func")

#     # Create tasks for all data points
#     tasks = [call_func(local_semaphore, func, d) for d in data]

#     # Run tasks with or without tqdm progress bar based on the use_tqdm flag
#     if use_tqdm:
#         results = []
#         # Wrapping the gather call with tqdm for progress tracking
#         async for result in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
#             results.append(await result)
#     else:
#         # Without tqdm, just await all tasks as usual
#         results = await asyncio.gather(*tasks)

#     return results


async def parallelized_call(
    func: Coroutine,
    data: list[any],
    max_concurrent_queries: int = 100,
) -> list[any]:
    """
    Run async func in parallel on the given data.
    func will usually be a partial which uses query_api or whatever in some way.

    Example usage:
        partial_eval_method = functools.partial(eval_method, model=model, **kwargs)
        results = await parallelized_call(partial_eval_method, [format_post(d) for d in data])
    """

    if os.getenv("SINGLE_THREAD"):
        LOGGER.info(f"Running {func} on {len(data)} datapoints sequentially")
        return [await func(d) for d in data]

    max_concurrent_queries = min(
        max_concurrent_queries,
        int(os.getenv("MAX_CONCURRENT_QUERIES", max_concurrent_queries)),
    )

    LOGGER.info(
        f"Running {func} on {len(data)} datapoints with {max_concurrent_queries} concurrent queries"
    )

    local_semaphore = asyncio.Semaphore(max_concurrent_queries)

    async def call_func(sem, func, datapoint):
        async with sem:
            return await func(datapoint)

    tasks = [call_func(local_semaphore, func, d) for d in data]
    return await asyncio.gather(*tasks)


def format_prompt(
    prompt: str = None,
    messages: list[dict[str, str] | BaseMessage] = None,
    input_string: str = None,
    tokenizer: Union["AutoTokenizer", None] = None,
    system_message: str | None = None,
    words_in_mouth: str | None = None,
    tools: list[callable] = None,
    natively_supports_tools: bool = False,
    msg_type: Literal["langchain", "dict"] = "dict",
) -> dict[
    Literal["messages", "input_string"], str | list[dict[str, str] | BaseMessage]
]:
    """
    Three types of prompts:
        prompt: 'What is the capital of the moon?'
        messages: a list[dict[str, str] | BaseMessage] in the Chat Message format
        input_string: a string that the LLM can directly <im_start> etc.

    This converts prompt -> messages -> input_string. Returns both messages
    and input_string. If both prompt and messages are provided, prompt is
    ignored. You cannot convert anything in the reverse order.

    Args:
        prompt: str: Prompt to convert. Either prompt or messages must be provided
            to calculate input_string.
        messages: list[dict[str, str] | BaseMessage]: Messages to convert. Either prompt or
            messages must be provided to calculate input_string.
        tokenizer: AutoTokenizer: Tokenizer to use for the conversion.
        system_message: str | None: System message to add to the messages. Will be
            ignored if messages is provided.
        words_in_mouth: str | None: Words to append to the prompt. Will be ignored
            if input_string is provided.
        tools: list[callable]: If tool use is enabled, this will be a list of python functions.
        natively_supports_tools: bool: If True, the model natively supports tools and we can pass
            them in the `tools` parameter in apply_chat_template if tokenizer is supplied. If False,
            we will use the default tool prompt in tool_use.HuggingFaceToolCaller.TOOLS_PROMPT, and
            we will append it as the first system message to the messages.
        msg_type: Literal["langchain", "dict"]: Type of messages. If "langchain", messages
            will be langchain.BaseMessages.If "dict", messages will be in the dictionary format
    Returns:
        dict: with keys "messages" and "input_string". "input_string" will be None
            if tokenizer is None.
    """
    if input_string is None:
        if messages is None:
            assert prompt is not None

            messages = []
            if system_message is not None:
                if msg_type == "langchain":
                    messages.append(SystemMessage(content=system_message))
                else:
                    messages.append({"role": "system", "content": system_message})

            if msg_type == "langchain":
                messages.append(HumanMessage(content=prompt))
            else:
                messages.append({"role": "user", "content": prompt})

        if tools:
            tool_msg = solib.tool_use.tool_rendering.get_tool_prompt(
                tools, natively_supports_tools
            )
            if msg_type == "langchain":
                messages.insert(0, SystemMessage(content=tool_msg))
            elif not natively_supports_tools:  # local models (where msg_type="dict") that natively support tools will have the tool prompt added by apply_chat_template
                messages.insert(  # otherwise, we manually create one
                    0,
                    {
                        "role": "system",
                        "content": tool_msg,
                    },
                )

        if tokenizer is not None:
            if tools and natively_supports_tools:
                input_string = tokenizer.apply_chat_template(
                    messages, tokenize=False, tools=tools
                )
            else:
                input_string = tokenizer.apply_chat_template(messages, tokenize=False)
        if words_in_mouth is not None:
            input_string += words_in_mouth
    return {"messages": messages, "input_string": input_string}


def convert_langchain_to_dict(
    messages: list[BaseMessage | dict[str, str]],
) -> list[dict[str, str]]:
    # TODO: modify this to support ToolMessage
    messages_out = []
    for m in messages:
        if (
            isinstance(m, HumanMessage)
            or isinstance(m, AIMessage)
            or isinstance(m, SystemMessage)
        ):
            messages_out.append({"role": m.type, "content": m.content})
        elif isinstance(m, ToolMessage):
            warnings.warn("ToolMessage is not supported yet")
        elif isinstance(m, dict):
            messages_out.append(m)
    return messages_out


@functools.cache
def load_hf_model(model: str, hf_quantization_config=True):
    print("Loading Hugging Face model", model, hf_quantization_config)
    from transformers import AutoTokenizer, AutoModelForCausalLM

    quant_config = (
        BitsAndBytesConfig(load_in_8bit=True) if hf_quantization_config else None
    )

    api_key = os.getenv("HF_TOKEN")
    model = model.split("hf:")[1]
    tokenizer = AutoTokenizer.from_pretrained(model)
    device_map = "cuda" if hf_quantization_config else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        model,
        device_map=device_map,
        token=api_key,
        quantization_config=quant_config,
    )

    return (tokenizer, model)


def get_llm(model: str, use_async=False, hf_quantization_config=True):
    if model.startswith("hf:"):  # Hugging Face local models
        msg_type = "dict"

        client = load_hf_model(model, hf_quantization_config)
        tokenizer, model = client

        # TODO: add cost logging for local models
        def generate(
            prompt: str = None,
            messages: list[dict[str, str]] = None,
            input_string: str = None,
            system_message: str | None = None,
            words_in_mouth: str | None = None,
            tools: list[callable] = None,
            max_tokens: int = 2048,
            **kwargs,
        ):
            natively_supports_tools = False
            if tools:
                bmodel = HuggingFaceToolCaller(tokenizer, model, tools)
                natively_supports_tools = bmodel.natively_supports_tools()
            else:
                bmodel = model

            input_string = format_prompt(
                prompt=prompt,
                messages=messages,
                input_string=input_string,
                tokenizer=tokenizer,
                system_message=system_message,
                words_in_mouth=words_in_mouth,
                tools=tools,
                natively_supports_tools=natively_supports_tools,
                msg_type=msg_type,
            )

            input_ids = tokenizer.encode(
                input_string["input_string"],
                return_tensors="pt",
                add_special_tokens=False,
            ).to(bmodel.device)

            input_length = input_ids.shape[1]
            output = bmodel.generate(
                input_ids,
                max_length=max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )[0]

            decoded = tokenizer.decode(output[input_length:], skip_special_tokens=True)
            return decoded

        # TODO: add cost logging for local models
        def return_probs(
            return_probs_for: list[str],
            prompt: str = None,
            messages: list[dict[str, str]] = None,
            input_string: str = None,
            system_message: str | None = None,
            words_in_mouth: str | None = None,
            **kwargs,
        ):
            input_string = format_prompt(  # don't pass tools to judges
                prompt=prompt,
                messages=messages,
                input_string=input_string,
                tokenizer=tokenizer,
                system_message=system_message,
                words_in_mouth=words_in_mouth,
                msg_type=msg_type,
            )
            input_ids = tokenizer.encode(
                input_string["input_string"], return_tensors="pt"
            ).to(model.device)
            output = model(input_ids).logits[0, -1, :]
            output_probs = output.softmax(dim=0)
            probs = {token: 0 for token in return_probs_for}
            for token in probs:
                # workaround for weird difference between word as a continuation vs standalone
                token_enc = tokenizer.encode(f"({token}", add_special_tokens=False)[-1]
                probs[token] = output_probs[token_enc].item()
            total_prob = sum(probs.values())
            try:
                probs_relative = {
                    token: prob / total_prob for token, prob in probs.items()
                }
            except ZeroDivisionError:
                import pdb

                pdb.set_trace()
            return probs_relative

        return {
            "client": client,
            "generate": generate,
            "return_probs": return_probs,
        }

    else:
        msg_type = "langchain"
        natively_supports_tools = (
            True  # assume all chat API models natively support tools
        )

        if model.startswith("or:"):  # OpenRouter
            from langchain_openai import ChatOpenAI

            api_key = os.getenv("OPENROUTER_API_KEY")
            client = ChatOpenAI(
                model=model, base_url="https://openrouter.ai/api/v1", api_key=api_key
            )

        else:
            if model.startswith(("gpt", "openai", "babbage", "davinci")):
                from langchain_openai import ChatOpenAI

                api_key = os.getenv("OPENAI_API_KEY")
                client = ChatOpenAI(model=model, api_key=api_key)

            elif model.startswith(("claude", "anthropic")):
                from langchain_anthropic import ChatAnthropic

                api_key = os.getenv("ANTHROPIC_API_KEY")
                client = ChatAnthropic(model=model, api_key=api_key)

            elif model.startswith("mistral"):
                from langchain_mistralai import ChatMistralAI

                api_key = os.getenv("MISTRAL_API_KEY")
                client = ChatMistralAI(model=model, api_key=api_key)

            else:
                raise ValueError(f"Model {model} is not supported for now")

        client = client.configurable_fields(
            max_tokens=ConfigurableField(id="max_tokens"),
            temperature=ConfigurableField(id="temperature"),
        )

        def process_response(response):
            if isinstance(
                response, list
            ):  # handle tool calling case, make the output match the hugging face case somewhat
                raw_response = (
                    solib.tool_use.tool_rendering.render_tool_call_conversation(
                        response
                    )
                )
                token_types = [
                    k
                    for k, v in response[0].usage_metadata.items()
                    if isinstance(v, int)
                ]
                usage = defaultdict(int)
                for token_type in token_types:
                    for r in response:
                        if not hasattr(r, "usage_metadata"):
                            continue
                        usage[token_type] += r.usage_metadata[token_type]

            elif isinstance(response, BaseMessage):  # handle singleton messages
                raw_response = response.content
                usage = response.usage_metadata
            elif isinstance(
                response, dict
            ):  # otherwise, we are using structured output
                raw = response["raw"]
                # probably should do some error handling here if 'parsing_error' is set
                parsed = response["parsed"]
                if response["parsing_error"] is not None:
                    LOGGER.error(raw)
                    raise ValueError(
                        f"Error parsing structured output: {response['parsing_error']}"
                    )

                raw_response = parsed
                usage = raw.usage_metadata

            return raw_response, usage

        _get_messages = lambda kwargs: convert_langchain_to_dict(  # noqa
            format_prompt(
                prompt=kwargs.get("prompt"),
                messages=kwargs.get("messages"),
                system_message=kwargs.get("system_message"),
                msg_type=msg_type,
                natively_supports_tools=natively_supports_tools,
            )["messages"]
        )

        def apply_client_bindings(
            tools: list[callable] = None,
            max_tokens: int = 2048,
            temperature: float = 0.0,
            response_model: Union["BaseModel", None] = None,
            top_logprobs: int = 5,
        ):
            if tools and response_model:
                raise ValueError("Cannot use tools with response_model")

            bclient = client.with_config(
                configurable={"max_tokens": max_tokens, "temperature": temperature}
            )

            if top_logprobs:
                bclient = bclient.bind(top_logprobs=top_logprobs, logprobs=True)

            if response_model:
                bclient = bclient.with_structured_output(
                    response_model, include_raw=True
                )

            get_response = bclient.ainvoke if use_async else bclient.invoke

            if tools:
                tool_map, structured_tools = (
                    solib.tool_use.tool_rendering.get_structured_tools(tools)
                )
                bclient = bclient.bind_tools(structured_tools)
                if use_async:
                    get_response = partial(
                        tool_use.tool_use_loop_generate_async,
                        get_response=bclient.ainvoke,
                        tool_map=tool_map,
                    )
                else:
                    get_response = partial(
                        tool_use.tool_use_loop_generate,
                        get_response=bclient.invoke,
                        tool_map=tool_map,
                    )

            return get_response

        def generate_format_and_bind(
            prompt,
            messages,
            system_message,
            max_tokens,
            response_model,
            tools,
            temperature,
            **kwargs,
        ):
            if "words_in_mouth" in kwargs:
                warnings.warn(
                    f"words_in_mouth is not supported for model type `{model}`",
                    UserWarning,
                )

            get_response = apply_client_bindings(
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
                response_model=response_model,
            )

            messages = format_prompt(
                prompt=prompt,
                messages=messages,
                system_message=system_message,
                tools=tools,
                msg_type=msg_type,
                natively_supports_tools=natively_supports_tools,
            )["messages"]

            return get_response, messages

        @costly(
            simulator=LLM_Simulator.simulate_llm_call,
            messages=_get_messages,
            disable_costly=DISABLE_COSTLY,
        )
        def generate(
            model: str = model,
            prompt: str = None,
            messages: list[dict[str, str]] = None,
            system_message: str | None = None,
            max_tokens: int = 2048,
            response_model: Union["BaseModel", None] = None,
            tools: list[callable] = None,
            temperature: float = 0.0,
            cost_log: Costlog = GLOBAL_COST_LOG,
            simulate: bool = SIMULATE,
            **kwargs,
        ):
            get_response, messages = generate_format_and_bind(
                prompt,
                messages,
                system_message,
                max_tokens,
                response_model,
                tools,
                temperature,
                **kwargs,
            )
            LOGGER.debug(messages)
            response = get_response(messages)

            raw_response, usage = process_response(response)

            LOGGER.debug(f"raw_response: {raw_response}")

            if response_model is not None and isinstance(raw_response, dict):
                raw_response = response_model(**raw_response)

            LOGGER.debug(f"raw_response: {raw_response}")

            return CostlyResponse(
                output=raw_response,
                cost_info=usage,
            )

        @costly(
            simulator=LLM_Simulator.simulate_llm_call,
            messages=_get_messages,
            disable_costly=DISABLE_COSTLY,
        )
        async def generate_async(
            model: str = model,
            prompt: str = None,
            messages: list[dict[str, str]] = None,
            system_message: str | None = None,
            max_tokens: int = 2048,
            response_model: Union["BaseModel", None] = None,
            tools: list[callable] = None,
            temperature: float = 0.0,
            cost_log: Costlog = GLOBAL_COST_LOG,
            simulate: bool = SIMULATE,
            **kwargs,
        ):
            get_response, messages = generate_format_and_bind(
                prompt,
                messages,
                system_message,
                max_tokens,
                response_model,
                tools,
                temperature,
                **kwargs,
            )

            response = await get_response(messages)

            raw_response, usage = process_response(response)

            LOGGER.debug(f"raw_response: {raw_response}")

            if response_model is not None and isinstance(raw_response, dict):
                raw_response = response_model(**raw_response)

            LOGGER.debug(f"raw_response: {raw_response}")

            return CostlyResponse(
                output=raw_response,
                cost_info=usage,
            )

        def extract_probability(response, return_probs_for):
            all_logprobs = response.response_metadata["logprobs"]["content"][0][
                "top_logprobs"
            ]
            all_logprobs_dict = {x["token"]: x["logprob"] for x in all_logprobs}
            probs = {token: 0 for token in return_probs_for}
            for token, prob in all_logprobs_dict.items():
                if token in probs:
                    probs[token] = math.exp(prob)
            total_prob = sum(probs.values())
            probs_relative = {token: prob / total_prob for token, prob in probs.items()}
            return probs_relative

        def probability_format_and_bind(
            prompt,
            messages,
            system_message,
            return_probs_for,
            top_logprobs,
            temperature,
            **kwargs,
        ):
            max_tokens = max(len(token) for token in return_probs_for)

            get_response = apply_client_bindings(
                max_tokens=max_tokens,
                temperature=temperature,
                top_logprobs=top_logprobs,
            )

            messages = format_prompt(
                prompt=prompt,
                messages=messages,
                system_message=system_message,
                msg_type=msg_type,
            )["messages"]

            return get_response, messages

        @costly(
            simulator=LLM_Simulator.simulate_llm_probs,
            messages=_get_messages,
            disable_costly=DISABLE_COSTLY,
        )
        def return_probs(
            return_probs_for: list[str],
            model: str = model,
            prompt: str = None,
            messages: list[dict[str, str]] = None,
            system_message: str | None = None,
            top_logprobs: int = 5,
            temperature: float = 0.0,
            cost_log: Costlog = GLOBAL_COST_LOG,
            simulate: bool = SIMULATE,
            **kwargs,
        ):
            get_response, messages = probability_format_and_bind(
                prompt,
                messages,
                system_message,
                return_probs_for,
                top_logprobs,
                temperature,
                **kwargs,
            )

            response = get_response(messages)

            raw_response, usage = process_response(response)
            probs_relative = extract_probability(response, return_probs_for)

            return CostlyResponse(
                output=probs_relative,
                cost_info=usage,
            )

        @costly(
            simulator=LLM_Simulator.simulate_llm_probs,
            messages=_get_messages,
            disable_costly=DISABLE_COSTLY,
        )
        async def return_probs_async(
            return_probs_for: list[str],
            model: str = model,
            prompt: str = None,
            messages: list[dict[str, str]] = None,
            system_message: str | None = None,
            top_logprobs: int = 5,
            temperature: float = 0.0,
            cost_log: Costlog = GLOBAL_COST_LOG,
            simulate: bool = SIMULATE,
            **kwargs,
        ):
            get_response, messages = probability_format_and_bind(
                prompt,
                messages,
                system_message,
                return_probs_for,
                top_logprobs,
                temperature,
                **kwargs,
            )

            response = await get_response(messages)

            raw_response, usage = process_response(response)
            probs_relative = extract_probability(response, return_probs_for)

            return CostlyResponse(
                output=probs_relative,
                cost_info=usage,
            )

        return {
            "client": client,
            "generate": generate,
            "generate_async": generate_async,
            "return_probs": return_probs,
            "return_probs_async": return_probs_async,
        }


class LLM_Agent:
    def __init__(
        self,
        model: str = None,
        tools: list[callable] = None,
        hf_quantization_config=True,
    ):
        self.model = model or "gpt-4o-mini"
        self.tools = tools
        self.hf_quantization_config = hf_quantization_config
        self.ai = get_llm(
            model=self.model,
            use_async=False,
            hf_quantization_config=hf_quantization_config,
        )
        if self.supports_async:
            self.ai_async = get_llm(
                model=self.model,
                use_async=True,
                hf_quantization_config=hf_quantization_config,
            )

    @property
    def supports_async(self):
        return not self.model.startswith("hf:")

    @method_cache(ignore=["cost_log"])
    def get_response_sync(
        self,
        response_model: Union["BaseModel", None] = None,
        prompt: str = None,
        messages: list[dict[str, str]] = None,
        input_string: str = None,
        system_message: str | None = None,
        words_in_mouth: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        cost_log: Costlog = GLOBAL_COST_LOG,
        simulate: bool = SIMULATE,
        **kwargs,
    ):
        return self.ai["generate"](
            model=self.model,
            tools=self.tools,
            response_model=response_model,
            prompt=prompt,
            messages=messages,
            input_string=input_string,
            system_message=system_message,
            words_in_mouth=words_in_mouth,
            max_tokens=max_tokens,
            temperature=temperature,
            cost_log=cost_log,
            simulate=simulate,
            **kwargs,
        )

    @method_cache(ignore=["cost_log"])
    async def get_response_async(
        self,
        response_model: Union["BaseModel", None] = None,
        prompt: str = None,
        messages: list[dict[str, str]] = None,
        input_string: str = None,
        system_message: str | None = None,
        words_in_mouth: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        cost_log: Costlog = GLOBAL_COST_LOG,
        simulate: bool = SIMULATE,
        **kwargs,
    ):
        async with GLOBAL_LLM_SEMAPHORE:
            return await self.ai_async["generate_async"](
                model=self.model,
                tools=self.tools,
                response_model=response_model,
                prompt=prompt,
                messages=messages,
                input_string=input_string,
                system_message=system_message,
                words_in_mouth=words_in_mouth,
                max_tokens=max_tokens,
                temperature=temperature,
                cost_log=cost_log,
                simulate=simulate,
                **kwargs,
            )

    async def get_response(self, *args, **kwargs):
        if self.supports_async:
            return await self.get_response_async(*args, **kwargs)
        return self.get_response_sync(*args, **kwargs)

    async def get_probs(self, *args, **kwargs):
        if self.supports_async:
            return await self.get_probs_async(*args, **kwargs)
        return self.get_probs_sync(*args, **kwargs)

    @method_cache(ignore=["cost_log"])
    def get_probs_sync(
        self,
        return_probs_for: list[str],
        prompt: str = None,
        messages: list[dict[str, str]] = None,
        input_string: str = None,
        system_message: str | None = None,
        words_in_mouth: str | None = None,
        top_logprobs: int = 5,
        temperature: float = 0.0,
        cost_log: Costlog = GLOBAL_COST_LOG,
        simulate: bool = SIMULATE,
        **kwargs,
    ):
        return self.ai["return_probs"](
            model=self.model,
            tools=self.tools,
            return_probs_for=return_probs_for,
            prompt=prompt,
            messages=messages,
            input_string=input_string,
            system_message=system_message,
            words_in_mouth=words_in_mouth,
            top_logprobs=top_logprobs,
            temperature=temperature,
            cost_log=cost_log,
            simulate=simulate,
            **kwargs,
        )

    @method_cache(ignore=["cost_log"])
    async def get_probs_async(
        self,
        return_probs_for: list[str],
        prompt: str = None,
        messages: list[dict[str, str]] = None,
        input_string: str = None,
        system_message: str | None = None,
        words_in_mouth: str | None = None,
        top_logprobs: int = 5,
        temperature: float = 0.0,
        cost_log: Costlog = GLOBAL_COST_LOG,
        simulate: bool = SIMULATE,
        **kwargs,
    ):
        async with GLOBAL_LLM_SEMAPHORE:
            return await self.ai_async["return_probs_async"](
                model=self.model,
                tools=self.tools,
                return_probs_for=return_probs_for,
                prompt=prompt,
                messages=messages,
                input_string=input_string,
                system_message=system_message,
                words_in_mouth=words_in_mouth,
                top_logprobs=top_logprobs,
                temperature=temperature,
                cost_log=cost_log,
                simulate=simulate,
                **kwargs,
            )


@cache(ignore="cost_log")
def get_llm_response(
    model: str = None,
    response_model: Union["BaseModel", None] = None,
    prompt: str = None,
    messages: list[dict[str, str]] = None,
    input_string: str = None,
    system_message: str | None = None,
    words_in_mouth: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    tools: list[callable] = None,
    hf_quantization_config=None,
    cost_log: Costlog = GLOBAL_COST_LOG,
    simulate: bool = SIMULATE,
    **kwargs,  # kwargs necessary for costly
):
    """NOTE: you should generally use the LLM_Agent class instead of this function.
    This is deprecated, or maybe we can just use it for one-time calls etc.
    """
    model = model or "gpt-4o-mini"
    ai = get_llm(
        model=model, use_async=False, hf_quantization_config=hf_quantization_config
    )
    return ai["generate"](
        model=model,
        prompt=prompt,
        messages=messages,
        input_string=input_string,
        system_message=system_message,
        words_in_mouth=words_in_mouth,
        max_tokens=max_tokens,
        response_model=response_model,
        tools=tools,
        temperature=temperature,
        cost_log=cost_log,
        simulate=simulate,
        **kwargs,
    )


@cache(ignore="cost_log")
async def get_llm_response_async(
    model: str = None,
    response_model: Union["BaseModel", None] = None,
    prompt: str = None,
    messages: list[dict[str, str]] = None,
    input_string: str = None,
    system_message: str | None = None,
    words_in_mouth: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    tools: list[callable] = None,
    hf_quantization_config=None,
    cost_log: Costlog = GLOBAL_COST_LOG,
    simulate: bool = SIMULATE,
    **kwargs,
):
    """NOTE: you should generally use the LLM_Agent class instead of this function.
    This is deprecated, or maybe we can just use it for one-time calls etc.
    """
    model = model or "gpt-4o-mini"
    ai = get_llm(
        model=model, use_async=True, hf_quantization_config=hf_quantization_config
    )
    async with GLOBAL_LLM_SEMAPHORE:
        return await ai["generate_async"](
            model=model,
            prompt=prompt,
            messages=messages,
            input_string=input_string,
            system_message=system_message,
            words_in_mouth=words_in_mouth,
            max_tokens=max_tokens,
            response_model=response_model,
            tools=tools,
            temperature=temperature,
            cost_log=cost_log,
            simulate=simulate,
            **kwargs,
        )


@cache(ignore="cost_log")
def get_llm_probs(
    return_probs_for: list[str],
    model: str = None,
    prompt: str = None,
    messages: list[dict[str, str]] = None,
    input_string: str = None,
    system_message: str | None = None,
    words_in_mouth: str | None = None,
    top_logprobs: int = 5,
    temperature: float = 0.0,
    hf_quantization_config=None,
    cost_log: Costlog = GLOBAL_COST_LOG,
    simulate: bool = SIMULATE,
    **kwargs,
):
    """NOTE: you should generally use the LLM_Agent class instead of this function.
    This is deprecated, or maybe we can just use it for one-time calls etc.
    """
    model = model or "gpt-4o-mini"
    ai = get_llm(
        model=model, use_async=False, hf_quantization_config=hf_quantization_config
    )
    return ai["return_probs"](
        model=model,
        return_probs_for=return_probs_for,
        prompt=prompt,
        messages=messages,
        input_string=input_string,
        system_message=system_message,
        words_in_mouth=words_in_mouth,
        top_logprobs=top_logprobs,
        temperature=temperature,
        cost_log=cost_log,
        simulate=simulate,
        **kwargs,
    )


@cache(ignore="cost_log")
async def get_llm_probs_async(
    return_probs_for: list[str],
    model: str = None,
    prompt: str = None,
    messages: list[dict[str, str]] = None,
    input_string: str = None,
    system_message: str | None = None,
    words_in_mouth: str | None = None,
    top_logprobs: int = 5,
    temperature: float = 0.0,
    hf_quantization_config=None,
    cost_log: Costlog = GLOBAL_COST_LOG,
    simulate: bool = SIMULATE,
    **kwargs,
):
    """NOTE: you should generally use the LLM_Agent class instead of this function.
    This is deprecated, or maybe we can just use it for one-time calls etc.
    """
    model = model or "gpt-4o-mini"
    ai = get_llm(
        model=model, use_async=True, hf_quantization_config=hf_quantization_config
    )
    async with GLOBAL_LLM_SEMAPHORE:
        return await ai["return_probs_async"](
            model=model,
            return_probs_for=return_probs_for,
            prompt=prompt,
            messages=messages,
            input_string=input_string,
            system_message=system_message,
            words_in_mouth=words_in_mouth,
            top_logprobs=top_logprobs,
            temperature=temperature,
            cost_log=cost_log,
            simulate=simulate,
            **kwargs,
        )
