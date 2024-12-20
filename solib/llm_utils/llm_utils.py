from collections import defaultdict
from functools import partial
import functools
import os
import math
from typing import Literal, Union, TYPE_CHECKING, List, Dict, Any, Optional
import requests
from transformers import BitsAndBytesConfig
from pydantic import BaseModel
import instructor
from instructor import Mode
from openai import OpenAI, AsyncOpenAI
from costly import Costlog, CostlyResponse, costly
from costly.simulators.llm_simulator_faker import LLM_Simulator_Faker
import warnings
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_random_exponential,
    after_log,
)
import logging

from solib.globals import *
from solib.llm_utils.rate_limiter import get_rate_limiter

from solib.llm_utils.caching import method_cache
import solib.tool_use.tool_rendering
from solib.datatypes import Prob
from solib.tool_use import tool_use
from solib.tool_use.tool_use import HuggingFaceToolCaller

if TYPE_CHECKING:
    from transformers import AutoTokenizer


class LLM_Simulator(LLM_Simulator_Faker):
    @classmethod
    def _fake_custom(cls, t: type):
        assert issubclass(t, Prob)
        import random

        return t(prob=random.random())


def format_prompt(
    prompt: str = None,
    messages: List[Dict[str, Any]] = None,
    input_string: str = None,
    tokenizer: Union["AutoTokenizer", None] = None,
    system_message: str | None = None,
    words_in_mouth: str | None = None,
    tools: list[callable] = None,
    natively_supports_tools: bool = False,
) -> dict[
    Literal["messages", "input_string"], str | List[Dict[str, Any]]
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
    Returns:
        dict: with keys "messages" and "input_string". "input_string" will be None
            if tokenizer is None.
    """
    if input_string is None:
        if messages is None:
            assert prompt is not None

            messages = []
            if system_message is not None:
                messages.append({"role": "system", "content": system_message})

            messages.append({"role": "user", "content": prompt})

        if tools:
            tool_msg = solib.tool_use.tool_rendering.get_tool_prompt(
                tools, natively_supports_tools
            )
            if not natively_supports_tools:
                messages.insert(0, {"role": "system", "content": tool_msg})

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


# cache this because it's surprisingly slow
@functools.cache
def load_api_model(model):
    if model.startswith("or:"): # OpenRouter
        base_url = "https://openrouter.ai/api/v1"
        api_key = os.getenv("OPENROUTER_API_KEY")
        client = instructor.patch(OpenAI(base_url=base_url, api_key=api_key), mode=Mode.TOOLS)
    else:
        if model.startswith(("gpt", "openai", "babbage", "davinci")):
            api_key = os.getenv("OPENAI_API_KEY") 
            client = instructor.patch(OpenAI(api_key=api_key), mode=Mode.TOOLS)
            
        elif model.startswith(("claude", "anthropic")):
            # Anthropic has its own client but can be used similarly
            from anthropic import Anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            client = instructor.patch(Anthropic(api_key=api_key), mode=Mode.TOOLS)
            
        elif model.startswith("mistral"):
            from mistralai.client import MistralClient
            api_key = os.getenv("MISTRAL_API_KEY")
            client = instructor.patch(MistralClient(api_key=api_key), mode=Mode.TOOLS)
            
        else:
            raise ValueError(f"Model {model} is not supported for now")

    return client


def get_hf_llm(model: str, hf_quantization_config=True):
    client = load_hf_model(model, hf_quantization_config)
    tokenizer, model = client

    def generate_format_and_bind(
        prompt: str = None,
        messages: list[dict] = None,
        input_string: str = None,
        system_message: str | None = None,
        words_in_mouth: str | None = None,
        tools: list[callable] = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        response_model: Union["BaseModel", None] = None,
        top_logprobs: int = 0,
        use_async=False,
    ):
        messages = format_prompt(
            prompt=prompt,
            messages=messages,
            input_string=input_string,
            tokenizer=tokenizer,
            system_message=system_message,
            words_in_mouth=words_in_mouth,
            tools=tools,
            natively_supports_tools=False,
        )["messages"]

        return messages

    @costly(
        simulator=LLM_Simulator.simulate_llm_call,
        messages=lambda kwargs: generate_format_and_bind(
            prompt=kwargs.get("prompt"),
            messages=kwargs.get("messages"),
            input_string=kwargs.get("input_string"),
            system_message=kwargs.get("system_message"),
        ),
        disable_costly=DISABLE_COSTLY,
    )
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

    @costly(
        simulator=LLM_Simulator.simulate_llm_call,
        messages=lambda kwargs: generate_format_and_bind(
            prompt=kwargs.get("prompt"),
            messages=kwargs.get("messages"),
            input_string=kwargs.get("input_string"),
            system_message=kwargs.get("system_message"),
        ),
        disable_costly=DISABLE_COSTLY,
    )
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
            probs_relative = {token: prob / total_prob for token, prob in probs.items()}
        except ZeroDivisionError:
            import pdb

            pdb.set_trace()
        return probs_relative

    return {
        "client": client,
        "generate": generate,
        "return_probs": return_probs,
    }


def get_api_llm(model: str):
    natively_supports_tools = True
    client = load_api_model(model)
    rate_limiter = get_rate_limiter(model)

    def process_response(response, response_model=None):
        """Process response from instructor/OpenAI client"""
        if isinstance(response, list): # Tool calls
            raw_response = solib.tool_use.tool_rendering.render_tool_call_conversation(response)
            usage = response[0].usage
        else:
            raw_response = response.choices[0].message.content
            usage = response.usage
            
            if response_model:
                parsed = instructor.validate(response_model, raw_response)
                raw_response = parsed

        return raw_response, usage

    _get_messages = lambda kwargs: format_prompt(  # noqa
        format_prompt(
            prompt=kwargs.get("prompt"),
            messages=kwargs.get("messages"),
            system_message=kwargs.get("system_message"),
            natively_supports_tools=natively_supports_tools,
        )["messages"]
    )

    def apply_client_bindings(
        tools: list[callable] = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        response_model: Union["BaseModel", None] = None,
        top_logprobs: int = 0,
        use_async=False,
    ):
        if tools and response_model:
            raise ValueError("Cannot use tools with response_model")

        completion_kwargs = {
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if top_logprobs:
            completion_kwargs["top_logprobs"] = top_logprobs
            completion_kwargs["logprobs"] = True

        if response_model:
            patched_client = instructor.patch(client, mode=Mode.TOOLS)
            get_response = partial(
                patched_client.chat.completions.create,
                response_model=response_model,
                **completion_kwargs
            )
        else:
            get_response = partial(client.chat.completions.create, **completion_kwargs)

        if tools:
            tool_map, structured_tools = (
                solib.tool_use.tool_rendering.get_structured_tools(tools)
            )
            completion_kwargs["tools"] = structured_tools
            completion_kwargs["tool_choice"] = "auto"

        if use_async:
            get_response = partial(client.chat.completions.create, **completion_kwargs)

        def rate_limited_response(*args, **kwargs):
            rate_limiter.acquire()
            return get_response(*args, **kwargs)

        get_response = retry(
            wait=wait_random_exponential(multiplier=0.5, max=60),
            after=after_log(LOGGER, logging.DEBUG),
            before_sleep=before_sleep_log(LOGGER, logging.DEBUG, exc_info=True),
        )(rate_limited_response)

        if tools:
            if use_async:
                get_response = partial(
                    tool_use.tool_use_loop_generate_async,
                    get_response=get_response,
                    tool_map=tool_map,
                )
            else:
                get_response = partial(
                    tool_use.tool_use_loop_generate,
                    get_response=get_response,
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
        use_async,
        **kwargs,
    ):
        if kwargs.get("words_in_mouth"):
            warnings.warn(
                f"words_in_mouth is not supported for model type `{model}`",
                UserWarning,
            )

        get_response = apply_client_bindings(
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            response_model=response_model,
            use_async=use_async,
        )

        messages = format_prompt(
            prompt=prompt,
            messages=messages,
            system_message=system_message,
            tools=tools,
            natively_supports_tools=natively_supports_tools,
        )["messages"]

        return get_response, messages

    @costly(
        simulator=LLM_Simulator.simulate_llm_call,
        messages=lambda kwargs: generate_format_and_bind(
            prompt=kwargs.get("prompt"),
            messages=kwargs.get("messages"),
            system_message=kwargs.get("system_message"),
        ),
        disable_costly=DISABLE_COSTLY,
    )
    @method_cache(ignore=["cost_log"])
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
        # Store response_model and clear it from kwargs to ensure proper caching
        _response_model = response_model
        kwargs.pop('response_model', None)

        get_response, messages = generate_format_and_bind(
            prompt,
            messages,
            system_message,
            max_tokens,
            None,  # Don't pass response_model to avoid caching issues
            tools,
            temperature,
            use_async=False,
            **kwargs,
        )
        LOGGER.debug(messages)
        response = get_response(messages)

        raw_response, usage = process_response(response)

        LOGGER.debug(f"raw_response: {raw_response}")

        if _response_model is not None:
            if isinstance(raw_response, dict):
                raw_response = _response_model(**raw_response)
            else:
                # Parse the raw string response into the model
                raw_response = instructor.validate(_response_model, raw_response)

        LOGGER.debug(f"raw_response: {raw_response}")

        return CostlyResponse(
            output=raw_response,
            cost_info=usage,
        )

    @costly(
        simulator=LLM_Simulator.simulate_llm_call,
        messages=lambda kwargs: generate_format_and_bind(
            prompt=kwargs.get("prompt"),
            messages=kwargs.get("messages"),
            system_message=kwargs.get("system_message"),
        ),
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
        # Store response_model and clear it from kwargs to ensure proper caching
        _response_model = response_model
        kwargs.pop('response_model', None)

        get_response, messages = generate_format_and_bind(
            prompt,
            messages,
            system_message,
            max_tokens,
            None,  # Don't pass response_model to avoid caching issues
            tools,
            temperature,
            use_async=True,
            **kwargs,
        )

        response = await get_response(messages)

        raw_response, usage = process_response(response)

        LOGGER.debug(f"raw_response: {raw_response}")

        if _response_model is not None:
            if isinstance(raw_response, dict):
                raw_response = _response_model(**raw_response)
            else:
                # Parse the raw string response into the model
                raw_response = instructor.validate(_response_model, raw_response)

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
        probs = defaultdict(float)  # {token: 0 for token in return_probs_for}
        for token in return_probs_for:
            if token in all_logprobs_dict:
                probs[token] = math.exp(all_logprobs_dict[token])
            elif f"({token}" in all_logprobs_dict:
                probs[token] = math.exp(all_logprobs_dict[f"({token}"])
        total_prob = sum(probs.values())
        try:
            probs_relative = {
                token: probs[token] / total_prob for token in return_probs_for
            }
        except ZeroDivisionError:
            import pdb

            pdb.set_trace()
        return probs_relative

    def probability_format_and_bind(
        prompt,
        messages,
        system_message,
        return_probs_for,
        top_logprobs,
        temperature,
        use_async,
        **kwargs,
    ):
        max_tokens = max(len(token) for token in return_probs_for)

        get_response = apply_client_bindings(
            max_tokens=max_tokens,
            temperature=temperature,
            top_logprobs=top_logprobs,
            use_async=use_async,
        )

        messages = format_prompt(
            prompt=prompt,
            messages=messages,
            system_message=system_message,
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
            use_async=False,
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
            use_async=True,
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


def get_llm(model: str, hf_quantization_config=True):
    if model.startswith("hf:"):  # Hugging Face local models
        return get_hf_llm(model, hf_quantization_config)
    else:
        return get_api_llm(model)


class LLM_Agent:
    def __init__(
        self,
        model: str = None,
        tools: list[callable] = None,
        hf_quantization_config=True,
        sync_mode=False,
    ):
        self.model = model or "gpt-4o-mini"
        self.tools = tools
        self.hf_quantization_config = hf_quantization_config
        self.sync_mode = sync_mode

        self.ai = get_llm(
            model=self.model,
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
        cache_breaker: int = 0,
        cost_log: Costlog = GLOBAL_COST_LOG,
        simulate: bool = SIMULATE,
        **kwargs,
    ):
        if not simulate:
            LOGGER.info(f"Running get_response_sync for {self.model}; NOT FROM CACHE")
            pass

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
        cache_breaker: int = 0,
        cost_log: Costlog = GLOBAL_COST_LOG,
        simulate: bool = SIMULATE,
        **kwargs,
    ):
        if not simulate:
            LOGGER.info(f"Running get_response_async for {self.model}; NOT FROM CACHE")
            pass

        return await self.ai["generate_async"](
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
        if self.supports_async and not self.sync_mode:
            return await self.get_response_async(*args, **kwargs)
        return self.get_response_sync(*args, **kwargs)

    async def get_probs(self, *args, **kwargs):
        if self.supports_async and not self.sync_mode:
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
        cache_breaker: int = 0,
        cost_log: Costlog = GLOBAL_COST_LOG,
        simulate: bool = SIMULATE,
        **kwargs,
    ):
        if not simulate:
            LOGGER.info(f"Running get_probs_sync for {self.model}; NOT FROM CACHE")
            pass

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
        cache_breaker: int = 0,
        cost_log: Costlog = GLOBAL_COST_LOG,
        simulate: bool = SIMULATE,
        **kwargs,
    ):
        if not simulate:
            LOGGER.info(f"Running get_probs_async for {self.model}; NOT FROM CACHE")
            pass

        return await self.ai["return_probs_async"](
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
