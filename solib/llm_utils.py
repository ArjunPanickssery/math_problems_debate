from collections import defaultdict
from functools import partial
import os
import asyncio
import json
import math
from dotenv import load_dotenv
from typing import Literal, Union, Coroutine, TYPE_CHECKING
from pydantic import BaseModel
from perscache import Cache
from perscache.serializers import JSONSerializer
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
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_anthropic import ChatAnthropic


import solib.tool_use.tool_rendering
from solib.utils import apply, apply_async
from solib.datatypes import Prob
from solib.tool_use import tool_use
from solib.tool_use.tool_use import ToolStopper, HuggingFaceToolCaller

if TYPE_CHECKING:
    from transformers import AutoTokenizer


class PydanticJSONSerializer(JSONSerializer):
    @staticmethod
    def default(obj):
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        raise TypeError(
            f"Object of type {obj.__class__.__name__} is not JSON serializable"
        )

    @classmethod
    def dumps(cls, data):
        return json.dumps(data, default=cls.default).encode("utf-8")

    @classmethod
    def loads(cls, data):
        return json.loads(data.decode("utf-8"))


load_dotenv()
cache = Cache(serializer=PydanticJSONSerializer())
global_cost_log = Costlog(mode="jsonl")
simulate = os.getenv("SIMULATE", "False").lower() == "true"
disable_costly = os.getenv("DISABLE_COSTLY", "False").lower() == "true"


class LLM_Simulator(LLM_Simulator_Faker):
    @classmethod
    def _fake_custom(cls, t: type):
        assert issubclass(t, Prob)
        import random

        return t(prob=random.random())


async def parallelized_call(
    func: Coroutine,
    data: list[str],
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
        print(f"Running {func} on {len(data)} datapoints sequentially")
        return [await func(d) for d in data]

    max_concurrent_queries = min(
        max_concurrent_queries,
        int(os.getenv("MAX_CONCURRENT_QUERIES", max_concurrent_queries)),
    )

    print(
        f"Running {func} on {len(data)} datapoints with {max_concurrent_queries} concurrent queries"
    )

    # Create a local semaphore
    local_semaphore = asyncio.Semaphore(max_concurrent_queries)

    async def call_func(sem, func, datapoint):
        async with sem:
            return await func(datapoint)

    print("Calling call_func")
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
            elif (
                not natively_supports_tools
            ):  # local models (where msg_type="dict") that natively support tools will have the tool prompt added by apply_chat_template
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
    messages: list[BaseMessage | dict[str, str]]
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


def get_llm(model: str, use_async=False, hf_quantization_config=None):
    if model.startswith("hf:"):  # Hugging Face local models
        msg_type = "dict"
        from transformers import AutoTokenizer, AutoModelForCausalLM

        api_key = os.getenv("HF_TOKEN")
        model = model.split("hf:")[1]
        tokenizer = AutoTokenizer.from_pretrained(model)
        device_map = "cuda" if hf_quantization_config is not None else "auto"
        model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map=device_map,
            token=api_key,
            quantization_config=hf_quantization_config,
        )

        client = (tokenizer, model)

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
                token_enc = tokenizer.encode(token)[-1]
                if token_enc in output_probs:
                    probs[token] = output_probs[token_enc].item()
            total_prob = sum(probs.values())
            probs_relative = {token: prob / total_prob for token, prob in probs.items()}
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
            api_key = os.getenv("OPENROUTER_API_KEY")
            client = ChatOpenAI(
                model=model, base_url="https://openrouter.ai/api/v1", api_key=api_key
            )

        else:
            if model.startswith(("gpt", "openai", "babbage", "davinci")):
                api_key = os.getenv("OPENAI_API_KEY")
                client = ChatOpenAI(model=model, api_key=api_key)

            elif model.startswith(("claude", "anthropic")):
                api_key = os.getenv("ANTHROPIC_API_KEY")
                client = ChatAnthropic(model=model, api_key=api_key)

            elif model.startswith("mistral"):
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
                    print(raw)
                    raise ValueError(
                        f"Error parsing structured output: {response['parsing_error']}"
                    )

                raw_response = parsed
                usage = raw.usage_metadata

            return raw_response, usage

        _get_messages = lambda kwargs: convert_langchain_to_dict(
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
            disable_costly=disable_costly,
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
            cost_log: Costlog = global_cost_log,
            simulate: bool = simulate,
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
            print(messages)
            response = get_response(messages)

            raw_response, usage = process_response(response)
            return CostlyResponse(
                output=raw_response,
                cost_info=usage,
            )

        @costly(
            simulator=LLM_Simulator.simulate_llm_call,
            messages=_get_messages,
            disable_costly=disable_costly,
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
            cost_log: Costlog = global_cost_log,
            simulate: bool = simulate,
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
            disable_costly=disable_costly,
        )
        def return_probs(
            return_probs_for: list[str],
            model: str = model,
            prompt: str = None,
            messages: list[dict[str, str]] = None,
            system_message: str | None = None,
            top_logprobs: int = 5,
            temperature: float = 0.0,
            cost_log: Costlog = global_cost_log,
            simulate: bool = simulate,
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
            disable_costly=disable_costly,
        )
        async def return_probs_async(
            return_probs_for: list[str],
            model: str = model,
            prompt: str = None,
            messages: list[dict[str, str]] = None,
            system_message: str | None = None,
            top_logprobs: int = 5,
            temperature: float = 0.0,
            cost_log: Costlog = global_cost_log,
            simulate: bool = simulate,
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
        hf_quantization_config=None,
    ):
        self.model = model or "gpt-4o-mini"
        self.tools = tools
        self.hf_quantization_config = hf_quantization_config
        self.ai = get_llm(
            model=model, use_async=False, hf_quantization_config=hf_quantization_config
        )
        self.ai_async = get_llm(
            model=model, use_async=True, hf_quantization_config=hf_quantization_config
        )

    @cache(ignore=["self", "cost_log"])
    def get_response(
        self,
        response_model: Union["BaseModel", None] = None,
        prompt: str = None,
        messages: list[dict[str, str]] = None,
        input_string: str = None,
        system_message: str | None = None,
        words_in_mouth: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        cost_log: Costlog = global_cost_log,
        simulate: bool = simulate,
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

    @cache(ignore=["self", "cost_log"])
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
        cost_log: Costlog = global_cost_log,
        simulate: bool = simulate,
        **kwargs,
    ):
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

    @cache(ignore=["self", "cost_log"])
    def get_probs(
        self,
        return_probs_for: list[str],
        prompt: str = None,
        messages: list[dict[str, str]] = None,
        input_string: str = None,
        system_message: str | None = None,
        words_in_mouth: str | None = None,
        top_logprobs: int = 5,
        temperature: float = 0.0,
        cost_log: Costlog = global_cost_log,
        simulate: bool = simulate,
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

    @cache(ignore=["self", "cost_log"])
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
        cost_log: Costlog = global_cost_log,
        simulate: bool = simulate,
        **kwargs,
    ):
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
    cost_log: Costlog = global_cost_log,
    simulate: bool = simulate,
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
    cost_log: Costlog = global_cost_log,
    simulate: bool = simulate,
    **kwargs,
):
    """NOTE: you should generally use the LLM_Agent class instead of this function.
    This is deprecated, or maybe we can just use it for one-time calls etc.
    """
    model = model or "gpt-4o-mini"
    ai = get_llm(
        model=model, use_async=True, hf_quantization_config=hf_quantization_config
    )
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
    cost_log: Costlog = global_cost_log,
    simulate: bool = simulate,
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
    cost_log: Costlog = global_cost_log,
    simulate: bool = simulate,
    **kwargs,
):
    """NOTE: you should generally use the LLM_Agent class instead of this function.
    This is deprecated, or maybe we can just use it for one-time calls etc.
    """
    model = model or "gpt-4o-mini"
    ai = get_llm(
        model=model, use_async=True, hf_quantization_config=hf_quantization_config
    )
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
