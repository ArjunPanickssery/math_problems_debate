# adapted from https://github.com/dpaleka/consistency-forecasting/
"""
Copyright 2024 Alejandro Alvarez, Abhimanyu Pallavi Sudhir, Daniel Paleka

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation
files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from functools import partial
import os
import asyncio
import json
import math
from typing import Literal, Union, Coroutine, TYPE_CHECKING
from pydantic import BaseModel
from perscache import Cache
from perscache.serializers import JSONSerializer
from costly import Costlog, CostlyResponse, costly
from costly.simulators.llm_simulator_faker import LLM_Simulator_Faker

from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage, BaseMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_anthropic import ChatAnthropic


from solib.utils import apply, apply_async
from solib.datatypes import Prob
from solib import tool_use
from solib.tool_use import ToolStopper, HuggingFaceToolCaller

if TYPE_CHECKING:
    from transformers import AutoTokenizer


class PydanticJSONSerializer(JSONSerializer):
    @staticmethod
    def default(obj):
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

    @classmethod
    def dumps(cls, data):
        return json.dumps(data, default=cls.default).encode('utf-8')

    @classmethod
    def loads(cls, data):
        return json.loads(data.decode('utf-8'))

# Replace the existing cache initialization with this:
cache = Cache(serializer=PydanticJSONSerializer())


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
) -> dict[Literal["messages", "input_string"], str | list[dict[str, str] | BaseMessage]]:
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
        msg_type: Literal["langchain", "mistral", "dict"]: Type of messages. If "langchain", messages
            will be langchain.BaseMessages. If "mistral", messages will be in the mistral ChatMessages.
            If "dict", messages will be in the dictionary format
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

        if tools and not natively_supports_tools:   # technically, we should check msg_type here, but we're assuming all chat API models natively support tools
            messages.insert(0, {"role": "system", "content": tool_use.get_tool_prompt(tools)})

        if tokenizer is not None:
            if tools and natively_supports_tools:
                input_string = tokenizer.apply_chat_template(messages, tokenize=False, tools=tools)
            else:
                input_string = tokenizer.apply_chat_template(messages, tokenize=False)
        if words_in_mouth is not None:
            input_string += words_in_mouth
    return {"messages": messages, "input_string": input_string}


def get_llm(
    model: str,
    use_async=False,
    use_instructor: bool = False,
    tools: list[callable] = None,
    hf_quantization_config = None
):
    if model.startswith("hf:"):  # Hugging Face local models
        from transformers import AutoTokenizer, AutoModelForCausalLM

        api_key = os.getenv("HF_TOKEN")
        model = model.split("hf:")[1]
        tokenizer = AutoTokenizer.from_pretrained(model)
        device_map = 'cuda' if hf_quantization_config is not None else 'auto'
        model = AutoModelForCausalLM.from_pretrained(
            model, device_map=device_map, token=api_key, quantization_config=hf_quantization_config
        )

        natively_supports_tools = False
        if tools:
            model = HuggingFaceToolCaller(tokenizer, model, tools)
            natively_supports_tools = model.natively_supports_tools()

        client = (tokenizer, model)

        # TODO: add cost logging for local models
        def generate(
            prompt: str = None,
            messages: list[dict[str, str]] = None,
            input_string: str = None,
            system_message: str | None = None,
            words_in_mouth: str | None = None,
            max_tokens: int = 2048,
            **kwargs,
        ):
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

            input_ids = tokenizer.encode(input_string['input_string'], return_tensors="pt",
                                         add_special_tokens=False).to(model.device)

            input_length = input_ids.shape[1]
            output = model.generate(input_ids, max_length=max_tokens, do_sample=False, temperature=None, top_p=None)[0]

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
            )
            input_ids = tokenizer.encode(input_string['input_string'], return_tensors="pt").to(model.device)
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
        natively_supports_tools = True  # assume all chat API models natively support tools
        if tools is not None:
            tool_map, structured_tools = tool_use.get_structured_tools(tools)
        msg_type = 'dict' if use_instructor else 'langchain'
        input_name = "input" if msg_type == "langchain" else "messages"   # langchain uses 'input' instead of 'messages'

        if model.startswith("or:"):  # OpenRouter
            model_or = model.split("or:")[1]
            api_key = os.getenv("OPENROUTER_API_KEY")
            client = ChatOpenAI(model=model, base_url="https://openrouter.ai/api/v1", api_key=api_key)

            if use_instructor:
                import instructor
                from instructor.mode import Mode

                mode = Mode.MD_JSON
                if model_or.startswith("mistral"):
                    mode = Mode.MISTRAL_TOOLS
                elif model_or.startswith("gpt"):
                    mode = Mode.TOOLS_STRICT
                client = instructor.from_openai(client, mode=mode)
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

            if use_instructor:
                import instructor
                from instructor.mode import Mode

                mode = Mode.TOOLS
                if model.startswith("mistral"):
                    mode = Mode.MISTRAL_TOOLS
                    client = instructor.from_mistral(client, mode=mode)
                elif model.startswith("gpt"):
                    mode = Mode.TOOLS_STRICT  # use OpenAI structured outputs
                    client = instructor.from_openai(client, mode=mode)
                elif model.startswith(("anthropic", "claude")):
                    mode = Mode.ANTHROPIC_JSON
                    client = instructor.from_anthropic(client, mode=mode)

        if use_instructor:
            get_response = client.chat.completions.create_with_completion

            def process_response(response):
                raw_response, completion = response
                usage = completion.usage
                usage = {'input_tokens': usage.prompt_tokens,
                         'output_tokens': usage.completion_tokens}
                return raw_response, usage

        else:
            if tools is not None:
                client = client.bind_tools(structured_tools)
                if use_async:
                    get_response = partial(tool_use.tool_use_loop_generate_async, get_response=client.ainvoke, tool_map=tool_map)
                else:
                    get_response = partial(tool_use.tool_use_loop_generate, get_response=client.invoke, tool_map=tool_map)
            else:
                get_response = client.ainvoke if use_async else client.invoke

            def process_response(response):
                if isinstance(response, list):   # handle tool calling case, make the output match the hugging face case somewhat
                    raw_response = tool_use.render_tool_call_conversation(response)
                    usage = {k: sum([r.usage_metadata[k] for r in response
                                     if hasattr(r, 'usage_metadata')]) for k in response[0].usage_metadata}
                else:
                    raw_response = response.content
                    usage = response.usage_metadata

                return raw_response, usage


        _get_messages = lambda kwargs: format_prompt(
            prompt=kwargs.get("prompt"),
            messages=kwargs.get("messages"),
            system_message=kwargs.get("system_message"),
            msg_type=msg_type
        )["messages"]

        @costly(simulator=LLM_Simulator.simulate_llm_call, messages=_get_messages)
        def generate(
            model: str = model,
            prompt: str = None,
            messages: list[dict[str, str]] = None,
            system_message: str | None = None,
            max_tokens: int | None = None,
            response_model: Union["BaseModel", None] = None,
            temperature: float = 0.0,
            **kwargs,
        ):
            messages = format_prompt(
                prompt=prompt,
                messages=messages,
                system_message=system_message,
                msg_type=msg_type
            )["messages"]

            response = apply(
                get_response,
                **{
                    input_name: messages,
                    "model": model,
                    "max_tokens": max_tokens,
                    "response_model": response_model,
                    "temperature": temperature,
                },
            )

            raw_response, usage = process_response(response)
            return CostlyResponse(
                output=raw_response,
                cost_info=usage,
            )

        @costly(simulator=LLM_Simulator.simulate_llm_call, messages=_get_messages)
        async def generate_async(
            model: str = model,
            prompt: str = None,
            messages: list[dict[str, str]] = None,
            system_message: str | None = None,
            max_tokens: int | None = None,
            response_model: Union["BaseModel", None] = None,
            temperature: float = 0.0,
            **kwargs,
        ):
            messages = format_prompt(
                prompt=prompt,
                messages=messages,
                system_message=system_message,
                msg_type=msg_type
            )["messages"]

            response = await apply_async(
                get_response,
                **{
                    input_name: messages,
                    "model": model,
                    "max_tokens": max_tokens,
                    "response_model": response_model,
                    "temperature": temperature,
                },
            )

            raw_response, usage = process_response(response)
            return CostlyResponse(
                output=raw_response,
                cost_info=usage,
            )

        @costly(simulator=LLM_Simulator.simulate_llm_probs, messages=_get_messages)
        def return_probs(
            return_probs_for: list[str],
            model: str = model,
            prompt: str = None,
            messages: list[dict[str, str]] = None,
            system_message: str | None = None,
            top_logprobs: int = 5,
            temperature: float = 0.0,
            **kwargs,
        ):
            messages = format_prompt(
                prompt=prompt,
                messages=messages,
                system_message=system_message,
                msg_type=msg_type
            )["messages"]

            max_tokens = max(len(token) for token in return_probs_for)

            response = apply(
                get_response,
                **{
                    input_name: messages,
                    "model": model,
                    "max_tokens": max_tokens,
                    "logprobs": True,
                    "top_logprobs": top_logprobs,
                    "temperature": temperature,
                },
            )
            raw_response, usage = process_response(response)
            all_logprobs = response.choices[0].logprobs.content[0].top_logprobs
            all_logprobs_dict = {x.token: x.logprob for x in all_logprobs}
            probs = {token: 0 for token in return_probs_for}
            for token, prob in all_logprobs_dict.items():
                if token in probs:
                    probs[token] = math.exp(prob)
            total_prob = sum(probs.values())
            probs_relative = {token: prob / total_prob for token, prob in probs.items()}
            return CostlyResponse(
                output=probs_relative,
                cost_info=usage,
            )

        @costly(simulator=LLM_Simulator.simulate_llm_probs, messages=_get_messages)
        async def return_probs_async(
            return_probs_for: list[str],
            model: str = model,
            prompt: str = None,
            messages: list[dict[str, str]] = None,
            system_message: str | None = None,
            top_logprobs: int = 5,
            temperature: float = 0.0,
            **kwargs,
        ):
            messages = format_prompt(
                prompt=prompt,
                messages=messages,
                system_message=system_message,
                msg_type=msg_type
            )["messages"]

            max_tokens = max(len(token) for token in return_probs_for)

            response = await apply_async(
                get_response,
                **{
                    input_name: messages,
                    "model": model,
                    "max_tokens": max_tokens,
                    "logprobs": True,
                    "top_logprobs": top_logprobs,
                    "temperature": temperature,
                },
            )
            raw_response, usage = process_response(response)
            all_logprobs = response.choices[0].logprobs.content[0].top_logprobs
            all_logprobs_dict = {x.token: x.logprob for x in all_logprobs}
            probs = {token: 0 for token in return_probs_for}
            for token, prob in all_logprobs_dict.items():
                if token in probs:
                    probs[token] = math.exp(prob)
            total_prob = sum(probs.values())
            probs_relative = {token: prob / total_prob for token, prob in probs.items()}
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


@cache(ignore="cost_log")
def get_llm_response(
    model: str = "gpt-4o-mini",
    response_model: Union["BaseModel", None] = None,
    prompt: str = None,
    messages: list[dict[str, str]] = None,
    input_string: str = None,
    system_message: str | None = None,
    words_in_mouth: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    cost_log: Costlog = None,  # need to give explicitly b/c cache
    tools: list[callable] = None,
    hf_quantization_config = None,
    **kwargs,  # kwargs necessary for costly
):
    model = model or "gpt-4o-mini"
    ai = get_llm(
        model=model,
        use_async=False,
        use_instructor=(response_model is not None),
        tools=tools,
        hf_quantization_config=hf_quantization_config
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
        temperature=temperature,
        cost_log=cost_log,
        **kwargs,
    )


@cache(ignore="cost_log")
async def get_llm_response_async(
    model: str = "gpt-4o-mini",
    response_model: Union["BaseModel", None] = None,
    prompt: str = None,
    messages: list[dict[str, str]] = None,
    input_string: str = None,
    system_message: str | None = None,
    words_in_mouth: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    cost_log: Costlog = None,  # need to give explicitly b/c cache
    tools: list[callable] = None,
    hf_quantization_config = None,
    **kwargs,
):
    model = model or "gpt-4o-mini"
    ai = get_llm(
        model=model,
        use_async=True,
        use_instructor=(response_model is not None),
        tools=tools,
        hf_quantization_config=hf_quantization_config
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
        temperature=temperature,
        cost_log=cost_log,
        **kwargs,
    )


@cache(ignore="cost_log")
def get_llm_probs(
    return_probs_for: list[str],
    model: str = "gpt-4o-mini",
    prompt: str = None,
    messages: list[dict[str, str]] = None,
    input_string: str = None,
    system_message: str | None = None,
    words_in_mouth: str | None = None,
    top_logprobs: int = 5,
    temperature: float = 0.0,
    cost_log: Costlog = None,  # need to give explicitly b/c cache
    tools: list[callable] = None,
    hf_quantization_config = None,
    **kwargs,
):
    model = model or "gpt-4o-mini"
    ai = get_llm(
        model=model,
        use_async=False,
        use_instructor=False,
        tools=tools,
        hf_quantization_config=hf_quantization_config
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
        **kwargs,
    )

@cache(ignore="cost_log")
async def get_llm_probs_async(
    return_probs_for: list[str],
    model: str = "gpt-4o-mini",
    prompt: str = None,
    messages: list[dict[str, str]] = None,
    input_string: str = None,
    system_message: str | None = None,
    words_in_mouth: str | None = None,
    top_logprobs: int = 5,
    temperature: float = 0.0,
    cost_log: Costlog = None,  # need to give explicitly b/c cache
    tools: list[callable] = None,
    hf_quantization_config = None,
    **kwargs,
):
    model = model or "gpt-4o-mini"
    ai = get_llm(
        model=model,
        use_async=True,
        use_instructor=False,
        tools=tools,
        hf_quantization_config=hf_quantization_config
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
        **kwargs,
    )
