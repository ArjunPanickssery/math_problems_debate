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

import os
import math
import sys
import asyncio
from dataclasses import dataclass
from perscache import Cache
from perscache.serializers import JSONSerializer
from contextlib import contextmanager
from typing import Literal
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from mistralai.async_client import MistralAsyncClient
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from anthropic import AsyncAnthropic, Anthropic
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM  # AutoModelForSeq2SeqLM
import transformers
import instructor
from instructor.client import Instructor
from instructor.mode import Mode

cache = Cache(
    serializer=JSONSerializer(),
)


class CostEstimator:

    class APIItem:

        # input_tokens: dollar per input token
        # output_tokens: dollar per output token
        # time: seconds per output token
        prices = {
            "gpt-4o": {
                "input_tokens": 5.0e-6,
                "output_tokens": 15.0e-6,
                "time": 18e-3,
            },
            "gpt-4o-mini": {
                "input_tokens": 0.15e-6,
                "output_tokens": 0.6e-6,
                "time": 9e-3,
            },
        }

        def __init__(
            self,
            model: str,
            input_tokens: int,
            output_tokens: list[int],
            description: str = "",
        ):
            self.model = model
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens
            self.description = description

        @property
        def cost(self) -> list[float]:
            return [
                self.input_tokens * self.prices[self.model]["input_tokens"]
                + output_tokens_bound * self.prices[self.model]["output_tokens"]
                for output_tokens_bound in self.output_tokens
            ]

        @property
        def time(self) -> list[float]:
            return [
                output_tokens_bound * self.prices[self.model]["time"]
                for output_tokens_bound in self.output_tokens
            ]

    class ManualItem:

        def __init__(self, cost: list[float], time: list[float], description: str = ""):
            self.cost = cost
            self.time = time
            self.description = description

    def __init__(self):
        self.log = []
        self.calls = 0
        self.cost = [0.0, 0.0]
        self.time = [0.0, 0.0]

    @contextmanager
    def add_api_item(
        self, model: str, input_tokens: int, output_tokens: list[int], description: str = ""
    ):
        item = self.APIItem(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            description=description,
        )
        self.log.append(item)
        self.calls += 1
        self.cost[0] += item.cost[0]
        self.cost[1] += item.cost[1]
        self.time[0] += item.time[0]
        self.time[1] += item.time[1]
        yield

    @contextmanager
    def add_manual_item(self, cost: list[float], time: list[float], description: str = ""):
        item = self.ManualItem(cost=cost, time=time, description=description)
        self.log.append(item)
        self.cost[0] += item.cost[0]
        self.cost[1] += item.cost[1]
        self.time[0] += item.time[0]
        self.time[1] += item.time[1]
        yield
        

    def report(self):
        print(f"Total cost in range: {self.cost[0]:.2f} - {self.cost[1]:.2f} USD")
        print(f"Total time in range: {self.time[0]:.2f} - {self.time[1]:.2f} sec")
        print(f"Total calls: {self.calls}")
        print("Breakdown:")
        for item in self.log:
            if isinstance(item, self.APIItem):
                print(
                    f"API call to {item.model}:\n"
                    f"Input tokens: {item.input_tokens}, "
                    f"Output tokens in range: {item.output_tokens}\n"
                    f"Cost in range: {item.cost[0]:.2f} - {item.cost[1]:.2f} USD, "
                    f"Time in range: {self.time[0]:.2f} - {self.time[1]:.2f} sec\n"
                    f"Description: {item.description}\n"
                )
            else:
                print(
                    f"Manual item:\n"
                    f"Cost in range: {item.cost[0]:.2f} - {item.cost[1]:.2f} USD\n"
                    f"Time in range: {item.time[0]:.2f} - {item.time[1]:.2f} USD\n"
                    f"Description: {item.description}\n"
                )


def get_async_openai_client_pydantic() -> Instructor:
    api_key = os.getenv("OPENAI_API_KEY")
    _client = AsyncOpenAI(api_key=api_key)
    return instructor.from_openai(_client)


def get_async_openai_client_native() -> AsyncOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    return AsyncOpenAI(api_key=api_key)


def get_openai_client_pydantic() -> Instructor:
    api_key = os.getenv("OPENAI_API_KEY")
    _client = OpenAI(api_key=api_key)
    return instructor.from_openai(_client)


def get_openai_client_native() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=api_key)


def get_async_openrouter_client_pydantic(**kwargs) -> Instructor:
    print(
        "Only some OpenRouter endpoints have `response_format`. If you encounter errors, please check on the OpenRouter website."
    )
    api_key = os.getenv("OPENROUTER_API_KEY")
    print(f"OPENROUTER_API_KEY: {api_key}")
    _client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    return instructor.from_openai(_client, mode=Mode.MD_JSON, **kwargs)


def get_async_openrouter_client_native() -> AsyncOpenAI:
    print("Calling models through OpenRouter")
    api_key = os.getenv("OPENROUTER_API_KEY")
    print(f"OPENROUTER_API_KEY: {api_key}")
    return AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def get_openrouter_client_pydantic(**kwargs) -> Instructor:
    print(
        "Only some OpenRouter endpoints have `response_format`. If you encounter errors, please check on the OpenRouter website."
    )
    print("Calling models through OpenRouter")
    _client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    print(f"OPENROUTER_API_KEY: {os.getenv('OPENROUTER_API_KEY')}")
    return instructor.from_openai(_client, mode=Mode.TOOLS, **kwargs)


def get_openrouter_client_native() -> OpenAI:
    print("Calling models through OpenRouter")
    api_key = os.getenv("OPENROUTER_API_KEY")
    print(f"OPENROUTER_API_KEY: {api_key}")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def get_mistral_async_client_pydantic() -> Instructor:
    api_key = os.getenv("MISTRAL_API_KEY")
    _client = MistralAsyncClient(api_key=api_key)
    return instructor.from_openai(_client, mode=instructor.Mode.MISTRAL_TOOLS)


def get_mistral_async_client_native() -> MistralAsyncClient:
    api_key = os.getenv("MISTRAL_API_KEY")
    return MistralAsyncClient(api_key=api_key)


def get_mistral_client_pydantic() -> Instructor:
    api_key = os.getenv("MISTRAL_API_KEY")
    _client = MistralClient(api_key=api_key)
    return instructor.from_openai(_client, mode=instructor.Mode.MISTRAL_TOOLS)


def get_mistral_client_native() -> MistralClient:
    api_key = os.getenv("MISTRAL_API_KEY")
    return MistralClient(api_key=api_key)


def get_anthropic_async_client_pydantic() -> Instructor:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    _client = AsyncAnthropic(api_key=api_key)
    return instructor.from_anthropic(_client, mode=instructor.Mode.ANTHROPIC_JSON)


def get_anthropic_async_client_native() -> AsyncAnthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    return AsyncAnthropic(api_key=api_key)


def get_anthropic_client_pydantic() -> Instructor:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    _client = Anthropic(api_key=api_key)
    return instructor.from_anthropic(_client, mode=instructor.Mode.ANTHROPIC_JSON)


def get_anthropic_client_native() -> Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    return Anthropic(api_key=api_key)


def get_togetherai_client_native() -> OpenAI:
    url = "https://api.together.xyz/v1"
    api_key = os.getenv("TOGETHER_API_KEY")
    return OpenAI(api_key=api_key, base_url=url)


def get_huggingface_local_client(hf_repo):
    # hf_model_path = os.path.join(os.getenv("HF_MODELS_DIR"), hf_repo)
    # if not os.path.exists(hf_model_path):
    #     snapshot_download(hf_repo, local_dir=hf_model_path)

    tokenizer = AutoTokenizer.from_pretrained(hf_repo)

    model = AutoModelForCausalLM.from_pretrained(
        hf_repo, device_map="auto", token=os.getenv("HF_TOKEN")
    )
    # pipeline = transformers.pipeline(
    #     "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2048
    # )
    # return pipeline
    return tokenizer, model


def is_openai(model: str) -> bool:
    keywords = [
        "ft:gpt",
        "gpt-4o",
        "gpt-4",
        "gpt-3.5",
        "babbage",
        "davinci",
        "openai",
        "open-ai",
    ]
    return any(keyword in model for keyword in keywords)


def is_mistral(model: str) -> bool:
    if model.startswith("mistral"):
        return True


def is_anthropic(model: str) -> bool:
    keywords = ["anthropic", "claude"]
    return any(keyword in model for keyword in keywords)


def is_huggingface_local(model: str) -> bool:
    keywords = ["huggingface", "hf"]
    return any(keyword in model for keyword in keywords)


def get_provider(model: str) -> str:
    if is_openai(model):
        return "openai"
    elif is_mistral(model):
        return "mistral"
    elif is_anthropic(model):
        return "anthropic"
    elif is_huggingface_local(model):
        return "huggingface_local"
    else:
        raise NotImplementedError(f"Model {model} is not supported for now")


def get_client_native(
    model: str, use_async=True
) -> tuple[AsyncOpenAI | OpenAI | MistralAsyncClient | MistralClient, str]:
    provider = get_provider(model)

    if os.getenv("USE_OPENROUTER"):
        client = (
            get_async_openrouter_client_native()
            if use_async
            else get_openrouter_client_native()
        )
    elif provider == "openai":
        client = (
            get_async_openai_client_native()
            if use_async
            else get_openai_client_native()
        )
    elif provider == "mistral":
        client = (
            get_mistral_async_client_native()
            if use_async
            else get_mistral_client_native()
        )
    elif provider == "huggingface_local":
        assert model.startswith("hf:")
        model = model.split("hf:")[1]
        client = get_huggingface_local_client(model)
    else:
        raise NotImplementedError(f"Model {model} is not supported for now")

    return client, provider


def get_client_pydantic(model: str, use_async=True) -> tuple[Instructor, str]:
    provider = get_provider(model)
    if provider == "togetherai" and "nitro" not in model:
        raise NotImplementedError(
            "Most models on TogetherAI API, and the same models on OpenRouter API too, do not support function calling / JSON output mode. So, no Pydantic outputs for now. The exception seem to be Nitro-hosted models on OpenRouter."
        )

    use_openrouter = (
        os.getenv("USE_OPENROUTER") and os.getenv("USE_OPENROUTER") != "False"
    )
    if use_openrouter:
        kwargs = {}
        if provider == "mistral":
            # https://python.useinstructor.com/hub/mistral/
            print(
                "Only some Mistral endpoints have `response_format` on OpenRouter. If you encounter errors, please check on the OpenRouter website."
            )
            kwargs["mode"] = instructor.Mode.MISTRAL_TOOLS
        elif provider == "anthropic":
            raise NotImplementedError(
                "Anthropic over OpenRouter does not work as of June 4 2024"
            )
        client = (
            get_async_openrouter_client_pydantic(**kwargs)
            if use_async
            else get_openrouter_client_pydantic(**kwargs)
        )
    elif provider == "openai":
        client = (
            get_async_openai_client_pydantic()
            if use_async
            else get_openai_client_pydantic()
        )
    elif provider == "mistral":
        client = (
            get_mistral_async_client_pydantic()
            if use_async
            else get_mistral_client_pydantic()
        )
    elif provider == "anthropic":
        client = (
            get_anthropic_async_client_pydantic()
            if use_async
            else get_anthropic_client_pydantic()
        )
    else:
        raise NotImplementedError(f"Model {model} is not supported for now")

    return client, provider


def is_llama2_tokenized(model: str) -> bool:
    keywords = ["Llama-2", "pythia"]
    return any(keyword in model for keyword in keywords)


def _mistral_message_transform(messages):
    mistral_messages = []
    for message in messages:
        mistral_message = ChatMessage(role=message["role"], content=message["content"])
        mistral_messages.append(mistral_message)
    return mistral_messages


def prepare_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]


def get_hf_response(
    prompt: str | list[dict[str, str]],
    model_name: str,
    max_tokens: int | None = None,
    return_probs_for: list[str] | None = None,
    prompt_type: Literal["plain", "messages", "formatted"] = "plain",
    words_in_mouth: str | None = None,
) -> str | dict[str, float]:
    """
    Get response from Hugging Face model.

    Args:
        prompt: str | list[dict[str, str]]: Prompt to send to the LLM.
        model_name: str: Name of the Hugging Face model to use.
        max_tokens: int | None: Maximum number of tokens to generate.
        return_probs_for: list[str] | None: List of tokens to return relative probabilities for.
            If None, simply returns the text response.
        prompt_type: Literal["plain", "messages", "formatted"]: Type of prompt.
            "plain" is a plain question, which needs to be formatted into the HF model's syntax.
            "messages" means the prompt is a list[dict] as in the ChatCompletions API.
            "formatted" means the prompt should be sent as-is to the HF model.
        words_in_mouth: str | None: Words to append to the prompt.
            E.g. " Sure, here's my response:\n\n"
    """
    hf_repo = model_name.split("hf:")[1]
    tokenizer, model = get_huggingface_local_client(hf_repo)
    if words_in_mouth is None:
        words_in_mouth = ""
    if prompt_type == "plain":
        assert isinstance(prompt, str)
        prompt = prepare_messages(prompt)
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False)
    elif prompt_type == "messages":
        assert isinstance(prompt, list)
        assert isinstance(prompt[0], dict)
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False)
    elif prompt_type == "formatted":
        assert isinstance(prompt, str)
        ...
    prompt += words_in_mouth
    if max_tokens is None:
        max_tokens = 2048

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    if return_probs_for:
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
    else:
        output = model.generate(input_ids, max_length=max_tokens)
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        prompt_end = prompt[-9:]
        response = decoded.split(prompt_end)[-1]
        return response


@cache
async def get_llm_response_async(
    prompt: str | list[dict[str, str]],
    model: str | None = None,
    return_probs_for: list[str] | None = None,
    max_tokens: int | None = None,
    words_in_mouth: str | None = None,
    verbose:bool=False,
    simulate:bool|int=False,
    cost_estimator:CostEstimator=None,
    **kwargs,
) -> str | dict[str, float]:
    """
    Get LLM response to a prompt.

    Args:
        prompt: str | list[dict[str, str]]: Prompt to send to the LLM.
        verbose: bool: Whether to print the prompt and response.
        model: str | None: Model to use for the LLM. Defaults to "gpt-4o-mini".
        return_probs_for: list[str] | None: List of tokens to return relative probabilities for.
            If None, simply returns the text response.
        max_tokens: int | None: Maximum number of tokens to generate.
        words_in_mouth: str | None: Words to append to the prompt, only used for huggingface models.
            E.g. " Sure, here's my response:\n\n"
        verbose: bool: Whether to print the prompt and response.
        simulate: bool | int: Whether to simulate the response. If int, number of tokens to return
            in output. Defaults to False.
        cost_estimator: CostEstimator: CostEstimator object to use for cost estimation.

    Keyword Args:
        response_model: pydantic.BaseModel: Pydantic model to use for response, if using
            instructor. Defaults to None (in which case instructor is not used).
        top_logprobs: int: Number of top logprobs to return. Defaults to 5.
    """

    print("NOT USING CACHE")

    default_options = {
        "model": "gpt-4o-mini",
    }
    options = default_options | kwargs
    options["model"] = model or options["model"]
    if isinstance(prompt, str):
        prompt = prepare_messages(prompt)
    if "response_model" in options:
        client, client_name = get_client_pydantic(options["model"], use_async=True)
    else:
        client, client_name = get_client_native(options["model"], use_async=True)
    call_messages = (
        _mistral_message_transform(prompt) if client_name == "mistral" else prompt
    )
    
    if simulate and cost_estimator:
        with cost_estimator.add_api_item(
            model=options["model"],
            input_tokens=len("".join([m["content"] for m in prompt])) // 3,
            output_tokens=[1, max_tokens],
        ):
            if return_probs_for:
                return {token: 0 for token in return_probs_for}
            simstr = "This is a test output."
            return simstr * (simulate // (len(simstr) // 3))

    if client_name == "huggingface_local":
        return get_hf_response(
            prompt=call_messages,
            model_name=options["model"],
            max_tokens=max_tokens,
            return_probs_for=return_probs_for,
            prompt_type="messages",
            words_in_mouth=words_in_mouth,
        )
    elif client_name == "mistral" and not os.getenv("USE_OPENROUTER"):
        response = await client.chat(
            messages=call_messages,
            max_tokens=max_tokens,
            logprobs=bool(return_probs_for),
            top_logprobs=(5 if return_probs_for else None),
            **options,
        )
    else:
        response = await client.chat.completions.create(
            messages=call_messages,
            max_tokens=max_tokens,
            logprobs=bool(return_probs_for),
            top_logprobs=(options.get("top_logprobs", 5) if return_probs_for else None),
            **options,
        )

    text_response = response.choices[0].message.content

    if verbose or os.getenv("VERBOSE") == "True":
        print(f"...\nText: {prompt[-1]['content']}\nResponse: {text_response}\n")

    if return_probs_for:
        all_logprobs = response.choices[0].logprobs.content[0].top_logprobs
        all_logprobs_dict = {x.token: x.logprob for x in all_logprobs}
        probs = {token: 0 for token in return_probs_for}
        for token, prob in all_logprobs_dict.items():
            if token in probs:
                probs[token] = math.exp(prob)
        total_prob = sum(probs.values())
        probs_relative = {token: prob / total_prob for token, prob in probs.items()}
        return probs_relative

    return text_response


@cache
def get_llm_response(
    prompt: str | list[dict[str, str]],
    model: str | None = None,
    return_probs_for: list[str] | None = None,
    max_tokens: int | None = None,
    words_in_mouth: str | None = None,
    verbose:bool=False,
    simulate:bool|int=False,
    cost_estimator:CostEstimator=None,
    **kwargs,
) -> str | dict[str, float]:
    """
    Get LLM response to a prompt.

    Args:
        prompt: str | list[dict[str, str]]: Prompt to send to the LLM.
        verbose: bool: Whether to print the prompt and response.
        model: str | None: Model to use for the LLM. Defaults to "gpt-4o-2024-05-13".
        return_probs_for: list[str] | None: List of tokens to return relative probabilities for.
            If None, simply returns the text response.
        max_tokens: int | None: Maximum number of tokens to generate.
        words_in_mouth: str | None: Words to append to the prompt, only used for huggingface models.
            E.g. " Sure, here's my response:\n\n"
        verbose: bool: Whether to print the prompt and response.
        simulate: bool | int: Whether to simulate the response. If int, number of tokens to return
            in output. Defaults to False.
        cost_estimator: CostEstimator: CostEstimator object to use for cost estimation.

    Keyword Args:
        response_model: pydantic.BaseModel: Pydantic model to use for response, if using
            instructor. Defaults to None (in which case instructor is not used).
        top_logprobs: int: Number of top logprobs to return. Defaults to 5.
    """

    print("NOT USING CACHE")

    default_options = {
        "model": "gpt-4o-2024-05-13",
    }
    options = default_options | kwargs
    options["model"] = model or options["model"]

    if isinstance(prompt, str):
        prompt = prepare_messages(prompt)
    if "response_model" in options:
        client, client_name = get_client_pydantic(options["model"], use_async=False)
    else:
        client, client_name = get_client_native(options["model"], use_async=False)
    call_messages = (
        _mistral_message_transform(prompt) if client_name == "mistral" else prompt
    )
    
    if simulate and cost_estimator:
        with cost_estimator.add_api_item(
            model=options["model"],
            input_tokens=len("".join([m["content"] for m in prompt])) // 3,
            output_tokens=[1, (max_tokens or 2048)],
        ):
            if return_probs_for:
                return {token: 0 for token in return_probs_for}
            simstr = "This is a test output."
            return simstr * (simulate // (len(simstr) // 3))

    if client_name == "huggingface_local":
        return get_hf_response(
            prompt=call_messages,
            model_name=options["model"],
            max_tokens=max_tokens,
            return_probs_for=return_probs_for,
            prompt_type="messages",
            words_in_mouth=words_in_mouth,
        )
    elif client_name == "mistral" and not os.getenv("USE_OPENROUTER"):
        response = client.chat(
            messages=call_messages,
            max_tokens=max_tokens,
            logprobs=bool(return_probs_for),
            top_logprobs=(options.get("top_logprobs", 5) if return_probs_for else None),
            **options,
        )
    else:
        response = client.chat.completions.create(
            messages=call_messages,
            max_tokens=max_tokens,
            logprobs=bool(return_probs_for),
            top_logprobs=(5 if return_probs_for else None),
            **options,
        )

    text_response = response.choices[0].message.content

    if verbose or os.getenv("VERBOSE") == "True":
        print(f"...\nText: {prompt[-1]['content']}\nResponse: {text_response}\n")

    if return_probs_for:
        all_logprobs = response.choices[0].logprobs.content[0].top_logprobs
        all_logprobs_dict = {x.token: x.logprob for x in all_logprobs}
        probs = {token: 0 for token in return_probs_for}
        for token, prob in all_logprobs_dict.items():
            if token in probs:
                probs[token] = math.exp(prob)
        total_prob = sum(probs.values())
        probs_relative = {token: prob / total_prob for token, prob in probs.items()}
        return probs_relative

    return text_response
