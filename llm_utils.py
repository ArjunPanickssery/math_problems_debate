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
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from mistralai.async_client import MistralAsyncClient
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from anthropic import AsyncAnthropic, Anthropic
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import transformers
import instructor
from instructor.client import Instructor
from instructor.mode import Mode


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


def get_huggingface_local_client(hf_repo) -> transformers.pipeline:
    hf_model_path = os.path.join(os.getenv("HF_MODELS_DIR"), hf_repo)
    if not os.path.exists(hf_model_path):
        snapshot_download(hf_repo, local_dir=hf_model_path)

    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, legacy=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(hf_model_path)
    pipeline = transformers.pipeline(
        "text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2048
    )
    return pipeline


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


async def get_llm_response_async(
    prompt: str | list[dict[str, str]],
    model: str | None = None,
    return_probs_for: list[str] | None = None,
    max_tokens: int | None = None,
    verbose=False,
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
    
    Keyword Args:
        response_model: pydantic.BaseModel: Pydantic model to use for response, if using
            instructor. Defaults to None (in which case instructor is not used).
        top_logprobs: int: Number of top logprobs to return. Defaults to 5.
    """

    default_options = {
        "model": "gpt-4o-2024-05-13",
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

    print(
        options,
        f"Approx num tokens: {len(''.join([m['content'] for m in prompt])) // 3}",
    )
    if client_name == "mistral" and not os.getenv("USE_OPENROUTER"):
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


def get_llm_response(
    prompt: str | list[dict[str, str]],
    model: str | None = None,
    return_probs_for: list[str] | None = None,
    max_tokens: int | None = None,
    verbose=False,
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
    
    Keyword Args:
        response_model: pydantic.BaseModel: Pydantic model to use for response, if using
            instructor. Defaults to None (in which case instructor is not used).
        top_logprobs: int: Number of top logprobs to return. Defaults to 5.
    """
    
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

    print(
        options,
        f"Approx num tokens: {len(''.join([m['content'] for m in prompt])) // 3}",
    )
    if client_name == "mistral" and not os.getenv("USE_OPENROUTER"):
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
