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


def get_async_openai_client() -> AsyncOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    return AsyncOpenAI(api_key=api_key)

def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=api_key)

def get_async_openrouter_client() -> AsyncOpenAI:
    print("Calling models through OpenRouter")
    api_key = os.getenv("OPENROUTER_API_KEY")
    return AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

def get_openrouter_client() -> OpenAI:
    print("Calling models through OpenRouter")
    api_key = os.getenv("OPENROUTER_API_KEY")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

def get_mistral_async_client() -> MistralAsyncClient:
    api_key = os.getenv("MISTRAL_API_KEY")
    return MistralAsyncClient(api_key=api_key)

def get_mistral_client() -> MistralClient:
    api_key = os.getenv("MISTRAL_API_KEY")
    return MistralClient(api_key=api_key)

def get_anthropic_async_client() -> AsyncAnthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    return AsyncAnthropic(api_key=api_key)

def get_anthropic_client() -> Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    return Anthropic(api_key=api_key)

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

def get_client(
    model: str, use_async=True
) -> tuple[AsyncOpenAI | OpenAI | MistralAsyncClient | MistralClient, str]:
    provider = get_provider(model)

    if os.getenv("USE_OPENROUTER"):
        client = (
            get_async_openrouter_client()
            if use_async
            else get_openrouter_client()
        )
    elif provider == "openai":
        client = (
            get_async_openai_client()
            if use_async
            else get_openai_client()
        )
    elif provider == "mistral":
        client = (
            get_mistral_async_client()
            if use_async
            else get_mistral_client()
        )
    elif provider == "huggingface_local":
        assert model.startswith("hf:")
        model = model.split("hf:")[1]
        client = get_huggingface_local_client(model)
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


async def query_api_chat(
    messages: list[dict[str, str]],
    verbose=False,
    model: str | None = None,
    **kwargs,
) -> str:
    default_options = {
        "model": "gpt-4o-2024-05-13",
    }
    options = default_options | kwargs
    options["model"] = model or options["model"]

    client, client_name = get_client(options["model"], use_async=True)
    call_messages = (
        _mistral_message_transform(messages) if client_name == "mistral" else messages
    )

    print(
        options,
        f"Approx num tokens: {len(''.join([m['content'] for m in messages])) // 3}",
    )
    if client_name == "mistral" and not os.getenv("USE_OPENROUTER"):
        response = await client.chat(
            messages=call_messages,
            **options,
        )
    else:
        response = await client.chat.completions.create(
            messages=call_messages,
            **options,
        )

    text_response = response.choices[0].message.content

    if verbose or os.getenv("VERBOSE") == "True":
        print(f"...\nText: {messages[-1]['content']}\nResponse: {text_response}\n")

    return text_response

async def query_api_chat(
    messages: list[dict[str, str]],
    verbose=False,
    model: str | None = None,
    **kwargs,
) -> str:
    default_options = {
        "model": "gpt-4o-2024-05-13",
    }
    options = default_options | kwargs
    options["model"] = model or options["model"]

    client, client_name = get_client(options["model"], use_async=True)
    call_messages = (
        _mistral_message_transform(messages) if client_name == "mistral" else messages
    )

    print(
        options,
        f"Approx num tokens: {len(''.join([m['content'] for m in messages])) // 3}",
    )
    if client_name == "mistral" and not os.getenv("USE_OPENROUTER"):
        response = await client.chat(
            messages=call_messages,
            **options,
        )
    else:
        response = await client.chat.completions.create(
            messages=call_messages,
            **options,
        )

    text_response = response.choices[0].message.content

    if verbose or os.getenv("VERBOSE") == "True":
        print(f"...\nText: {messages[-1]['content']}\nResponse: {text_response}\n")

    return text_response
