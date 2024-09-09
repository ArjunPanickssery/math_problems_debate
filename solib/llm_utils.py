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
import asyncio
import warnings
import inspect
import math
from typing import Literal, Union, Coroutine, TYPE_CHECKING
from perscache import Cache
from perscache.serializers import JSONSerializer
from costly import Costlog, CostlyResponse, costly

if TYPE_CHECKING:
    from pydantic import BaseModel
    from transformers import AutoTokenizer
    from openai import AsyncOpenAI, OpenAI
    from anthropic import AsyncAnthropic, Anthropic
    from mistralai.async_client import MistralAsyncClient, MistralClient
    from mistralai.client import MistralClient
    from instructor import Instructor

cache = Cache(serializer=JSONSerializer())

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
    messages: list[dict[str, str]] = None,
    input_string: str = None,
    tokenizer: Union["AutoTokenizer", None] = None,
    system_message: str | None = None,
    words_in_mouth: str | None = None,
) -> dict[Literal["messages", "input_string"], str | list[dict[str, str]]]:
    """
    Three types of prompts:
        prompt: 'What is the capital of the moon?'
        messages: a list[dict[str, str]] in the Chat Message format
        input_string: a string that the LLM can directly <im_start> etc.
    
    This converts prompt -> messages -> input_string. Returns both messages
    and input_string. If both prompt and messages are provided, prompt is
    ignored. You cannot convert anything in the reverse order.

    Args:
        prompt: str: Prompt to convert. Either prompt or messages must be provided
            to calculate input_string.
        messages: list[dict[str, str]]: Messages to convert. Either prompt or 
            messages must be provided to calculate input_string.
        tokenizer: AutoTokenizer: Tokenizer to use for the conversion.
        system_message: str | None: System message to add to the messages. Will be
            ignored if messages is provided.
        words_in_mouth: str | None: Words to append to the prompt. Will be ignored
            if input_string is provided.
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
        if tokenizer is not None:
            input_string = tokenizer.apply_chat_template(messages, tokenize=False)
        if words_in_mouth is not None:
            input_string += words_in_mouth
    return {"messages": messages, "input_string": input_string}

def get_llm(
    model: str,
    use_async=False,
    use_instructor: bool = False,
):
    if model.startswith("hf:"):  # Hugging Face local models
        from transformers import AutoTokenizer, AutoModelForCausalLM

        api_key = os.getenv("HF_TOKEN")
        model = model.split("hf:")[1]
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForCausalLM.from_pretrained(
            model, device_map="auto", token=api_key
        )
        client = (tokenizer, model)

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
            )
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            output = model.generate(input_ids, max_length=max_tokens)
            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            prompt_end = prompt[-9:]  # HACK
            response = decoded.split(prompt_end)[-1]
            return response

        def return_probs(
            return_probs_for: list[str],
            prompt: str = None,
            messages: list[dict[str, str]] = None,
            input_string: str = None,
            system_message: str | None = None,
            words_in_mouth: str | None = None,
            **kwargs,
        ):
            input_string = format_prompt(
                prompt=prompt,
                messages=messages,
                input_string=input_string,
                tokenizer=tokenizer,
                system_message=system_message,
                words_in_mouth=words_in_mouth,
            )
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
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
        if model.startswith("or:"):  # OpenRouter
            import openai

            model_or = model.split("or:")[1]
            api_key = os.getenv("OPENROUTER_API_KEY")
            if use_async:
                client = openai.AsyncOpenAI(
                    base_url="https://openrouter.ai/api/v1", api_key=api_key
                )
            else:
                client = openai.OpenAI(
                    base_url="https://openrouter.ai/api/v1", api_key=api_key
                )
            get_response = client.chat.completions.create
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
                import openai

                api_key = os.getenv("OPENAI_API_KEY")
                if use_async:
                    client = openai.AsyncOpenAI(api_key=api_key)
                else:
                    client = openai.OpenAI(api_key=api_key)
                get_response = client.chat.completions.create
            elif model.startswith(("claude", "anthropic")):
                import anthropic

                api_key = os.getenv("ANTHROPIC_API_KEY")
                if use_async:
                    client = anthropic.AsyncAnthropic(api_key=api_key)
                else:
                    client = anthropic.Anthropic(api_key=api_key)
                get_response = client.messages.create
            elif model.startswith("mistral"):
                if use_async:
                    from mistralai.async_client import MistralAsyncClient

                    api_key = os.getenv("MISTRAL_API_KEY")
                    client = MistralAsyncClient(api_key=api_key)
                else:
                    from mistralai.client import MistralClient

                    api_key = os.getenv("MISTRAL_API_KEY")
                    client = MistralClient(api_key=api_key)
                get_response = client.chat
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
                    mode = Mode.TOOLS_STRICT # use OpenAI structured outputs
                    client = instructor.from_openai(client, mode=mode)
                elif model.startswith(("anthropic", "claude")):
                    mode = Mode.ANTHROPIC_JSON
                    client = instructor.from_anthropic(client, mode=mode)
                        
        if use_instructor:
            get_response = client.chat.completions.create_with_completion
            def process_response(response):
                raw_response, completion = response
                usage = completion.usage
                return raw_response, usage
        else:
            # create_fn already set above
            def process_response(response):
                raw_response = response.choices[0].message.content
                usage = response.usage
                return raw_response, usage
        def generate(
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
            )["messages"]
            
            if model.startswith("mistral"):
                from mistralai.client import ChatMessage
                messages = [ChatMessage(role=x['role'], content=x['content']) for x in messages]
                
            allowed_params = inspect.signature(get_response).parameters.keys()
            response_params = {
                'messages': messages,
                'model': model,
                'max_tokens': max_tokens,
                'response_model': response_model,
                'temperature': temperature,
            }
            filtered_params = {k: v for k, v in response_params.items() if k in allowed_params}
            
            response = get_response(**filtered_params)
            raw_response, usage = process_response(response)
            return raw_response, usage

        async def generate_async(
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
            )["messages"]

            if model.startswith("mistral"):
                from mistralai.client import ChatMessage
                messages = [ChatMessage(role=x['role'], content=x['content']) for x in messages]

            allowed_params = inspect.signature(get_response).parameters.keys()
            response_params = {
                'messages': messages,
                'model': model,
                'max_tokens': max_tokens,
                'response_model': response_model,
                'temperature': temperature,
            }
            filtered_params = {k: v for k, v in response_params.items() if k in allowed_params}
            
            response = await get_response(**filtered_params)
            raw_response, usage = process_response(response)
            return raw_response, usage
        
        def return_probs(
            return_probs_for: list[str],
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
            )["messages"]

            if model.startswith("mistral"):
                from mistralai.client import ChatMessage
                messages = [ChatMessage(role=x['role'], content=x['content']) for x in messages]

            allowed_params = inspect.signature(get_response).parameters.keys()
            response_params = {
                'messages': messages,
                'model': model,
                'max_tokens': 1,
                'logprobs': True,
                'top_logprobs': top_logprobs,
                'temperature': temperature,
            }
            filtered_params = {k: v for k, v in response_params.items() if k in allowed_params}
            response = get_response(**filtered_params)
            raw_response, usage = process_response(response)
            all_logprobs = response.choices[0].logprobs.content[0].top_logprobs
            all_logprobs_dict = {x.token: x.logprob for x in all_logprobs}
            probs = {token: 0 for token in return_probs_for}
            for token, prob in all_logprobs_dict.items():
                if token in probs:
                    probs[token] = math.exp(prob)
            total_prob = sum(probs.values())
            probs_relative = {token: prob / total_prob for token, prob in probs.items()}
            return probs_relative, usage
        
        async def return_probs_async(
            return_probs_for: list[str],
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
            )["messages"]

            if model.startswith("mistral"):
                from mistralai.client import ChatMessage
                messages = [ChatMessage(role=x['role'], content=x['content']) for x in messages]

            allowed_params = inspect.signature(get_response).parameters.keys()
            response_params = {
                'messages': messages,
                'model': model,
                'max_tokens': 1,
                'logprobs': True,
                'top_logprobs': top_logprobs,
                'temperature': temperature,
            }
            filtered_params = {k: v for k, v in response_params.items() if k in allowed_params}
            response = await get_response(**filtered_params)
            raw_response, usage = process_response(response)
            all_logprobs = response.choices[0].logprobs.content[0].top_logprobs
            all_logprobs_dict = {x.token: x.logprob for x in all_logprobs}
            probs = {token: 0 for token in return_probs_for}
            for token, prob in all_logprobs_dict.items():
                if token in probs:
                    probs[token] = math.exp(prob)
            total_prob = sum(probs.values())
            probs_relative = {token: prob / total_prob for token, prob in probs.items()}
            return probs_relative, usage
        
    return {
        "client": client,
        "generate": generate,
        "generate_async": generate_async,
        "return_probs": return_probs,
        "return_probs_async": return_probs_async,
    }

def get_llm_response(
    model: str,
    response_model: Union["BaseModel", None] = None,
    prompt: str = None,
    messages: list[dict[str, str]] = None,
    input_string: str = None,
    system_message: str | None = None,
    words_in_mouth: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
):
    ai = get_llm(
        model=model,
        use_async=False,
        use_instructor=(response_model is not None),
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
    )

async def get_llm_response_async(
    model: str,
    response_model: Union["BaseModel", None] = None,
    prompt: str = None,
    messages: list[dict[str, str]] = None,
    input_string: str = None,
    system_message: str | None = None,
    words_in_mouth: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
):
    ai = get_llm(
        model=model,
        use_async=True,
        use_instructor=(response_model is not None),
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
    )
        
def get_llm_probs(
    model: str,
    return_probs_for: list[str],
    prompt: str = None,
    messages: list[dict[str, str]] = None,
    input_string: str = None,
    system_message: str | None = None,
    words_in_mouth: str | None = None,
    top_logprobs: int = 5,
    temperature: float = 0.0,
):
    ai = get_llm(
        model=model,
        use_async=False,
        use_instructor=False,
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
    )

async def get_llm_probs_async(
    model: str,
    return_probs_for: list[str],
    prompt: str = None,
    messages: list[dict[str, str]] = None,
    input_string: str = None,
    system_message: str | None = None,
    words_in_mouth: str | None = None,
    top_logprobs: int = 5,
    temperature: float = 0.0,
):
    ai = get_llm(
        model=model,
        use_async=True,
        use_instructor=False,
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
    )