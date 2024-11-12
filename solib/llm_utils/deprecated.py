from typing import Union
from solib.globals import GLOBAL_COST_LOG, #LOGGER, SIMULATE
from solib.llm_utils.caching import cache
from solib.llm_utils.llm_utils import RateLimiter, get_llm


from costly import Costlog
from pydantic import BaseModel

from solib.utils import aretry, retry


@cache(ignore="cost_log")
@retry(attempts=5)
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
    cache_breaker: int = 0,
    cost_log: Costlog = GLOBAL_COST_LOG,
    simulate: bool = SIMULATE,
    **kwargs,  # kwargs necessary for costly
):
    """NOTE: you should generally use the LLM_Agent class instead of this function.
    This is deprecated, or maybe we can just use it for one-time calls etc.
    """
    if not simulate:
        #LOGGER.info(f"Running get_llm_response for {model}; NOT FROM CACHE")
        # Get rate limiter for this model if it exists
        rate_limiter = RateLimiter.get_rate_limiter(model)
        rate_limiter.wait_if_needed()
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
@aretry(attempts=5)
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
    cache_breaker: int = 0,
    cost_log: Costlog = GLOBAL_COST_LOG,
    simulate: bool = SIMULATE,
    **kwargs,
):
    """NOTE: you should generally use the LLM_Agent class instead of this function.
    This is deprecated, or maybe we can just use it for one-time calls etc.
    """
    if not simulate:
        #LOGGER.info(f"Running get_llm_response_async for {model}; NOT FROM CACHE")
        # Get rate limiter for this model if it exists
        rate_limiter = RateLimiter.get_rate_limiter(model)
        rate_limiter.wait_if_needed()
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
@retry(attempts=5)
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
    cache_breaker: int = 0,
    cost_log: Costlog = GLOBAL_COST_LOG,
    simulate: bool = SIMULATE,
    **kwargs,
):
    """NOTE: you should generally use the LLM_Agent class instead of this function.
    This is deprecated, or maybe we can just use it for one-time calls etc.
    """
    if not simulate:
        #LOGGER.info(f"Running get_llm_probs for {model}; NOT FROM CACHE")
        # Get rate limiter for this model if it exists
        rate_limiter = RateLimiter.get_rate_limiter(model)
        rate_limiter.wait_if_needed()
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
@aretry(attempts=5)
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
    cache_breaker: int = 0,
    cost_log: Costlog = GLOBAL_COST_LOG,
    simulate: bool = SIMULATE,
    **kwargs,
):
    """NOTE: you should generally use the LLM_Agent class instead of this function.
    This is deprecated, or maybe we can just use it for one-time calls etc.
    """
    if not simulate:
        #LOGGER.info(f"Running get_llm_probs_async for {model}; NOT FROM CACHE")
        # Get rate limiter for this model if it exists
        rate_limiter = RateLimiter.get_rate_limiter(model)
        rate_limiter.wait_if_needed()
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