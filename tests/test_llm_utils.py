# import sys
# sys.path.append("src")

import pytest
from llm_utils import get_llm_response, get_llm_response_async
import asyncio

@pytest.fixture(
    params=[
        "gpt-4o-mini",
        "hf:meta-llama/Llama-2-7b-chat-hf",
        # "hf:meta-llama/Meta-Llama-3-8B-Instruct",
    ]
)
def model(request):
    return request.param


def test_get_llm_response_returns_string(model, request):
    if model.startswith("hf:") and not request.config.getoption("--runslow", False):
        pytest.skip("slow test, add --runslow option to run")
    prompt = "What is the capital of France?"
    response = get_llm_response(prompt, model=model)
    assert isinstance(response, str)


def test_get_llm_response_returns_dict(model, request):
    if model.startswith("hf:") and not request.config.getoption("--runslow", False):
        pytest.skip("skipping test with hf model, add --runslow option to run")
    prompt = (
        "Take a random guess as to what the 1,000,001st digit of pi is. "
        'Answer exactly "0", "1", ... or "9", with nothing else in your response.'
    )
    response = get_llm_response(
        prompt,
        model=model,
        return_probs_for=[str(n) for n in range(10)],
    )
    assert isinstance(response, dict)


def test_get_llm_response_with_max_tokens(model, request):
    if model.startswith("hf:") and not request.config.getoption("--runslow", False):
        pytest.skip("slow test, add --runslow option to run")
    prompt = "What is the capital of France?"
    max_tokens = 50
    response = get_llm_response(prompt, model=model, max_tokens=max_tokens)
    assert isinstance(response, str)
    assert len(response.split()) <= max_tokens


def test_get_llm_response_with_words_in_mouth(model, request):
    if model.startswith("hf:") and not request.config.getoption("--runslow", False):
        pytest.skip("slow test, add --runslow option to run")
    prompt = (
        "Take a random guess as to what the 1,000,001st digit of pi is. "
        'Answer exactly "0", "1", ... or "9", with nothing else in your response.'
    )
    words_in_mouth = " My guess for the correct answer is:\n\n"
    response = get_llm_response(prompt, model=model, words_in_mouth=words_in_mouth)
    assert isinstance(response, str)


@pytest.mark.asyncio
async def test_get_llm_response_async_returns_string(model, request):
    if model.startswith("hf:") and not request.config.getoption("--runslow", False):
        pytest.skip("slow test, add --runslow option to run")
    prompt = "What is the capital of France?"
    response = await get_llm_response_async(prompt, model=model)
    assert isinstance(response, str)


@pytest.mark.asyncio
async def test_get_llm_response_async_returns_dict(model, request):
    if model.startswith("hf:") and not request.config.getoption("--runslow", False):
        pytest.skip("slow test, add --runslow option to run")
    prompt = (
        "Take a random guess as to what the 1,000,001st digit of pi is. "
        'Answer exactly "0", "1", ... or "9", with nothing else in your response.'
    )
    response = await get_llm_response_async(
        prompt,
        model=model,
        return_probs_for=[str(n) for n in range(10)],
    )
    assert isinstance(response, dict)


@pytest.mark.asyncio
async def test_get_llm_response_async_with_max_tokens(model, request):
    if model.startswith("hf:") and not request.config.getoption("--runslow", False):
        pytest.skip("slow test, add --runslow option to run")
    prompt = "What is the capital of France?"
    max_tokens = 50
    response = await get_llm_response_async(prompt, model=model, max_tokens=max_tokens)
    assert isinstance(response, str)
    assert len(response.split()) <= max_tokens


@pytest.mark.asyncio
async def test_get_llm_response_async_with_words_in_mouth(model, request):
    if model.startswith("hf:") and not request.config.getoption("--runslow", False):
        pytest.skip("slow test, add --runslow option to run")
    prompt = (
        "Take a random guess as to what the 1,000,001st digit of pi is. "
        'Answer exactly "0", "1", ... or "9", with nothing else in your response.'
    )
    words_in_mouth = " My guess for the correct answer is:\n\n"
    response = await get_llm_response_async(
        prompt, model=model, words_in_mouth=words_in_mouth
    )
    assert isinstance(response, str)
