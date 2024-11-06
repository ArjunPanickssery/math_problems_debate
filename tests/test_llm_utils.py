import pytest
from solib.llm_utils import LLM_Agent
from pydantic import BaseModel

pytest_plugins = ("anyio",)


@pytest.fixture(
    params=[
        "gpt-4o-mini",
        "hf:meta-llama/Llama-2-7b-chat-hf",
        "hf:meta-llama/Meta-Llama-3-8B-Instruct",
    ]
)
def model(request):
    return request.param


@pytest.fixture
def llm_agent(model):
    return LLM_Agent(model=model)


def test_get_response_returns_string(llm_agent, request):
    if llm_agent.model.startswith("hf:") and not request.config.getoption(
        "--runhf", False
    ):
        pytest.skip("slow test, add --runhf option to run")
    prompt = "What is the capital of France?"
    response = llm_agent.get_response_sync(prompt=prompt)
    assert isinstance(response, str)


def test_get_probs_returns_dict(llm_agent, request):
    if llm_agent.model.startswith("hf:") and not request.config.getoption(
        "--runhf", False
    ):
        pytest.skip("skipping test with hf model, add --runhf option to run")
    prompt = (
        "Take a random guess as to what the 1,000,001st digit of pi is. "
        'Answer exactly "0", "1", ... or "9", with nothing else in your response.'
    )
    response = llm_agent.get_probs_sync(
        prompt=prompt,
        return_probs_for=[str(n) for n in range(10)],
    )
    assert isinstance(response, dict)


def test_get_response_with_max_tokens(llm_agent, request):
    if llm_agent.model.startswith("hf:") and not request.config.getoption(
        "--runhf", False
    ):
        pytest.skip("slow test, add --runhf option to run")
    prompt = "What is the capital of France?"
    max_tokens = 50
    response = llm_agent.get_response_sync(prompt=prompt, max_tokens=max_tokens)
    assert isinstance(response, str)
    assert len(response.split()) <= max_tokens


def test_get_response_with_words_in_mouth(llm_agent, request):
    if llm_agent.model.startswith("hf:") and not request.config.getoption(
        "--runhf", False
    ):
        pytest.skip("slow test, add --runhf option to run")
    prompt = (
        "Take a random guess as to what the 1,000,001st digit of pi is. "
        'Answer exactly "0", "1", ... or "9", with nothing else in your response.'
    )
    words_in_mouth = " My guess for the correct answer is:\n\n"
    response = llm_agent.get_response_sync(prompt=prompt, words_in_mouth=words_in_mouth)
    assert isinstance(response, str)


@pytest.mark.asyncio
async def test_get_response_async_returns_string(llm_agent, request):
    if llm_agent.model.startswith("hf:"):
        pytest.skip("hf models don't support async")
    prompt = "What is the capital of France?"
    response = await llm_agent.get_response_async(prompt=prompt)
    assert isinstance(response, str)


@pytest.mark.asyncio
async def test_get_probs_async_returns_dict(llm_agent, request):
    if llm_agent.model.startswith("hf:"):
        pytest.skip("hf models don't support async")
    prompt = (
        "Take a random guess as to what the 1,000,001st digit of pi is. "
        'Answer exactly "0", "1", ... or "9", with nothing else in your response.'
    )
    response = await llm_agent.get_probs_async(
        prompt=prompt,
        return_probs_for=[str(n) for n in range(10)],
    )
    assert isinstance(response, dict)


@pytest.mark.asyncio
async def test_get_response_async_with_max_tokens(llm_agent, request):
    if llm_agent.model.startswith("hf:"):
        pytest.skip("hf models don't support async")
    prompt = "What is the capital of France?"
    max_tokens = 50
    response = await llm_agent.get_response_async(prompt=prompt, max_tokens=max_tokens)
    assert isinstance(response, str)
    assert len(response.split()) <= max_tokens


@pytest.mark.asyncio
async def test_get_response_async_with_words_in_mouth(llm_agent, request):
    if llm_agent.model.startswith("hf:"):
        pytest.skip("hf models don't support async")
    prompt = (
        "Take a random guess as to what the 1,000,001st digit of pi is. "
        'Answer exactly "0", "1", ... or "9", with nothing else in your response.'
    )
    words_in_mouth = " My guess for the correct answer is:\n\n"
    response = await llm_agent.get_response_async(
        prompt=prompt, words_in_mouth=words_in_mouth
    )
    assert isinstance(response, str)


def test_get_response_with_response_model(llm_agent, request):
    if llm_agent.model.startswith("hf:") and not request.config.getoption(
        "--runhf", False
    ):
        pytest.skip("slow test, add --runhf option to run")

    class ResponseModel(BaseModel):
        capital: str
        country: str

    prompt = "What is the capital of France?"
    response = llm_agent.get_response_sync(prompt=prompt, response_model=ResponseModel)

    # need to check instance weirdly since cloudpickle hack messes up class namespace a bit
    assert type(response).__name__ == ResponseModel.__name__
    assert response.country == "France"
    assert response.capital == "Paris"


@pytest.mark.asyncio
async def test_get_response_async_with_response_model(llm_agent, request):
    if llm_agent.model.startswith("hf:"):
        pytest.skip("hf models don't support async")

    class ResponseModel(BaseModel):
        capital: str
        country: str

    prompt = "What is the capital of France?"

    response = await llm_agent.get_response_async(
        prompt=prompt, response_model=ResponseModel
    )

    # need to check instance weirdly since cloudpickle hack messes up class namespace a bit
    assert type(response).__name__ == ResponseModel.__name__
    assert response.country == "France"
    assert response.capital == "Paris"
