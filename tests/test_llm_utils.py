import pytest
import asyncio
import os
from solib.utils.llm_utils import RUNLOCAL, RATE_LIMITER, LLM_Agent
from solib.utils.rate_limits.rate_limits import RATE_LIMITER
from solib.datatypes import Prob
from solib.utils.default_tools import math_eval
from pydantic import BaseModel, Field

models = [
    "openrouter/gpt-4o-mini-2024-07-18",
    "claude-3-5-sonnet-20241022",
    "gemini/gemini-1.5-flash",
    "openrouter/deepseek/deepseek-chat",
    # "localhf://TinyLlama/TinyLlama-1.1B-Chat-v1.0"
]
if RUNLOCAL:
    models.extend(
        [
            "ollama_chat/Llama-2:7b",
            "ollama_chat/llama-3.1:8b",
            "localhf://meta-llama/Meta-Llama-3.1-8B",
        ]
    )


def requires_api_key(model: str) -> bool:
    """Check if we have the required API key for a given model"""
    if "claude" in model.lower():
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    elif "gemini" in model.lower():
        return bool(os.getenv("GOOGLE_API_KEY"))
    elif model.startswith("gpt-"):
        return bool(os.getenv("OPENAI_API_KEY"))
    elif model.startswith("openrouter/"):
        return bool(os.getenv("OPENROUTER_API_KEY"))
    elif "deepseek" in model.lower() and not model.startswith("openrouter/"):
        return bool(os.getenv("DEEPSEEK_API_KEY"))
    elif model.startswith("ollama") or model.startswith("localhf"):
        return RUNLOCAL
    return True  # Default to True for unknown models


@pytest.fixture(params=models)
def model(request):
    model_name = request.param
    if not requires_api_key(model_name):
        pytest.skip(f"Skipping test for {model_name} - required API key not found")
    return model_name


@pytest.fixture
def llm_agent(model):
    return LLM_Agent(model=model)


@pytest.fixture
def llm_agent_with_tool(model):
    return LLM_Agent(model=model, tools=[math_eval])


@pytest.fixture(scope="module")
def event_loop():
    """
    Create an instance of the default event loop for the test session.
    https://github.com/BerriAI/litellm/issues/8142
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.mark.asyncio
async def test_get_response_returns_string(llm_agent):
    prompt = "What is the capital of France?"
    response = await llm_agent.get_response(prompt=prompt)
    assert isinstance(response, str)


@pytest.mark.asyncio
async def test_get_probs_returns_dict(llm_agent):
    if any(model in llm_agent.model for model in ["claude", "gemini", "deepseek"]):
        pytest.skip(f"Model {model} does not support get_probs")
    prompt = (
        "Take a random guess as to what the 1,000,001st digit of pi is. "
        'Answer exactly "0", "1", ... or "9", with nothing else in your response.'
    )
    response = await llm_agent.get_probs(
        prompt=prompt,
        return_probs_for=[str(n) for n in range(10)],
    )
    assert isinstance(response, dict)


@pytest.mark.asyncio
async def test_get_response_with_max_tokens(llm_agent):
    prompt = "What is das capital of France?"
    max_tokens = 50
    response = await llm_agent.get_response(prompt=prompt, max_tokens=max_tokens)
    assert isinstance(response, str)
    assert len(response.split()) <= max_tokens


@pytest.mark.asyncio
async def test_get_response_with_words_in_mouth(llm_agent):
    prompt = (
        "Take a random guess as to what the 1,000,001st digit of pi is. "
        'Answer exactly "0", "1", ... or "9", with nothing else in your response.'
    )
    words_in_mouth = " My guess for the correct answer is:\n\n"
    response = await llm_agent.get_response(
        prompt=prompt, words_in_mouth=words_in_mouth
    )
    assert isinstance(response, str)


@pytest.mark.asyncio
async def test_get_response_with_response_model(llm_agent):
    class ResponseModel(BaseModel):
        capital: str
        country: str

    prompt = "What is the capital of France?"
    response = await llm_agent.get_response(prompt=prompt, response_model=ResponseModel)

    # need to check instance weirdly since cloudpickle hack messes up class namespace a bit
    assert type(response).__name__ == ResponseModel.__name__
    assert response.country == "France"
    assert response.capital == "Paris"


@pytest.mark.asyncio
async def test_get_response_with_response_model_Prob(llm_agent):

    prompt = "What is the probability of the Russia-Ukraine war ending by 2025?"
    response = await llm_agent.get_response(prompt=prompt, response_model=Prob)

    # need to check instance weirdly since cloudpickle hack messes up class namespace a bit
    assert type(response).__name__ == Prob.__name__


@pytest.mark.asyncio
async def test_get_response_with_response_model_more_complicated(llm_agent):
    class ResponseModel(BaseModel):
        students: list[str] = Field(
            description="Each key is an ID, each value is a name."
        )

    ## dicts don't work

    # prompt = "Create an object consisting of an attribute students, which is a dictionary of 5 students with IDs as keys and names as values."
    prompt = "Create a list of 5 example students."
    response = await llm_agent.get_response(prompt=prompt, response_model=ResponseModel)
    assert isinstance(response, ResponseModel)
    assert len(response.students) == 5
    assert all(isinstance(student, str) for student in response.students)



@pytest.mark.asyncio
async def test_invalid_response_model(llm_agent):
    class InvalidModel(BaseModel):
        impossible_field: complex  # A type that can't be parsed from JSON

    prompt = "Say something."
    with pytest.raises(Exception):
        await llm_agent.get_response(prompt=prompt, response_model=InvalidModel)


@pytest.mark.parametrize("max_tokens", [10, 20, 50])
async def test_response_length_with_max_tokens(llm_agent, max_tokens):
    prompt = "Write a story about a dog."
    response = await llm_agent.get_response(prompt=prompt, max_tokens=max_tokens)
    # Rough approximation: tokens ≈ words * 1.3
    words = len(response.split())
    assert words <= max_tokens * 1.3


@pytest.mark.asyncio
async def test_tool_calling(llm_agent_with_tool):
    prompt = "What is 25 * 48?"
    response = await llm_agent_with_tool.get_response(prompt=prompt)
    assert isinstance(response, str)
    assert "1200" in response  # The result should appear in the response
