import pytest
import asyncio
import time
from solib.utils.llm_utils import (
    RUNHF,
    RATE_LIMITERS,
    format_prompt,
    should_use_words_in_mouth,
    supports_async,
    render_tool_call,
    render_tool_call_result,
)
from solib.utils import LLM_Agent
from pydantic import BaseModel, Field


# Define the models based on RUNHF
models = [
    "gpt-4o-mini",
    "claude-3-5-sonnet-20241022",
    "gemini/gemini-2.0-flash-exp",
]  # "deepseek/deepseek-chat", "deepseek/deepseek-reasoner" # not working
if RUNHF:
    models.extend(
        [
            "hf:meta-llama/Llama-2-7b-chat-hf",
            "hf:meta-llama/Meta-Llama-3-8B-Instruct",
        ]
    )


@pytest.fixture(params=models)
def model(request):
    return request.param


@pytest.fixture
def llm_agent_sync(model):
    return LLM_Agent(model=model, sync_mode=True)


@pytest.fixture
def llm_agent_async(model):
    return LLM_Agent(model=model, sync_mode=False)


@pytest.fixture(scope="module")
def event_loop():
    """
    Create an instance of the default event loop for the test session.
    https://github.com/BerriAI/litellm/issues/8142
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def test_get_response_returns_string(llm_agent_sync):
    prompt = "What is the capital of France?"
    response = llm_agent_sync.get_response_sync(prompt=prompt)
    assert isinstance(response, str)


def test_get_probs_returns_dict(llm_agent_sync):
    if "claude" in llm_agent_sync.model or "gemini" in llm_agent_sync.model:
        pytest.skip("Claude models do not support get_probs")
    prompt = (
        "Take a random guess as to what the 1,000,001st digit of pi is. "
        'Answer exactly "0", "1", ... or "9", with nothing else in your response.'
    )
    response = llm_agent_sync.get_probs_sync(
        prompt=prompt,
        return_probs_for=[str(n) for n in range(10)],
    )
    assert isinstance(response, dict)


def test_get_response_with_max_tokens(llm_agent_sync):
    prompt = "What is the capital of France?"
    max_tokens = 50
    response = llm_agent_sync.get_response_sync(prompt=prompt, max_tokens=max_tokens)
    assert isinstance(response, str)
    assert len(response.split()) <= max_tokens


def test_get_response_with_words_in_mouth(llm_agent_sync):
    prompt = (
        "Take a random guess as to what the 1,000,001st digit of pi is. "
        'Answer exactly "0", "1", ... or "9", with nothing else in your response.'
    )
    words_in_mouth = " My guess for the correct answer is:\n\n"
    response = llm_agent_sync.get_response_sync(
        prompt=prompt, words_in_mouth=words_in_mouth
    )
    assert isinstance(response, str)


@pytest.mark.asyncio
async def test_get_response_async_returns_string(llm_agent_async):
    prompt = "What is the capital of France?"
    response = await llm_agent_async.get_response_async(prompt=prompt)
    assert isinstance(response, str)


@pytest.mark.asyncio
async def test_get_probs_async_returns_dict(llm_agent_async):
    if "claude" in llm_agent_async.model or "gemini" in llm_agent_async.model:
        pytest.skip("Claude models do not support get_probs")
    prompt = (
        "Take a random guess as to what the 1,000,001st digit of pi is. "
        'Answer exactly "0", "1", ... or "9", with nothing else in your response.'
    )
    response = await llm_agent_async.get_probs_async(
        prompt=prompt,
        return_probs_for=[str(n) for n in range(10)],
    )
    assert isinstance(response, dict)


@pytest.mark.asyncio
async def test_get_response_async_with_max_tokens(llm_agent_async):
    prompt = "What is the capital of France?"
    max_tokens = 50
    response = await llm_agent_async.get_response_async(
        prompt=prompt, max_tokens=max_tokens
    )
    assert isinstance(response, str)
    assert len(response.split()) <= max_tokens


@pytest.mark.asyncio
async def test_get_response_async_with_words_in_mouth(llm_agent_async):
    prompt = (
        "Take a random guess as to what the 1,000,001st digit of pi is. "
        'Answer exactly "0", "1", ... or "9", with nothing else in your response.'
    )
    words_in_mouth = " My guess for the correct answer is:\n\n"
    response = await llm_agent_async.get_response_async(
        prompt=prompt, words_in_mouth=words_in_mouth
    )
    assert isinstance(response, str)


def test_get_response_with_response_model(llm_agent_sync):
    class ResponseModel(BaseModel):
        capital: str
        country: str

    prompt = "What is the capital of France?"
    response = llm_agent_sync.get_response_sync(
        prompt=prompt, response_model=ResponseModel
    )

    # need to check instance weirdly since cloudpickle hack messes up class namespace a bit
    assert type(response).__name__ == ResponseModel.__name__
    assert response.country == "France"
    assert response.capital == "Paris"


@pytest.mark.asyncio
async def test_get_response_async_with_response_model(llm_agent_async):
    class ResponseModel(BaseModel):
        capital: str
        country: str

    prompt = "What is the capital of France?"

    response = await llm_agent_async.get_response_async(
        prompt=prompt, response_model=ResponseModel
    )

    # need to check instance weirdly since cloudpickle hack messes up class namespace a bit
    assert type(response).__name__ == ResponseModel.__name__
    assert response.country == "France"
    assert response.capital == "Paris"


def test_get_response_with_response_model_more_complicated(llm_agent_sync):
    class ResponseModel(BaseModel):
        students: list[str] = Field(
            description="Each key is an ID, each value is a name."
        )

    ## dicts don't work

    # prompt = "Create an object consisting of an attribute students, which is a dictionary of 5 students with IDs as keys and names as values."
    prompt = "Create a list of 5 example students."
    response = llm_agent_sync.get_response_sync(
        prompt=prompt, response_model=ResponseModel
    )
    assert isinstance(response, ResponseModel)
    assert len(response.students) == 5
    assert all(isinstance(student, str) for student in response.students)


async def test_get_response_async_with_response_model_more_complicated(llm_agent_async):
    class ResponseModel(BaseModel):
        students: list[str] = Field(
            description="Each key is an ID, each value is a name."
        )

    ## dicts don't work

    # prompt = "Create an object consisting of an attribute students, which is a dictionary of 5 students with IDs as keys and names as values."
    prompt = "Create a list of 5 example students."
    response = await llm_agent_async.get_response_async(
        prompt=prompt, response_model=ResponseModel
    )
    assert isinstance(response, ResponseModel)
    assert len(response.students) == 5
    assert all(isinstance(student, str) for student in response.students)

@pytest.mark.asyncio
async def test_rapid_requests_respect_rate_limits(llm_agent_async):
    prompts = ["What is 1+1?", "What is 2+2?", "What is 3+3?"]
    start_time = asyncio.get_event_loop().time()
    
    responses = await asyncio.gather(*[
        llm_agent_async.get_response_async(prompt=prompt)
        for prompt in prompts
    ])
    
    end_time = asyncio.get_event_loop().time()
    elapsed = end_time - start_time
    
    assert all(isinstance(r, str) for r in responses)
    if "claude" in llm_agent_async.model:
        # Check if requests were properly rate-limited for Claude
        assert elapsed >= len(prompts) * (60 / 4000)  # Based on 4000 RPM limit

@pytest.mark.asyncio
async def test_invalid_response_model(llm_agent_async):
    class InvalidModel(BaseModel):
        impossible_field: complex  # A type that can't be parsed from JSON
        
    prompt = "Say something."
    with pytest.raises(Exception):
        await llm_agent_async.get_response_async(prompt=prompt, response_model=InvalidModel)

@pytest.mark.parametrize("max_tokens", [10, 20, 50])
def test_response_length_with_max_tokens(llm_agent_sync, max_tokens):
    prompt = "Write a story about a dog."
    response = llm_agent_sync.get_response_sync(prompt=prompt, max_tokens=max_tokens)
    # Rough approximation: tokens ≈ words * 1.3
    words = len(response.split())
    assert words <= max_tokens * 1.3

def test_tool_calling(llm_agent_sync):
    def calculator(expression: str) -> str:
        """Calculate the result of a mathematical expression"""
        try:
            return str(eval(expression))
        except:
            return "Error in calculation"
    
    calculator.json = {
        "function": {
            "name": "calculator",
            "description": "Calculate the result of a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
    }
    
    agent = LLM_Agent(model=llm_agent_sync.model, tools=[calculator], sync_mode=True)
    prompt = "What is 25 * 48?"
    response = agent.get_response_sync(prompt=prompt)
    assert isinstance(response, str)
    assert "1200" in response  # The result should appear in the response