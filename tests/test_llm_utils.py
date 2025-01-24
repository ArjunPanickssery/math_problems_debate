import pytest
from solib.globals import RUNHF  # Ensure RUNHF is imported
from solib.llm_utils import LLM_Agent
from pydantic import BaseModel, Field

pytest_plugins = ("anyio",)

# Define the models based on RUNHF
models = ["gpt-4o-mini", "claude-3-5-sonnet-20241022", "gemini-2.0-flash-exp", "deepseek/deepseek-chat", "deepseek/deepseek-reasoner"]
if RUNHF:
    models.extend([
        "hf:meta-llama/Llama-2-7b-chat-hf",
        "hf:meta-llama/Meta-Llama-3-8B-Instruct",
    ])

@pytest.fixture(params=models)
def model(request):
    return request.param

@pytest.fixture
def llm_agent_sync(model):
    return LLM_Agent(model=model, sync_mode=True)

@pytest.fixture
def llm_agent_async(model):
    return LLM_Agent(model=model, sync_mode=False)

def test_get_response_returns_string(llm_agent_sync):
    prompt = "What is the capital of France?"
    response = llm_agent_sync.get_response_sync(prompt=prompt)
    assert isinstance(response, str)

def test_get_probs_returns_dict(llm_agent_sync):
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
        students: dict[int, str] = Field(description="Each key is an ID, each value is a name.")
    
    prompt = "Create a dictionary of 5 students with IDs as keys and names as values."
    response = llm_agent_sync.get_response_sync(prompt=prompt, response_model=ResponseModel)
    assert isinstance(response, ResponseModel)
    assert len(response.students) == 5
    assert all(isinstance(key, int) and isinstance(value, str) for key, value in response.students.items())

async def test_get_response_async_with_response_model_more_complicated(llm_agent_async):
    class ResponseModel(BaseModel):
        students: dict[int, str] = Field(description="Each key is an ID, each value is a name.")
    
    prompt = "Create a dictionary of 5 students with IDs as keys and names as values."
    response = await llm_agent_async.get_response_async(prompt=prompt, response_model=ResponseModel)
    assert isinstance(response, ResponseModel)
    assert len(response.students) == 5
    assert all(isinstance(key, int) and isinstance(value, str) for key, value in response.students.items())
