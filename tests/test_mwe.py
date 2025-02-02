import pytest
import litellm
from litellm import completion, acompletion
import asyncio

models = [
    "gpt-4o-mini",
    "claude-3-5-sonnet-20241022",
    "gemini/gemini-2.0-flash-exp",
]

@pytest.fixture(params=models)
def model(request):
    return request.param

@pytest.fixture(scope="module")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.mark.asyncio
async def test_1(model):
    prompt = "What is the capital of France?"
    response = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    response_text = response.choices[0].message.content
    assert isinstance(response_text, str)

@pytest.mark.asyncio
async def test_2(model):
    prompt = "What is the capital of France?"
    response = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    response_text = response.choices[0].message.content
    assert isinstance(response_text, str)