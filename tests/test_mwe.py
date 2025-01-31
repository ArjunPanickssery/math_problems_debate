import pytest
import litellm
from litellm import completion, acompletion
import asyncio

MODEL = "claude-3-5-sonnet-20241022"

@pytest.fixture(scope="module")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.mark.asyncio
async def test_1():
    prompt = "What is the capital of France?"
    response = await litellm.acompletion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    response_text = response.choices[0].message.content
    assert isinstance(response_text, str)

@pytest.mark.asyncio
async def test_2():
    prompt = "What is the capital of France?"
    response = await litellm.acompletion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    response_text = response.choices[0].message.content
    assert isinstance(response_text, str)