import pytest
import litellm
import asyncio
import os

models = [
    "gpt-4o-mini",
    "claude-3-5-sonnet-20241022",
    "gemini/gemini-2.0-flash-exp",
]


def requires_api_key(model: str) -> bool:
    """Check if we have the required API key for a given model"""
    if model.startswith("claude"):
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    elif model.startswith("gemini"):
        return bool(os.getenv("GOOGLE_API_KEY"))
    elif model.startswith("gpt-4o"):
        return bool(os.getenv("OPENROUTER_API_KEY"))
    return True  # Default to True for unknown models


@pytest.fixture(params=models)
def model(request):
    model_name = request.param
    if not requires_api_key(model_name):
        pytest.skip(f"Skipping test for {model_name} - required API key not found")
    return model_name


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
