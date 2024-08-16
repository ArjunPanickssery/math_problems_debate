import pytest
from solib.cost_estimator import CostItem, CostEstimator
from solib.llm_utils import get_llm_response, get_llm_response_async

PROMPT1 = (
    "Explain the Riemann Hypothesis to me in full mathematical "
    "detail, including all necessary prerequisites."
)
PROMPT2 = (
    "Take a random guess as to what the 1,000,001st digit of pi is. "
    'Answer exactly "0", "1", ... or "9", with nothing else in your response.'
)

PARAMS = [
    {"prompt": PROMPT1, "model": "gpt-4o-mini", "return_probs_for": None},
    {
        "prompt": PROMPT2,
        "model": "gpt-4o-mini",
        "return_probs_for": [str(n) for n in range(10)],
    },
    {
        "prompt": PROMPT1,
        "model": "hf:meta-llama/Llama-2-7b-chat-hf",
        "return_probs_for": None,
    },
    {
        "prompt": PROMPT2,
        "model": "hf:meta-llama/Llama-2-7b-chat-hf",
        "return_probs_for": [str(n) for n in range(10)],
    },
]


@pytest.mark.parametrize("params", PARAMS)
def test_estimate_contains_exact(params, request):

    prompt, model, return_probs_for = (
        params["prompt"],
        params["model"],
        params["return_probs_for"],
    )

    if model.startswith("hf:") and not request.config.getoption("--runhf", False):
        pytest.skip("skipping test with hf model, add --runhf option to run")

    ce_sim = CostEstimator()
    ce_real = CostEstimator()
    x = get_llm_response(
        prompt,
        model=model,
        return_probs_for=return_probs_for,
        simulate=True,
        cost_estimation={"cost_estimator": ce_sim},
    )
    y = get_llm_response(
        prompt,
        model=model,
        return_probs_for=return_probs_for,
        simulate=False,
        cost_estimation={"cost_estimator": ce_real},
    )
    assert ce_sim.cost_range[0] <= ce_real.cost_range[0]
    assert ce_sim.cost_range[1] >= ce_real.cost_range[1]
    assert ce_sim.time_range[0] <= ce_real.time_range[0]
    assert ce_sim.time_range[1] >= ce_real.time_range[1]
