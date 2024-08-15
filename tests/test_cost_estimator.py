import pytest
from solib.cost_estimator import CostItem, CostEstimator
from solib.llm_utils import get_llm_response, get_llm_response_async

PROMPT = (
    "Explain the Riemann Hypothesis to me in full mathematical "
    "detail, including all necessary prerequisites."
)


def test_estimate_contains_exact():
    ce_sim = CostEstimator()
    ce_real = CostEstimator()
    x = get_llm_response(
        PROMPT,
        model="gpt-4o-mini",
        simulate=True,
        cost_estimation={"cost_estimator": ce_sim},
    )
    y = get_llm_response(
        PROMPT,
        model="gpt-4o-mini",
        simulate=False,
        cost_estimation={"cost_estimator": ce_real},
    )
    assert ce_sim.cost_range[0] <= ce_real.cost_range[0]
    assert ce_sim.cost_range[1] >= ce_real.cost_range[1]
    assert ce_sim.time_range[0] <= ce_real.time_range[0]
    assert ce_sim.time_range[1] >= ce_real.time_range[1]