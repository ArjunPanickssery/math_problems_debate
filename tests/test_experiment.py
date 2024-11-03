import pytest  # noqa
from solib.protocols.abstract import QA_Agent, Judge
from solib.protocols.protocols import Debate
from solib.Experiment import Experiment

experiment = Experiment(
    agent_models=["gpt-4o-mini", "hf:meta-llama/Meta-Llama-3-8B-Instruct"],
    agent_toolss=[],
    judge_models=["gpt-4o-mini", "hf:meta-llama/Meta-Llama-3-8B-Instruct"],
)


def test_get_path():
    path = experiment.get_path(
        {
            "protocol": Debate,
            "init_kwargs": {
                "simultaneous": True,
                "num_turns": 4,
            },
            "call_kwargs": {
                "judge": Judge(),
                "agent": QA_Agent(),
                "adversary": QA_Agent(),
            },
        }
    )
    path = str(path)
    assert path.startswith("experiments/results_")
    assert path.endswith(
        "/Debate_t1_n4/J_gpt-4o-mini_A_gpt-4o-mini_A_gpt-4o-mini/results.json"
    )
