import asyncio  # noqa
import pytest  # noqa
from pathlib import Path
from datetime import datetime
from solib.protocols.protocols import Debate, Blind, Propaganda, Consultancy  # noqa
from solib.Experiment import Experiment
from solib.data.loading import GSM8K
from solib.tool_use.default_tools import math_eval
from solib.protocols.abstract import QA_Agent, Judge

experiment = Experiment(
    questions=[],
    agent_models=[
        "gpt-4o-mini","hf:meta-llama/Meta-Llama-3-8B-Instruct"],
    agent_toolss=[[]],
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
                "judge": Judge("gpt-4o-mini"),
                "agent": QA_Agent("gpt-4o-mini"),
                "adversary": QA_Agent("gpt-4o-mini"),
            },
        }
    )
    path = str(path)
    assert path.startswith("experiments/results_")
    assert path.endswith("/Debate_t1_n4/_Jgpt-4o-mini_Agpt-4o-mini_Agpt-4o-mini")


questions = GSM8K.data(limit=3)

test_experiment = Experiment(
    questions=questions,
    agent_models=[
        "claude-3-5-sonnet-20241022",
        # "hf:meta-llama/Meta-Llama-3-8B-Instruct",
    ],
    agent_toolss=[[], [math_eval]],
    judge_models=[
        "claude-3-5-sonnet-20241022",
        # "hf:meta-llama/Llama-2-7b-chat-hf",
    ],
    protocols=["blind", "propaganda", "debate", "consultancy"],
    bon_ns=[4],
    write_path=Path(__file__).parent
    / "test_results"
    / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
)


@pytest.mark.asyncio
async def test_experiment_runs():
    await test_experiment.experiment()
