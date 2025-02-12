import asyncio  # noqa
import pytest  # noqa
import time
from pathlib import Path
from datetime import datetime
from solib.protocols.protocols import Debate, Blind, Propaganda, Consultancy  # noqa
from solib.Experiment import Experiment
from solib.data.loading import GSM8K
from solib.utils import random
from solib.utils.default_tools import math_eval
from solib.protocols.abstract import QA_Agent, Judge
from solib.protocols.agents import BestOfN_Agent
from solib.analysis import Analyzer

experiment = Experiment(
    questions=[],
    agent_models=[
        "gpt-4o-mini","hf:meta-llama/Meta-Llama-3-8B-Instruct"],
    agent_toolss=[[]],
    judge_models=["gpt-4o-mini", "hf:meta-llama/Meta-Llama-3-8B-Instruct"],
)

DATE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
# TEST_RESULTS_PATH = Path(__file__).parent / "test_results" / DATE
TEST_RESULTS_PATH = Path(__file__).parent / "test_results" / "TEST_CONTINUE"
ANALYSIS_PATH = Path(__file__).parent / "test_analysis" / DATE

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
                "agent": QA_Agent("openrouter/gpt-4o-mini", tools=[math_eval]),
                "adversary": BestOfN_Agent(n=4, agent=QA_Agent("gpt-4o-mini")),
            },
        }
    )
    path = str(path)
    assert path.startswith("experiments/results_")
    assert path.endswith("/Debate_t1_n4/_Jgpt-4o-mini_Agpt-4o-mini-math_eval_BBON-4-gpt-4o-mini")


questions = GSM8K.data(limit=1)

test_experiment = Experiment(
    questions=questions,
    agent_models=[
        # "gpt-4o-mini",
        "claude-3-5-sonnet-20241022",
        # "openrouter/deepseek/deepseek-chat",
        # "hf:meta-llama/Meta-Llama-3-8B-Instruct",
    ],
    agent_toolss=[[], [math_eval]],
    judge_models=[
        "openrouter/gpt-4o-mini",
        # "claude-3-5-sonnet-20241022",
        # "hf:meta-llama/Llama-2-7b-chat-hf",
    ],
    protocols=["blind", "propaganda", "consultancy", "debate"],
    bon_ns=[1],
    write_path=TEST_RESULTS_PATH,
    continue_from=TEST_RESULTS_PATH,
)

@pytest.mark.asyncio
async def test_experiment_runs():
    await test_experiment.experiment(max_configs=None)

@pytest.mark.asyncio
async def test_experiment_continue_from():
    t0 = time.time()
    await test_experiment.experiment(max_configs=None)
    t1 = time.time()
    assert t1 - t0 < 10

def test_analysis():
    analyzer = Analyzer(TEST_RESULTS_PATH, ANALYSIS_PATH)
    analyzer.analyze_and_plot()