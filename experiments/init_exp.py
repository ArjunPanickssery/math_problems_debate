import pytest
import asyncio  # noqa
from pathlib import Path
from datetime import datetime
from solib.protocols.protocols import *  # noqa
from solib.Experiment import Experiment
from solib.data.math import train_data
from solib.tool_use.default_tools import math_eval

questions = train_data()

test_experiment = Experiment(
    questions=questions,
    agent_models=[
        "gpt-4o-2024-08-06",
        "gpt-4-turbo-2024-04-09"
        "claude-3-5-sonnet-20241022"
        "hf:meta-llama/Meta-Llama-3-8B-Instruct",
    ],
    agent_toolss=[[], [math_eval]],
    judge_models=[
        "gpt-4o-mini-2024-07-18",
        "claude-3-5-haiku-20241022",
        "hf:meta-llama/Llama-2-7b-chat-hf",
    ],
    protocols=["blind", "propaganda", "debate", "consultancy"],
    write_path=Path(__file__).parent
    / "test_results"
    / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
)


@pytest.mark.asyncio
async def test_experiment_runs():
    await test_experiment.experiment()
