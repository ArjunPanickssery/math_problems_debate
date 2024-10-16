import pytest
import asyncio
from pathlib import Path
from solib.protocols.protocols import *
from solib.Experiment import Experiment
from solib.data.math import train_data

questions = train_data()[:3]

test_experiment = Experiment(
    questions=questions,
    agent_models=[
        "gpt-4o",
        # "hf:meta-llama/Meta-Llama-3-8B-Instruct",
    ],
    agent_toolss=[[]],
    judge_models=[
        "gpt-4o-mini",
        # "hf:meta-llama/Llama-2-7b-chat-hf",
    ],
    protocols=["blind"],
    write_path=Path(__file__).parent / "test_experiments",
)


@pytest.mark.asyncio
async def test_experiment_runs():
    await test_experiment.experiment()
