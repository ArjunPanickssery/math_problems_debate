import pytest  # noqa
import asyncio  # noqa
from pathlib import Path
from datetime import datetime
from solib.protocols.protocols import *  # noqa
from solib.Experiment import Experiment
from solib.data.loading import GSM8K
from solib.tool_use.default_tools import math_eval

questions = GSM8K.data(limit=100)

init_exp = Experiment(
    questions=questions,
    agent_models=[
        "gpt-4o-2024-08-06",
        "gpt-4-turbo-2024-04-09",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-2024022",
        "hf:meta-llama/Meta-Llama-3-8B-Instruct",
    ],
    agent_toolss=[[], [math_eval]],
    judge_models=[
        "gpt-4o-mini-2024-07-18",
        # "claude-3-5-haiku-20241022",
        "hf:meta-llama/Llama-2-7b-chat-hf",
        "hf:meta-llama/Meta-Llama-3-8B-Instruct",
    ],
    protocols=["blind", "propaganda", "debate", "consultancy"],
    bon_ns=[4],  # , 8],#, 16, 32],
    write_path=Path(__file__).parent
    / f"init_exp_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
)


asyncio.run(init_exp.experiment())
