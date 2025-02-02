import asyncio  # noqa
from pathlib import Path
from datetime import datetime
from solib.protocols.protocols import *  # noqa
from solib.Experiment import Experiment
from solib.data.loading import GSM8K
from solib.utils.default_tools import math_eval

questions = GSM8K.data(limit=10)

init_exp = Experiment(
    questions=questions,
    agent_models=[
        # "gpt-4o-2024-08-06",
        # "gpt-4-turbo-2024-04-09",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "gemini-1.5-pro",
        # "claude-3-sonnet-20240229",
        # "hf:meta-llama/Meta-Llama-3-8B-Instruct",
    ],
    agent_toolss=[[], [math_eval]],
    judge_models=[
        "ollama_chat/llama2:7b",
        "ollama_chat/llama3.1:8b",
        "gpt-4o-mini-2024-07-18",
        "gemini-1.5-flash-8b",
    ],
    protocols=["blind", "propaganda", "debate", "consultancy"],
    bon_ns=[4],  # , 8],#, 16, 32],
    write_path=Path(__file__).parent
    / "results"
    / f"init_exp_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
)


asyncio.run(init_exp.experiment(max_configs=100))
