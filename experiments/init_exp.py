import asyncio  # noqa
from pathlib import Path
from datetime import datetime
from solib.protocols.protocols import *  # noqa
from solib.Experiment import Experiment
from solib.data.loading import GSM8K
from solib.utils.default_tools import math_eval

questions = GSM8K.data(limit=20)

init_exp = Experiment(
    questions=questions,
    agent_models=[
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "openrouter/deepseek/deepseek-chat" # "ollama_chat/deepseek-v3"
    ],
    agent_toolss=[[], [math_eval]],
    judge_models=[
        "localhf://meta-llama/Meta-Llama-3.1-8B",
        "openrouter/gpt-4o-mini-2024-07-18",
        # "gpt-4o-mini-2024-07-18",
    ],
    protocols=["propaganda", "debate", "consultancy"], # "blind"
    bon_ns=[1,4],  # , 8],#, 16, 32],
    write_path=Path(__file__).parent
    / "results"
    / f"init_exp_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
)


asyncio.run(init_exp.experiment(max_configs=None))
