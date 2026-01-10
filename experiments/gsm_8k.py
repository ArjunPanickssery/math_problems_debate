import asyncio  # noqa
from pathlib import Path
from datetime import datetime
from solib.protocols.protocols import *  # noqa
from solib.Experiment import Experiment
from solib.data.loading import GSM8K
from solib.utils.default_tools import math_eval

questions = GSM8K.data(limit=100)

init_exp = Experiment(
    questions=questions,
    agent_models=[
        "openrouter/deepseek/deepseek-v3.2",
        "openrouter/openai/gpt-oss-120b:exacto",
        # "deepinfra/openai/gpt-oss-120b" # cheaper than openrouter but I don't want to buy credits
        "openrouter/x-ai/grok-4.1-fast",
        "openrouter/minimax/minimax-m2.1",
        # "openai/gpt-5-mini",
        "gemini/gemini-3-flash-preview",
        "claude-haiku-4-5-20251001",
        # "novita/xiaomimimo/mimo-v2-flash"
    ],
    agent_toolss=[[], [math_eval]],
    judge_models=[
        "nvidia/nemotron-3-nano-30b-a3b",
        "openrouter/openai/gpt-oss-20b",
        "openai/gpt-5-nano",
        "google/gemini-2.5-flash-lite"
    ],
    protocols=["blind", "propaganda", "debate", "consultancy"],
    bon_ns=[1],#,4],#[1,4],  # , 8],#, 16, 32],
    write_path=Path(__file__).parent
    / "results"
    / f"gsm_8k_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    # continue_from = Path(__file__).parent / "results" / "init_exp_2025-02-14_07-50-51",
)


asyncio.run(init_exp.experiment(max_configs=None))
