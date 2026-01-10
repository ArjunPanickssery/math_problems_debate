import asyncio  # noqa
from pathlib import Path
from datetime import datetime
from solib.protocols.protocols import *  # noqa
from solib.Experiment import Experiment
from solib.data.loading import GSM8K, MMLU, TruthfulQA, PrOntoQA, GPQA
from solib.utils.default_tools import math_eval


def make_write_path(dataset_name: str):
    return (
        Path(__file__).parent
        / "results"
        / f"{dataset_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )


datasets = {
    "gsm8k": GSM8K.data(limit=1),
    "mmlu": MMLU.data(limit=1),
    "truthfulqa": TruthfulQA.data(limit=1),
    "prontoqa": PrOntoQA.data(limit=1),  # has source_text for quotes
    "gpqa": GPQA.data(limit=1),
}

CONFIG_1 = {
    "agent_models": [
        "openrouter/deepseek/deepseek-v3.2",
        # "openrouter/openai/gpt-oss-120b:exacto",
        # "deepinfra/openai/gpt-oss-120b" # cheaper than openrouter but I don't want to buy credits
        "openrouter/x-ai/grok-4.1-fast",
        # "openrouter/minimax/minimax-m2.1",
        # "openai/gpt-5-mini",
        "gemini/gemini-3-flash-preview",
        "claude-haiku-4-5-20251001",
        # "novita/xiaomimimo/mimo-v2-flash"
    ],
    # "agent_toolss": [[], [math_eval]],
    "agent_toolss": [[]],
    "judge_models": [
        "openrouter/nvidia/nemotron-3-nano-30b-a3b",
        "openrouter/openai/gpt-oss-20b",
        # "gpt-5-nano",
        "gemini/gemini-2.5-flash-lite",
    ],
    "protocols": ["blind", "propaganda", "debate", "consultancy"],
    "bon_ns": [1],  # ,4],#[1,4],  # , 8],#, 16, 32],
    # "write_path": Path(__file__).parent
    # / "results"
    # / f"gsm_8k_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    # "continue_from": Path(__file__).parent / "results" / "init_exp_2025-02-14_07-50-51",
}

CONFIG_MATH = {
    "agent_models": [
        "openrouter/deepseek/deepseek-v3.2",
        # "openrouter/openai/gpt-oss-120b:exacto",
        # "deepinfra/openai/gpt-oss-120b" # cheaper than openrouter but I don't want to buy credits
        "openrouter/x-ai/grok-4.1-fast",
        # "openrouter/minimax/minimax-m2.1",
        # "openai/gpt-5-mini",
        "gemini/gemini-3-flash-preview",
        "claude-haiku-4-5-20251001",
        # "novita/xiaomimimo/mimo-v2-flash"
    ],
    "agent_toolss": [[], [math_eval]],
    # "agent_toolss": [[]],
    "judge_models": [
        "openrouter/nvidia/nemotron-3-nano-30b-a3b",
        "openrouter/openai/gpt-oss-20b",
        # "gpt-5-nano",
        "gemini/gemini-2.5-flash-lite",
    ],
    "protocols": ["blind", "propaganda", "debate", "consultancy"],
    "bon_ns": [1],  # ,4],#[1,4],  # , 8],#, 16, 32],
    # "write_path": Path(__file__).parent
    # / "results"
    # / f"gsm_8k_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    # "continue_from": Path(__file__).parent / "results" / "init_exp_2025-02-14_07-50-51",
}

asyncio.run(
    Experiment(
        questions=datasets["gsm8k"],
        write_path=make_write_path("gsm8k"),
        **CONFIG_MATH,
    ).experiment(max_configs=None)
)

# asyncio.run(
#     Experiment(
#         questions=datasets["mmlu"],
#         write_path=make_write_path("mmlu"),
#         **CONFIG_1,
#     ).experiment(max_configs=None)
# )

asyncio.run(
    Experiment(
        questions=datasets["truthfulqa"],
        write_path=make_write_path("truthfulqa"),
        **CONFIG_1,
    ).experiment(max_configs=None)
)

asyncio.run(
    Experiment(
        questions=datasets["prontoqa"],
        write_path=make_write_path("prontoqa"),
        **CONFIG_1,
    ).experiment(max_configs=None)
)

# asyncio.run(
#     Experiment(
#         questions=datasets["gpqa"],
#         write_path=make_write_path("gpqa"),
#         **CONFIG_1,
#     ).experiment(max_configs=None)
# )