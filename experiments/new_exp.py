import asyncio  # noqa
from pathlib import Path
from datetime import datetime
from solib.protocols.protocols import *  # noqa
from solib.Experiment import Experiment
from solib.data.loading import GSM8K, MMLU, TruthfulQA, PrOntoQA, GPQA, QuALITY
from solib.utils.default_tools import math_eval

N = 1
AGENT_MODELS = [
        "openrouter/deepseek/deepseek-v3.2",
        # "openrouter/openai/gpt-oss-120b:exacto",
        # "deepinfra/openai/gpt-oss-120b" # cheaper than openrouter but I don't want to buy credits
        "openrouter/x-ai/grok-4.1-fast",
        # "openrouter/minimax/minimax-m2.1",
        # "openai/gpt-5-mini",
        "gemini/gemini-3-flash-preview",
        "claude-haiku-4-5-20251001",
        # "novita/xiaomimimo/mimo-v2-flash"
    ]
JUDGE_MODELS = [
    "openrouter/nvidia/nemotron-3-nano-30b-a3b",
    # "openrouter/openai/gpt-oss-20b",
    "gpt-5-nano",
    "gemini/gemini-2.5-flash-lite",
]
AGENT_TOOLS = [[]]
AGENT_TOOLS_MATH = [[math_eval]]
PROTOCOLS = ["blind", "propaganda", "debate", "consultancy"]
BON_NS = [1]
NUM_TURNSS = [2]
NUM_TURNSS_GPQA = [2, 4, 6]
DEBATE_TOGGLE = [True] # simultaneous only
CONSULTANCY_TOGGLE = [True] # consultant_goes_first only
QUOTE_MAX_LENGTH_QUALITY = 500

def make_write_path(dataset_name: str):
    return (
        Path(__file__).parent
        / "results"
        / f"{dataset_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )

datasets = {
    "gsm8k": GSM8K.data(limit=N),
    "mmlu": MMLU.data(limit=N),
    "truthfulqa": TruthfulQA.data(limit=N),
    "prontoqa": PrOntoQA.data(limit=N),  # has source_text for quotes
    "gpqa": GPQA.data(limit=N),
    "quality": QuALITY.data(limit=N),
}

CONFIG_1 = {
    "agent_models": AGENT_MODELS,
    "agent_toolss": AGENT_TOOLS,
    "judge_models": JUDGE_MODELS,
    "protocols": PROTOCOLS,
    "num_turnss": NUM_TURNSS,
    "bon_ns": BON_NS,
    "debate_toggle": DEBATE_TOGGLE,
    "consultancy_toggle": CONSULTANCY_TOGGLE,
    # "write_path": Path(__file__).parent
    # / "results"
    # / f"gsm_8k_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    # "continue_from": Path(__file__).parent / "results" / "init_exp_2025-02-14_07-50-51",
}

CONFIG_GPQA = {
    "agent_models": AGENT_MODELS,
    "agent_toolss": AGENT_TOOLS,
    "judge_models": JUDGE_MODELS,
    "protocols": PROTOCOLS,
    "num_turnss": NUM_TURNSS_GPQA,
    "bon_ns": BON_NS,
    "debate_toggle": DEBATE_TOGGLE,
    "consultancy_toggle": CONSULTANCY_TOGGLE,
    # "write_path": Path(__file__).parent
    # / "results"
    # / f"gsm_8k_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    # "continue_from": Path(__file__).parent / "results" / "init_exp_2025-02-14_07-50-51",
}

CONFIG_MATH = {
    "agent_models": AGENT_MODELS,
    "agent_toolss": AGENT_TOOLS_MATH,
    "judge_models": JUDGE_MODELS,
    "protocols": PROTOCOLS,
    "num_turnss": NUM_TURNSS,
    "bon_ns": BON_NS,
    "debate_toggle": DEBATE_TOGGLE,
    "consultancy_toggle": CONSULTANCY_TOGGLE,
    # "write_path": Path(__file__).parent
    # / "results"
    # / f"gsm_8k_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    # "continue_from": Path(__file__).parent / "results" / "init_exp_2025-02-14_07-50-51",
}


CONFIG_QUALITY = {
    "agent_models": AGENT_MODELS,
    "agent_toolss": AGENT_TOOLS,
    "judge_models": JUDGE_MODELS,
    "protocols": PROTOCOLS,
    "num_turnss": NUM_TURNSS,
    "bon_ns": BON_NS,
    "debate_toggle": DEBATE_TOGGLE,
    "consultancy_toggle": CONSULTANCY_TOGGLE,
    "quote_max_length": QUOTE_MAX_LENGTH_QUALITY,
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

asyncio.run(
    Experiment(
        questions=datasets["truthfulqa"],
        write_path=make_write_path("truthfulqa"),
        **CONFIG_1,
    ).experiment(max_configs=None)
)

asyncio.run(
    Experiment(
        questions=datasets["gpqa"],
        write_path=make_write_path("gpqa"),
        **CONFIG_GPQA,
    ).experiment(max_configs=None)
)

asyncio.run(
    Experiment(
        questions=datasets["quality"],
        write_path=make_write_path("quality"),
        **CONFIG_QUALITY,
    ).experiment(max_configs=None)
)


asyncio.run(
    Experiment(
        questions=datasets["mmlu"],
        write_path=make_write_path("mmlu"),
        **CONFIG_1,
    ).experiment(max_configs=None)
)


# asyncio.run(
#     Experiment(
#         questions=datasets["prontoqa"],
#         write_path=make_write_path("prontoqa"),
#         **CONFIG_1,
#     ).experiment(max_configs=None)
# )