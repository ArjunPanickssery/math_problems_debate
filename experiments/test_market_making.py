"""Test script to verify MarketMaking results are generated from Propaganda."""
import asyncio
from pathlib import Path
from datetime import datetime
from solib.Experiment import Experiment
from solib.data.loading import GSM8K

# Use just 2 questions to make it fast
questions = GSM8K.data(limit=2)

test_exp = Experiment(
    questions=questions,
    agent_models=[
        "openrouter/gpt-4o-mini-2024-07-18",
    ],
    agent_toolss=[[]],
    judge_models=[
        "openrouter/gpt-4o-mini-2024-07-18",
    ],
    protocols=["propaganda"],  # Only test Propaganda which should also generate MarketMaking
    bon_ns=[1],
    write_path=Path(__file__).parent
    / "results"
    / f"test_market_making_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
)

asyncio.run(test_exp.experiment(max_configs=None))
