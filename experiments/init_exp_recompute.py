from pathlib import Path
from solib.Experiment import Experiment

init_exp = Experiment(
    questions=None,
    judge_models=None,
    agent_models=None,
    write_path=Path(__file__).parent / "results" / "asymmetric",
)

init_exp.recompute_stats()
