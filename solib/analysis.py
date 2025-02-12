import matplotlib.pyplot as plt
import numpy as np
import json
import logging
from typing import Literal
from pathlib import Path
from solib.datatypes import Stats, Score

LOGGER = logging.getLogger(__name__)


class Analyzer:
    def __init__(self, path: Path):
        """
        Arguments:
            path (Path): results path
        """
        self.path = path
        self.load_results()

    def load_results(self):
        """
        results looks like

        {
            "Debate_t0_n2": {
                "_Aclaude...": {
                    config: dict,
                    stats: Stats
                },
            },
        }
        """
        results: dict[
            str, dict[str, dict[Literal["config", "stats"], dict | Stats]]
        ] = {}
        for protocol_dir in self.path.iterdir():
            LOGGER.info(f"Loading from protocol_dir {protocol_dir}")
            if not protocol_dir.is_dir():
                LOGGER.warning(f"protocol_dir {protocol_dir} is not a directory")
                continue
            results[str(protocol_dir)] = {}  # type: dict[str, dict[Literal["config", "stats"], dict|Stats]]
            for run_dir in protocol_dir.iterdir():
                if not run_dir.is_dir():
                    LOGGER.warning(f"run_dir {run_dir} is not a directory")
                    continue
                config_path = run_dir / "config.json"
                stats_path = run_dir / "stats.json"
                with open(config_path) as f:
                    config = json.load(f)
                with open(stats_path) as f:
                    stats = json.load(f)
                results[str(protocol_dir)][str(run_dir)] = {
                    "config": config,
                    "stats": Stats.model_validate(stats),
                }
        self.results = results

    def get_protocol_asd(self, protocol) -> Score:
        protocol_results: dict[str, dict[Literal["config", "stats"], dict | Stats]] = (
            self.results[protocol]
        )
        asd_mean: Score = np.mean(
            results["stats"].asd_mean for results in protocol_results.values()
        )
        return asd_mean

    def get_protocol_asd_vs_ase(
        self, protocol, beta: Literal["0", "1", "inf"] = "1"
    ) -> list[tuple[Score, Score]]:
        ase_attr_str: str = f"ase_b{beta}_mean"
        protocol_results: dict[str, dict[Literal["config", "stats"], dict | Stats]] = (
            self.results[protocol]
        )
        ase_asd: list[tuple[Score, Score]] = [
            (getattr(results["stats"], ase_attr_str), results["stats"].asd_mean)
            for results in protocol_results.values()
        ]
        return ase_asd
    
    
