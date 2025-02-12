from plotnine import (
    ggplot,
    aes,
    geom_bar,
    theme_minimal,
    theme,
    labs,
    geom_point,
    element_text,
    element_rect,
    element_line,
    save_as_pdf_pages,
)
import pandas as pd
import numpy as np
import json
import logging
from typing import Literal
from pathlib import Path
from solib.datatypes import Stats, Score
from solib.utils import serialize_to_json

LOGGER = logging.getLogger(__name__)


class Analyzer:
    def __init__(self, results_path: Path, plots_path: Path):
        self.results_path = results_path
        self.plots_path = plots_path
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
        for protocol_dir in self.results_path.iterdir():
            LOGGER.info(f"Loading from protocol_dir {protocol_dir}")
            if not protocol_dir.is_dir():
                LOGGER.warning(f"protocol_dir {protocol_dir} is not a directory")
                continue
            results[protocol_dir.name] = {}  # type: dict[str, dict[Literal["config", "stats"], dict|Stats]]
            for run_dir in protocol_dir.iterdir():
                if not run_dir.is_dir():
                    LOGGER.warning(f"run_dir {run_dir} is not a directory")
                    continue
                config_path = run_dir / "config.json"
                stats_path = run_dir / "stats.json"
                if not config_path.exists():
                    LOGGER.warning(f"config_path {config_path} does not exist")
                    continue
                if not stats_path.exists():
                    LOGGER.warning(f"stats_path {stats_path} does not exist")
                    continue
                with open(config_path) as f:
                    config = json.load(f)
                with open(stats_path) as f:
                    stats = json.load(f)
                results[protocol_dir.name][run_dir.name] = {
                    "config": config,
                    "stats": Stats.model_validate(stats),
                }
        self.results = results

    def get_protocol_asd(self, protocol) -> Score:
        protocol_results: dict[str, dict[Literal["config", "stats"], dict | Stats]] = (
            self.results[protocol]
        )
        asd_mean: Score = np.mean(
            [results["stats"].asd_mean for results in protocol_results.values()]
        )
        return asd_mean

    def get_protocol_asd_vs_ase(
        self, protocol, beta: Literal["0", "1", "inf"] = "1"
    ) -> list[tuple[Score, Score]]:
        """Get tuples of (ASE, ASD) for a given protocol"""
        ase_attr_str: str = f"ase_b{beta}_mean"
        protocol_results: dict[str, dict[Literal["config", "stats"], dict | Stats]] = (
            self.results[protocol]
        )
        ase_asd: list[tuple[Score, Score]] = [
            (getattr(results["stats"], ase_attr_str), results["stats"].asd_mean)
            for results in protocol_results.values()
        ]
        return ase_asd

    def get_asds(self) -> dict[str, Score]:
        """Get ASDs for all protocols in self.results"""
        return {protocol: self.get_protocol_asd(protocol) for protocol in self.results}

    def get_asd_vs_ases(
        self, beta: Literal["0", "1", "inf"] = "1"
    ) -> dict[str, list[tuple[Score, Score]]]:
        """Get ASDs vs ASEs for all protocols in self.results"""
        return {
            protocol: self.get_protocol_asd_vs_ase(protocol, beta)
            for protocol in self.results
        }

    def analyze_and_plot(
        self, scoring_rule: Literal["log", "logodds", "brier", "accuracy"] = "brier"
    ):
        """
        Run get_asds and get_asd_vs_ases and dump their results into
        self.plots_path/asds.json and self.plots_path/asd_vs_ases.json.

        Then generate:
        - a bar chart of ASDs for each protocol, and save it to self.plots_path/asds.png
        - a scatter plot of ASD vs ASE for each protocol, and save it to self.plots_path/{protocol}.png

        By default we take the brier score for everything.
        """
        asds_: dict[str, Score] = self.get_asds()
        asd_vs_ases_: dict[str, list[tuple[Score, Score]]] = self.get_asd_vs_ases()

        asds: dict[str, float] = {
            protocol: getattr(asd, scoring_rule) for protocol, asd in asds_.items()
        }
        asd_vs_ases: dict[str, list[tuple[float, float]]] = {
            protocol: [
                (getattr(ase, scoring_rule), getattr(asd, scoring_rule))
                for ase, asd in ase_asd_pairs
            ]
            for protocol, ase_asd_pairs in asd_vs_ases_.items()
        }

        self.plots_path.mkdir(parents=True, exist_ok=True)

        serialize_to_json(asds, self.plots_path / "asds.json")
        serialize_to_json(asd_vs_ases, self.plots_path / "asd_vs_ases.json")

        # Common theme with white background
        white_theme = theme_minimal() + theme(
            figure_size=(12, 6),
            panel_background=element_rect(fill='white'),
            plot_background=element_rect(fill='white'),
            panel_grid_major=element_line(color='lightgray'),
            panel_grid_minor=element_line(color='lightgray')
        )

        # Convert ASD data to DataFrame for plotting
        asd_df = pd.DataFrame(
            {"Protocol": list(asds.keys()), "ASD": list(asds.values())}
        )

        # Create and save bar plot
        asd_plot = (
            ggplot(asd_df, aes(x="Protocol", y="ASD"))
            + geom_bar(stat="identity", fill="steelblue", alpha=0.7)
            + white_theme
            + labs(
                title=f"Agent Score Difference (ASD) by Protocol ({scoring_rule})",
                x="Protocol",
                y="ASD Value",
            )
        )
        asd_plot.save(self.plots_path / "asds.png", dpi=300, verbose=False)

        # Create scatter plots for each protocol
        scatter_plots = []
        for protocol, ase_asd_pairs in asd_vs_ases.items():
            # Convert to DataFrame
            scatter_df = pd.DataFrame(ase_asd_pairs, columns=["ASE", "ASD"])
            scatter_df["Protocol"] = protocol

            plot = (
                ggplot(scatter_df, aes(x="ASE", y="ASD"))
                + geom_point(alpha=0.8, color="darkred", size=3, shape='x')
                + white_theme
                + theme(figure_size=(8, 6))  # Override figure size for scatter plots
                + labs(
                    title=f"ASD vs ASE for {protocol} ({scoring_rule})",
                    x="Agent Score Expected (ASE)",
                    y="Agent Score Difference (ASD)",
                )
            )

            # Save individual plot
            plot.save(self.plots_path / f"{protocol}.png", dpi=300, verbose=False)
            scatter_plots.append(plot)

        # save all scatter plots as a single PDF
        save_as_pdf_pages(scatter_plots, self.plots_path / "all_protocols.pdf")