from plotnine import (
    ggplot,
    aes,
    geom_bar,
    geom_errorbar,
    geom_errorbarh,
    theme_minimal,
    theme,
    labs,
    annotate,
    geom_point,
    geom_text,
    element_rect,
    element_line,
    save_as_pdf_pages,
)
from scipy import stats
from plotnine import stat_smooth
import pandas as pd
import numpy as np
import json
import logging
from typing import Literal
from pathlib import Path
from solib.datatypes import Stats, Score
from solib.utils import serialize_to_json

LOGGER = logging.getLogger(__name__)

def shortened_protocol_path(protocol_path: str) -> str:
    """Shorten the protocol name for better readability in plots"""
    PATH_SHORTENER = {
        "Debate_t0_n2": "Debate [sequential, 2 turns]",
        "Debate_t0_n4": "Debate [sequential, 4 turns]",
        "Debate_t1_n2": "Debate [simultaneous, 2 turns]",
        "Debate_t1_n4": "Debate [simultaneous, 4 turns]",
        "Consultancy_t0_n2": "Consultancy [client starts, 2 turns]",
        "Consultancy_t0_n4": "Consultancy [client starts, turns]",
        "Consultancy_t1_n2": "Consultancy [consultant starts, 2 turns]",
        "Consultancy_t1_n4": "Consultancy [consultant starts, 4 turns]",
    }
    return PATH_SHORTENER.get(protocol_path, protocol_path)

def shortened_call_path(call_path: str) -> str:
    """Shorten the run ID for better readability in plots"""
    # FRAGILE.
    index_of_A = call_path.index("_A")
    index_of_J = call_path.index("_J")
    try:
        index_of_B = call_path.index("_B")
    except ValueError:
        index_of_B = -1
    agent_name = call_path[index_of_A + 2:index_of_J]
    judge_name = call_path[index_of_J + 2:index_of_B]
    adversary_name = call_path[index_of_B + 2:]
    agent_name = agent_name.replace("-20241022", "")
    agent_name = agent_name.replace("-2024-07-18", "")
    agent_name = agent_name.replace("-math_eval", "+calculator")
    agent_name = agent_name.replace("deepseek-chat", "deepseek-v3")
    return agent_name


class Analyzer:

    WHITE_THEME = theme_minimal() + theme(
        figure_size=(12, 6),
        panel_background=element_rect(fill='white'),
        plot_background=element_rect(fill='white'),
        panel_grid_major=element_line(color='lightgray'),
        panel_grid_minor=element_line(color='lightgray')
    )

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
                    n: int,
                    stats: Stats
                },
            },
        }
        """
        results: dict[
            str, dict[str, dict[Literal["config", "n", "stats"], dict | int | Stats]]
        ] = {}
        for protocol_dir in self.results_path.iterdir():
            LOGGER.info(f"Loading from protocol_dir {protocol_dir}")
            if not protocol_dir.is_dir():
                LOGGER.warning(f"protocol_dir {protocol_dir} is not a directory")
                continue
            results[protocol_dir.name] = {}
            for run_dir in protocol_dir.iterdir():
                if not run_dir.is_dir():
                    LOGGER.warning(f"run_dir {run_dir} is not a directory")
                    continue
                config_path = run_dir / "config.json"
                stats_path = run_dir / "stats.json"
                results_path = run_dir / "results.jsonl"
                if not config_path.exists():
                    LOGGER.warning(f"config_path {config_path} does not exist")
                    continue
                if not stats_path.exists():
                    LOGGER.warning(f"stats_path {stats_path} does not exist")
                    continue
                if not results_path.exists():
                    LOGGER.warning(f"results_path {results_path} does not exist")
                    continue
                
                # Count lines in results.jsonl for n
                with open(results_path) as f:
                    n = sum(1 for _ in f)
                
                with open(config_path) as f:
                    config = json.load(f)
                with open(stats_path) as f:
                    stats = json.load(f)
                results[protocol_dir.name][run_dir.name] = {
                    "config": config,
                    "n": n,
                    "stats": Stats.model_validate(stats),
                }
            if not results[protocol_dir.name] or all(
                not run for run in results[protocol_dir.name].values()
            ):
                # avoid nans
                LOGGER.warning(f"protocol_dir {protocol_dir} has no valid runs")
                del results[protocol_dir.name]
        self.results = results

    def get_protocol_asd(self, protocol) -> tuple[Score, Score, int]:
        """Returns (mean, std, n) for the protocol's ASD"""
        protocol_results = self.results[protocol]
        asd_means = [results["stats"].asd_mean for results in protocol_results.values()]
        asd_stds = [results["stats"].asd_std for results in protocol_results.values()]
        ns = [results["n"] for results in protocol_results.values()]
        
        asd_mean = np.mean(asd_means)
        asd_std = np.sqrt(np.sum(np.array(asd_stds)**2)) / len(asd_stds)
        total_n = sum(ns)
        
        return asd_mean, asd_std, total_n

    def get_protocol_asd_vs_ase(
        self, protocol, beta: Literal["0", "1", "inf"] = "1"
    ) -> list[tuple[str, Score, Score, Score, Score, int]]:
        """Get tuples of (run_id, ASE_mean, ASE_std, ASD_mean, ASD_std, n) for a given protocol"""
        ase_mean_attr = f"ase_b{beta}_mean"
        ase_std_attr = f"ase_b{beta}_std"
        protocol_results = self.results[protocol]
        
        return [
            (
                run_id,
                getattr(results["stats"], ase_mean_attr),
                getattr(results["stats"], ase_std_attr),
                results["stats"].asd_mean,
                results["stats"].asd_std,
                results["n"]
            )
            for run_id, results in protocol_results.items()
        ]

    def get_asds(self) -> dict[str, Score]:
        """Get ASDs for all protocols in self.results"""
        return {protocol: self.get_protocol_asd(protocol) for protocol in self.results}

    def get_asd_vs_ases(
        self, beta: Literal["0", "1", "inf"] = "1"
    ) -> dict[str, list[tuple[str, Score, Score]]]:
        """Get ASDs vs ASEs for all protocols in self.results"""
        return {
            protocol: self.get_protocol_asd_vs_ase(protocol, beta)
            for protocol in self.results
        }

    def analyze_and_plot(
        self,
        scoring_rule: Literal["log", "logodds", "brier", "accuracy"] = "brier",
        beta: Literal["0", "1", "inf"] = "1",
        show_error_bars_barchart: bool = True,
        show_error_bars_scatter: bool = True,
        show_labels_scatter: bool = True,
        std_factor: float = 1.0,
    ):
        """
        Run get_asds and get_asd_vs_ases and dump their results into
        self.plots_path/asds.json and self.plots_path/asd_vs_ases.json.

        Then generate:
        - a bar chart of ASDs for each protocol, save to self.plots_path/asds.png
        - scatter plot of ASD vs ASE for each protocol, save to self.plots_path/{protocol}.png

        Args:
            scoring_rule: Which scoring rule to use for the metrics
            beta: Beta parameter for ASE calculation
            show_error_bars_barchart: Whether to show error bars on the bar chart
            show_error_bars_scatter: Whether to show error bars on scatter plots
            show_labels_scatter: Whether to show point labels on scatter plots
            std_factor: Number of standard deviations to use for error bars, e.g. std_factor for 95% CI
        """
        asds_: dict[str, tuple[Score, Score, int]] = {
            protocol: self.get_protocol_asd(protocol) for protocol in self.results
        }
        asd_vs_ases_: dict[str, list[tuple[str, Score, Score, Score, Score, int]]] = self.get_asd_vs_ases(beta)

        # Process ASDs with error bars
        asds: dict[str, tuple[float, float, int]] = {
            protocol: (
                getattr(asd_mean, scoring_rule),
                getattr(asd_std, scoring_rule),
                n
            )
            for protocol, (asd_mean, asd_std, n) in asds_.items()
        }
        
        # Process ASE vs ASD with error bars
        asd_vs_ases: dict[str, list[tuple[str, float, float, float, float, int]]] = {
            protocol: [
                (
                    run_id,
                    getattr(ase_mean, scoring_rule),
                    getattr(ase_std, scoring_rule),
                    getattr(asd_mean, scoring_rule),
                    getattr(asd_std, scoring_rule),
                    n
                )
                for run_id, ase_mean, ase_std, asd_mean, asd_std, n in ase_asd_pairs
            ]
            for protocol, ase_asd_pairs in asd_vs_ases_.items()
        }

        self.plots_path.mkdir(parents=True, exist_ok=True)
        serialize_to_json(asds, self.plots_path / "asds.json")
        serialize_to_json(asd_vs_ases, self.plots_path / "asd_vs_ases.json")

        # Convert ASD data to DataFrame with error bars
        asd_df = pd.DataFrame([
            {
                "Protocol": protocol,
                "ASD": asd,
                "ASD_std": std,
                "n": n
            }
            for protocol, (asd, std, n) in asds.items()
        ])

        # Calculate confidence intervals
        if show_error_bars_barchart:
            asd_df["ymin"] = asd_df["ASD"] - std_factor * asd_df["ASD_std"] / np.sqrt(asd_df["n"])
            asd_df["ymax"] = asd_df["ASD"] + std_factor * asd_df["ASD_std"] / np.sqrt(asd_df["n"])

        # Bar plot with optional error bars
        asd_plot = (
            ggplot(asd_df, aes(x="Protocol", y="ASD"))
            + geom_bar(stat="identity", fill="steelblue", alpha=0.7)
            + self.WHITE_THEME
            + labs(
                title=f"Agent Score Difference (ASD) by Protocol ({scoring_rule})",
                x="Protocol",
                y="ASD Value",
            )
        )
        
        if show_error_bars_barchart:
            asd_plot = asd_plot + geom_errorbar(aes(ymin="ymin", ymax="ymax"), width=0.2)
        
        asd_plot.save(self.plots_path / "asds.png", dpi=300, verbose=False)

        # Scatter plots with optional error bars and labels
        scatter_plots = []
        for protocol, ase_asd_pairs in asd_vs_ases.items():
            scatter_df = pd.DataFrame([
                {
                    "Run": run_id,
                    "ASE": ase,
                    "ASE_std": ase_std,
                    "ASD": asd,
                    "ASD_std": asd_std,
                    "n": n,
                    "Protocol": protocol,
                }
                for run_id, ase, ase_std, asd, asd_std, n in ase_asd_pairs
            ])
            
            scatter_df["ASE_min"] = scatter_df["ASE"] - std_factor * scatter_df["ASE_std"] / np.sqrt(scatter_df["n"])
            scatter_df["ASE_max"] = scatter_df["ASE"] + std_factor * scatter_df["ASE_std"] / np.sqrt(scatter_df["n"])
            scatter_df["ASD_min"] = scatter_df["ASD"] - std_factor * scatter_df["ASD_std"] / np.sqrt(scatter_df["n"])
            scatter_df["ASD_max"] = scatter_df["ASD"] + std_factor * scatter_df["ASD_std"] / np.sqrt(scatter_df["n"])
            
            if show_labels_scatter:
                scatter_df["Label"] = scatter_df["Run"].apply(shortened_call_path)

            # Calculate correlation
            corr, p_value = stats.pearsonr(scatter_df["ASE"], scatter_df["ASD"])
            corr_text = f"r = {corr:.3f} (p = {p_value:.3f})"

            plot = (
                ggplot(scatter_df, aes(x="ASE", y="ASD"))
                + geom_point(alpha=0.8, color="darkred", size=3, shape='x')
                + stat_smooth(method="lm", color="blue", alpha=0.3)
                # Place annotation at bottom-right using data coordinates
                + annotate(
                    "text",
                    x=scatter_df["ASE_max"].max(),
                    y=scatter_df["ASD_min"].min(),
                    label=corr_text,
                    ha="right",
                    va="bottom", 
                    color="darkred",
                    fontweight="bold",
                    size=8
                )
                + self.WHITE_THEME
                + theme(figure_size=(8, 6))
                + labs(
                    title=f"ASD vs ASE for {shortened_protocol_path(protocol)} ({scoring_rule})",
                    x="Agent Score Expected (ASE)",
                    y="Agent Score Difference (ASD)",
                )
            )

            if show_error_bars_scatter:
                plot = (
                    plot 
                    + geom_errorbar(aes(ymin="ASD_min", ymax="ASD_max"), width=0.002)
                    + geom_errorbarh(aes(xmin="ASE_min", xmax="ASE_max"), height=0.002)
                )
                
            if show_labels_scatter:
                plot = plot + geom_text(aes(label="Label"), nudge_x=0.002, nudge_y=0.002, size=8)

            plot.save(self.plots_path / f"{protocol}.png", dpi=300, verbose=False)
            scatter_plots.append(plot)

        save_as_pdf_pages(scatter_plots, self.plots_path / "all_protocols.pdf")