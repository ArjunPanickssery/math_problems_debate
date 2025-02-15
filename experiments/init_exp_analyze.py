from solib.analysis import Analyzer
from pathlib import Path

analyzer = Analyzer(
    results_path=Path(__file__).parent / "results" / "symmetric",
    plots_path=Path(__file__).parent / "analysis" / "symmetric",
)

analyzer.analyze_and_plot()
