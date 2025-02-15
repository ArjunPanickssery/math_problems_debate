from solib.analysis import Analyzer
from pathlib import Path

analyzer1 = Analyzer(
    results_path=Path(__file__).parent / "results" / "symmetric",
    plots_path=Path(__file__).parent / "analysis" / "symmetric",
)

analyzer2 = Analyzer(
    results_path=Path(__file__).parent / "results" / "asymmetric",
    plots_path=Path(__file__).parent / "analysis" / "asymmetric",
)


analyzer1.analyze_and_plot()
analyzer2.analyze_and_plot()
