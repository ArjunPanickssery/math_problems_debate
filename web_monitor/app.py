"""
Web interface to monitor experiment progress.
Run with: python -m web_monitor.app
"""
import json
import jsonlines
from pathlib import Path
from flask import Flask, render_template, jsonify, abort

app = Flask(__name__)

# Base directory for experiment results
RESULTS_BASE = Path(__file__).parent.parent / "experiments"


def find_result_dirs():
    """Find all result directories (results_* folders)."""
    result_dirs = []
    for p in RESULTS_BASE.iterdir():
        if p.is_dir() and p.name.startswith("results"):
            result_dirs.append(p)
    return sorted(result_dirs, key=lambda x: x.name, reverse=True)


def get_experiment_configs(results_dir: Path):
    """Get all experiment configurations in a results directory."""
    experiments = []

    if not results_dir.exists():
        return experiments

    # Walk through protocol directories
    for protocol_dir in results_dir.iterdir():
        if not protocol_dir.is_dir() or protocol_dir.name == "prompt_history":
            continue

        # Walk through run directories within each protocol
        for run_dir in protocol_dir.iterdir():
            if not run_dir.is_dir() or run_dir.name == "prompt_history":
                continue

            config_path = run_dir / "config.json"
            results_path = run_dir / "results.jsonl"
            stats_path = run_dir / "stats.json"

            exp_info = {
                "path": str(run_dir.relative_to(results_dir)),
                "full_path": str(run_dir),
                "protocol": protocol_dir.name,
                "run_name": run_dir.name,
                "has_config": config_path.exists(),
                "has_results": results_path.exists(),
                "has_stats": stats_path.exists(),
                "completed": 0,
                "total": 0,
                "config": None,
                "stats": None,
            }

            # Load config if exists
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        exp_info["config"] = json.load(f)
                except Exception:
                    pass

            # Count questions in results
            if results_path.exists():
                try:
                    with jsonlines.open(results_path, 'r') as reader:
                        exp_info["completed"] = sum(1 for _ in reader)
                except Exception:
                    pass

            # Load stats if exists
            if stats_path.exists():
                try:
                    with open(stats_path) as f:
                        exp_info["stats"] = json.load(f)
                except Exception:
                    pass

            experiments.append(exp_info)

    return sorted(experiments, key=lambda x: x["path"])


def load_questions(run_dir: Path):
    """Load all questions from a run's results.jsonl."""
    results_path = run_dir / "results.jsonl"
    questions = []

    if not results_path.exists():
        return questions

    try:
        with jsonlines.open(results_path, 'r') as reader:
            for i, q in enumerate(reader):
                questions.append({
                    "index": i,
                    "question_text": q.get("question", "")[:100] + "..." if len(q.get("question", "")) > 100 else q.get("question", ""),
                    "full_question": q.get("question", ""),
                    "data": q,
                })
    except Exception as e:
        print(f"Error loading questions: {e}")

    return questions


@app.route("/")
def index():
    """List all result directories."""
    result_dirs = find_result_dirs()
    return render_template("index.html", result_dirs=result_dirs)


@app.route("/results/<path:results_name>")
def experiments_list(results_name):
    """List all experiments in a results directory."""
    results_dir = RESULTS_BASE / results_name
    if not results_dir.exists():
        abort(404)

    experiments = get_experiment_configs(results_dir)
    return render_template("experiments.html",
                          results_name=results_name,
                          experiments=experiments)


@app.route("/results/<path:results_name>/experiment/<path:exp_path>")
def questions_list(results_name, exp_path):
    """List all questions in an experiment."""
    run_dir = RESULTS_BASE / results_name / exp_path
    if not run_dir.exists():
        abort(404)

    questions = load_questions(run_dir)

    # Load config
    config = None
    config_path = run_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
        except Exception:
            pass

    return render_template("questions.html",
                          results_name=results_name,
                          exp_path=exp_path,
                          questions=questions,
                          config=config)


@app.route("/results/<path:results_name>/experiment/<path:exp_path>/question/<int:q_idx>")
def question_detail(results_name, exp_path, q_idx):
    """Show detailed view of a single question's protocol run."""
    run_dir = RESULTS_BASE / results_name / exp_path
    if not run_dir.exists():
        abort(404)

    questions = load_questions(run_dir)
    if q_idx >= len(questions):
        abort(404)

    question = questions[q_idx]

    # Load config
    config = None
    config_path = run_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
        except Exception:
            pass

    return render_template("question_detail.html",
                          results_name=results_name,
                          exp_path=exp_path,
                          q_idx=q_idx,
                          question=question,
                          config=config,
                          total_questions=len(questions))


# API endpoints for AJAX if needed
@app.route("/api/results")
def api_results():
    """API endpoint to list all result directories."""
    result_dirs = find_result_dirs()
    return jsonify([{"name": d.name, "path": str(d)} for d in result_dirs])


@app.route("/api/results/<path:results_name>/experiments")
def api_experiments(results_name):
    """API endpoint to list experiments in a results directory."""
    results_dir = RESULTS_BASE / results_name
    if not results_dir.exists():
        return jsonify({"error": "Not found"}), 404

    experiments = get_experiment_configs(results_dir)
    return jsonify(experiments)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run experiment monitor web interface")
    parser.add_argument("--port", type=int, default=5000, help="Port to run on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    print(f"Starting experiment monitor on http://{args.host}:{args.port}")
    print(f"Looking for results in: {RESULTS_BASE}")
    app.run(host=args.host, port=args.port, debug=args.debug)
