# Usage

```bash
uv venv
uv sync
```

Create a `.env` file, see [.env.example](.env.example)

# Randomness

Throughout this repo, always use `solib.utils.random(*args, **kwargs)` instead of `random.random()`. This automatically sets the seed based on `args` (which should be the args of the function you're running), and optionally a `user_seed` kwarg. This is useful for caching and reproducibility.

The only exception to this is `costly` simulators.

# Pytest

```
uv run pytest -s
```

# LLM calls, simulation and cost estimation

We use the [`costly`](https://github.com/abhimanyupallavisudhir/costly) package for cost estimation ahead of making API calls. Specifically there are global variables `global_cost_log` and `simulate` defined in `solib.llm_utils`. As long as all LLM calls go through `get_llm_response()` from `solib.llm_utils`, stuff will work properly, and the cost estimate will be logged to `.costly/[datetime].jsonl`, `.costly/[datetime].totals.json`, `.costly/[datetime].totals_by_model.json`.

These files will be updated in real-time as the code is run.

NOTE: if you are not seeing anything being logged, or if the totals are not being created, it's probably because it's reading cached results and so doesn't have any costs.

`simulate` is controlled by an environment variable `SIMULATE` (which can be set to `True` or `False`), so you can run any command as `SIMULATE=True python your_script.py` to ensure all LLM calls are simulated.

# Caching

LLM calls will be cached into the `.litellm_cache` folder by default. To break the cache, pass `caching=False`, or environment variable `CACHING=False`. 

# Logging

Logs are written to `.logs/` by default. In general use the logger rather than print statements, warnings etc.

# Cleanup

`./clean.sh --all 5` will remove all but the 5 most recent items in each folder `.logs/`, `.litellm_cache/`, `.costly/`, and `tests/test_results/`. Options: `--logs`, `--cache`, `--costly`, `--tests`, `--all`. If you don't specify a number, it will remove everything.


