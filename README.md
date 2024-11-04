# Usage

```
$ python -m venv venv
$ source venv/bin/activate
$ poetry install
```

Create a `.env` file, e.g.

```
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
HF_TOKEN=hf_your_hf_key
```

# Pytest

```
poetry run pytest -s
```

# Randomness

Throughout this repo, always use `solib.utils.random(*args, **kwargs)` instead of `random.random()`. This automatically sets the seed based on `args` (which should be the args of the function you're running), and optionally a `user_seed` kwarg. This is useful for caching and reproducibility.

The only exception to this is `costly` simulators.

# LLM calls, simulation and cost estimation

We use the [`costly`](https://github.com/abhimanyupallavisudhir/costly) package for cost estimation ahead of making API calls. Specifically there are global variables `global_cost_log` and `simulate` defined in `solib.llm_utils`. As long as all LLM calls go through `get_llm_response()` from `solib.llm_utils`, stuff will work properly, and the cost estimate will be logged to `.costly/[datetime].jsonl`, `.costly/[datetime].totals.json`, `.costly/[datetime].totals_by_model.json`.

NOTE: if you are not seeing anything being logged, or if the totals are not being created, it's probably because it's reading cached results and so doesn't have any costs.

`simulate` is controlled by an environment variable `SIMULATE` (which can be set to `True` or `False`), so you can run any command as `SIMULATE=True python your_script.py` to ensure all LLM calls are simulated.

# Caching

LLM calls will be cached into the `.cache` folder by default. To break the cache, just pass some random extra argument to the function, e.g. `no_cache=True`.

# Logging

Logs are written to `.logs/` by default. In general use the logger rather than print statements, warnings etc.