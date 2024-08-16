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
poetry run pytest -s --runredundant # mostly just run this if pytest -s fails
poetry run pytest -s --runhf # to run even tests involving hf models
```

# Randomness

Throughout this repo, always use `solib.utils.random(*args, **kwargs)` instead of `random.random()`. This automatically sets the seed based on `args` (which should be the args of the function you're running), and optionally a `user_seed` kwarg. This is useful for caching and reproducibility.

# Simulation and cost estimation

You can do a simulated run of any function involving API calls in order to estimate cost before actually running the experiment. Simply initiate a `solib.cost_estimator.CostEstimator` object (say `ce`) and pass the arguments `simulate=True, cost_estimation={"cost_estimator": ce}` to the function you're calling, then `print(ce)` to see the results. 
```python
    cost_estimation = 
    {
        "cost_estimator": CostEstimator, # if not given, will not estimate cost
        "description": str, # optional, helpful for tracking and breakdown
        "input_tokens_estimator": Callable[[str], int], # optional, defaults to len(input_string) // 4.5
        "output_tokens_estimator": Callable[[str, int], list[int]], # optional, defaults to [1, 2048]
        "simstr_len": int, # optional, defaults to 1024
    }
```
In contributing to the repo, ensure that all functions involving API calls allow `**kwargs` to ensure an unbroken kwarg pathway to `get_llm_response`.

# Caching

LLM calls will be cached into the `.cache` folder by default. To break the cache, just pass some random extra argument to the function, e.g. `no_cache=True`.