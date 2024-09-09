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

We use the [`costly`](https://github.com/abhimanyupallavisudhir/costly) package for cost estimation ahead of making API calls.

In contributing to the repo, ensure that all functions involving API calls allow an unbroken kwarg pathway to the costly functions in `solib.llm_utils`. Actually, we only need `simulate`, `cost_estimation`, `description` to be passed through, so if you need to discriminate between `kwargs` for different purposes, just pass these parameters through all purposes.

# Caching

LLM calls will be cached into the `.cache` folder by default. To break the cache, just pass some random extra argument to the function, e.g. `no_cache=True`.