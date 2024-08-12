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