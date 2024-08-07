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