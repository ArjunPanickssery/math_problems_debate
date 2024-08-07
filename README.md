# Usage

```
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

Create a `.env` file, e.g.

```
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
HF_TOKEN=hf_your_hf_key
```

# Pytest

```
pytest -s
pytest -s --runmore # mostly just run this if pytest -s fails
pytest -s --runredundant # to run even tests involving hf models
```