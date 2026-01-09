# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**solib** (Scalable Oversight Library) is a research library implementing AI debate and oversight protocols for evaluating language models on math reasoning tasks. It explores how different communication protocols (debate, propaganda, consultancy, blind judging) affect an AI judge's ability to identify correct answers.

## Commands

```bash
# Setup
uv venv && uv sync

# Run tests
uv run pytest -s

# Run main experiment
python experiments/init_exp.py

# Cleanup old logs/cache (keep last N items)
./clean.sh --all 5
```

## Architecture

### Core Flow
1. Load dataset (e.g., `GSM8K.data(limit=100)`)
2. Configure `Experiment` with models, protocols, parameters
3. Call `exp.experiment()` which runs all config combinations
4. Results written incrementally to JSONL, stats aggregated at end

### Key Components

**`solib/Experiment.py`** - Orchestrates experiments across all protocol/model combinations. Generates configs from cartesian product of agents × judges × protocols × parameters.

**`solib/datatypes.py`** - Core data structures:
- `Question`: Contains question text and answer cases
- `Answer`: Answer choice with short label, text, and value
- `Prob`: Probability value (0-1) with arithmetic operators
- `Score`: Multi-metric scoring (log, brier, accuracy)
- `TranscriptItem`: Dialogue entry in debates

**`solib/protocols/abstract.py`** - Base classes:
- `Protocol`: Defines `run()` and `step()` methods
- `Judge`: Evaluates answers, assigns probabilities
- `QA_Agent`: LLM wrapper for generating arguments

**`solib/protocols/protocols/`** - Protocol implementations:
- `Blind`: Baseline - judge evaluates without AI assistance
- `Propaganda`: Single agent argues one side
- `Debate`: Two agents argue opposite sides (configurable turns, simultaneous mode)
- `Consultancy`: Consultant-client dialogue pattern

**`solib/protocols/judges/`** - Judge implementations:
- `TipOfTongueJudge`: Uses token probabilities
- `JustAskProbabilityJudge`: Explicitly asks for probability per answer
- `JustAskProbabilitiesJudge`: Asks for all probabilities at once

**`solib/utils/llm_utils.py`** - LLM integration:
- `acompletion_ratelimited()`: Async LLM calls with rate limiting & caching
- Supports multiple providers via LiteLLM (OpenAI, Anthropic, DeepSeek, HuggingFace)

### Data Flow
```
Dataset → Experiment.experiment() → Protocol.run() → Judge.__call__()
                                  ↓
                           QA_Agent.__call__() (generates arguments)
                                  ↓
                           results.jsonl + stats.json
```

## Important Conventions

### Randomness
Always use `solib.utils.random(*args, **kwargs)` instead of `random.random()`. This ensures reproducibility by seeding based on function arguments.

### LLM Calls
All LLM calls should go through `get_llm_response()` from `solib.llm_utils` for proper caching and cost tracking.

### Environment Variables
Key settings in `.env`:
- `SIMULATE=True`: Run without API calls (for cost estimation)
- `CACHING=True`: Cache LLM responses to `.litellm_cache/`
- `MAX_CONCURRENT_QUERIES=5`: Rate limiting

### Logging
Use the logger (`logging.getLogger(__name__)`) rather than print statements. Logs go to `.logs/`.

## File Locations
- Experiment results: `experiments/results/`
- LLM cache: `.litellm_cache/`
- Cost tracking: `.costly/`
- Logs: `.logs/`
- Prompt templates: `solib/prompts/` (Jinja2)
