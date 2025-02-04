from pathlib import Path
from typing import Coroutine
from pydantic import BaseModel
import logging
import json
import jsonlines
import aiofiles
import re
import random as rnd
import hashlib
import inspect
import os
import asyncio
from tqdm.asyncio import tqdm
from costly.estimators.llm_api_estimation import LLM_API_Estimation

LOGGER = logging.getLogger(__name__)


def dump_config(instance):
    if inspect.isfunction(instance):
        return instance.__name__
    elif isinstance(instance, (int, float, str, bool, type(None))):
        return instance
    elif isinstance(instance, list):
        return [dump_config(item) for item in instance]
    elif isinstance(instance, dict):
        return {k: dump_config(v) for k, v in instance.items()}
    elif isinstance(instance, tuple):
        return tuple(dump_config(item) for item in instance)
    elif isinstance(instance, set):
        return set(dump_config(item) for item in instance)
    elif isinstance(instance, BaseModel):
        return instance.model_dump()
    elif isinstance(instance, type):
        return instance.__name__
    elif hasattr(instance, "__class__") and hasattr(instance.__class__, "__name__"):
        result = {}
        result["__class__"] = instance.__class__.__name__
        if hasattr(instance, "dict"):
            result["dict"] = {k: dump_config(v) for k, v in instance.dict.items()}
        elif hasattr(instance, "__dict__"):
            result["__dict__"] = {
                k: dump_config(v) for k, v in instance.__dict__.items()
            }
    else:
        raise TypeError(f"Unsupported type: {type(instance)}")
    return result


def seed(*args, user_seed=0):
    input_str = "".join(map(str, args))
    hash_object = hashlib.md5(input_str.encode())
    seed = int(hash_object.hexdigest(), 16) % 2**32
    rnd.seed(seed + user_seed)


class random:
    def __init__(self, *args, user_seed=0):
        seed(*args, user_seed)

    def __getattr__(self, name):
        return getattr(rnd, name, None)


def update_recursive(source, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict) and key in source and isinstance(source[key], dict):
            update_recursive(source[key], value)
        else:
            source[key] = value
    return source


def update_nonnones(source, overrides):
    for key, value in overrides.items():
        if value is not None:
            source[key] = value
    return source


def coerce(val, coercer: callable):
    if val is None:
        return None
    return coercer(val)

AbstractionError = NotImplementedError("Must be implemented in subclass.")

class DefaultDict(dict):
    def __init__(self, key_fn):
        super().__init__()
        self._key_fn = key_fn

    def __missing__(self, key):
        v = self._key_fn(key)
        self[key] = v
        return v


def write_jsonl(
    data: dict | list[dict] | str | list[str] | BaseModel | list[BaseModel],
    path: Path | str | None = None,
    append: bool = True,
):
    if path:
        if not isinstance(data, list):
            data = [data]
        with jsonlines.open(path, mode="a" if append else "w") as writer:
            for item in data:
                if isinstance(item, BaseModel):
                    item = item.model_dump()
                elif isinstance(item, str):
                    item = json.loads(item)
                writer.write(item)
    else:
        pass


def write_json(data: dict | list | str | BaseModel, path: Path | str | None = None):
    if path:
        with open(path, "w") as f:
            if isinstance(data, BaseModel):
                data = data.model_dump()
            elif isinstance(data, str):
                data = json.loads(data)
            json.dump(data, f)
    else:
        pass


async def write_jsonl_async(
    data: dict | list[dict] | str | list[str] | BaseModel | list[BaseModel],
    path: Path | str | None = None,
    append: bool = True,
):
    if path:
        mode = "a" if append else "w"
        if not isinstance(data, list):
            data = [data]
        async with aiofiles.open(path, mode=mode) as f:
            for item in data:
                if isinstance(item, BaseModel):
                    item = item.model_dump()
                elif isinstance(item, str):
                    item = json.loads(item)
                await f.write(json.dumps(item) + "\n")
    else:
        pass


async def write_json_async(
    data: dict | list | str | BaseModel, path: Path | str | None = None
):
    if path:
        async with aiofiles.open(path, "w") as f:
            if isinstance(data, BaseModel):
                data = data.model_dump()
            elif isinstance(data, str):
                data = json.loads(data)
            await f.write(json.dumps(data))
    else:
        pass


def str_config(config: dict | list[dict]):
    if isinstance(config, list):
        return "\n".join(str_config(item) for item in config)
    else:
        return str(
            {
                k: (
                    v.__name__
                    if isinstance(v, type)
                    else str_config(v) if isinstance(v, dict) else v
                )
                for k, v in config.items()
            }
        )


async def parallelized_call(
    func: Coroutine,
    data: list[any],
    max_concurrent_queries: int = None,
    use_tqdm: bool = False,
) -> list[any]:
    """
    Run async func in parallel on the given data.
    func will usually be a partial which uses query_api or whatever in some way.

    Example usage:
        partial_eval_method = functools.partial(eval_method, model=model, **kwargs)
        results = await parallelized_call(partial_eval_method, [format_post(d) for d in data])
    """

    if os.getenv("SINGLE_THREAD"):
        LOGGER.info(f"Running {func} on {len(data)} datapoints sequentially")
        return [await func(d) for d in data]

    # max_concurrent_queries = min(
    #     max_concurrent_queries,
    #     int(os.getenv("MAX_CONCURRENT_QUERIES", max_concurrent_queries)),
    # )

    LOGGER.info(
        f"Running {func} on {len(data)} datapoints with {max_concurrent_queries} concurrent queries"
    )

    if max_concurrent_queries is not None:
        local_semaphore = asyncio.Semaphore(max_concurrent_queries)

        async def call_func(func, datapoint):
            async with local_semaphore:
                return await func(datapoint)

        tasks = [call_func(func, d) for d in data]
    else:
        tasks = [func(d) for d in data]
    return await tqdm.gather(*tasks, disable=not use_tqdm)

def parse_time_interval(interval_str):
    # Extract number and unit using regex
    match = re.match(r'(\d+)([smh])', interval_str)
    if not match:
        raise ValueError(f"Unexpected interval format: {interval_str}. Expected format: number followed by s, m, or h")
    
    number, unit = match.groups()
    if unit not in ['s', 'm', 'h']:
        raise ValueError(f"Unexpected time unit: {unit}. Expected s, m, or h")
    
    number = int(number)
    
    # Convert to seconds
    multipliers = {
        's': 1,
        'm': 60,
        'h': 3600
    }
    
    return number * multipliers[unit]

estimate_tokens: callable = LLM_API_Estimation.messages_to_input_tokens