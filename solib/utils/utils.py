from pathlib import Path
from typing import Coroutine, Any
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
import string

from tqdm.asyncio import tqdm as atqdm

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


def rand_suffix(size: int = 10) -> str:
    return "".join(rnd.choices(string.ascii_letters, k=size))


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

class NestedJSONSerializer:
    """
    A serializer that can handle nested structures of:
    - Pydantic models
    - Basic Python types (int, float, str, bool, None)
    - Collections (dict, list, tuple, set)
    - Common Python types (datetime, date, Decimal, UUID, Enum)
    - Numpy arrays and numbers
    - Pathlib Path objects
    """
    
    @classmethod
    def serialize(cls, obj: Any) -> Any:
        """
        Serialize an object to a JSON-compatible format.
        
        Args:
            obj: Any Python object to serialize
            
        Returns:
            JSON-serializable object
            
        Raises:
            TypeError: If object cannot be serialized
        """
        import numpy as np
        from pathlib import Path
        from datetime import datetime, date
        from enum import Enum
        from decimal import Decimal
        from uuid import UUID
        from pydantic import BaseModel
        # Handle None
        if obj is None:
            return None
            
        # Handle basic types
        if isinstance(obj, (bool, int, float, str)):
            return obj
            
        # Handle Pydantic models
        if isinstance(obj, BaseModel):
            return cls.serialize(obj.model_dump())
            
        # Handle dicts
        if isinstance(obj, dict):
            return {cls.serialize(k): cls.serialize(v) for k, v in obj.items()}
            
        # Handle lists, tuples, and sets
        if isinstance(obj, (list, tuple, set)):
            return [cls.serialize(item) for item in obj]
            
        # Handle datetime and date
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
            
        # Handle Decimal
        if isinstance(obj, Decimal):
            return str(obj)
            
        # Handle UUID
        if isinstance(obj, UUID):
            return str(obj)
            
        # Handle Enum
        if isinstance(obj, Enum):
            return obj.value
            
        # Handle Path
        if isinstance(obj, Path):
            return str(obj)
            
        # Handle numpy types
        if isinstance(obj, np.ndarray):
            return cls.serialize(obj.tolist())
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
            
        # Try to convert to dict if object has __dict__
        if hasattr(obj, '__dict__'):
            return cls.serialize(obj.__dict__)
            
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def serialize_to_json(obj: Any, file_path: str | Path, indent: int = 4) -> None:
    """
    Serialize an object and save it to a JSON file.
    
    Args:
        obj: Object to serialize
        file_path: Path to save the JSON file
        indent: Number of spaces for indentation (default: 2)
        
    Raises:
        TypeError: If object cannot be serialized
        OSError: If file cannot be written
    """
    # Convert to Path object if string
    if isinstance(file_path, str):
        file_path = Path(file_path)
        
    # Create directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Serialize and write to file
    with open(file_path, 'w') as f:
        json.dump(NestedJSONSerializer.serialize(obj), f, indent=indent)


async def parallelized_call(
    func: Coroutine,
    data: list[any],
    max_concurrent_queries: int | None = None,
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
    return await atqdm.gather(*tasks, disable=not use_tqdm)


def parse_time_interval(interval_str):
    # Extract number and unit using regex
    match = re.match(r"(\d+)([smh])", interval_str)
    if not match:
        raise ValueError(
            f"Unexpected interval format: {interval_str}. Expected format: number followed by s, m, or h"
        )

    number, unit = match.groups()
    if unit not in ["s", "m", "h"]:
        raise ValueError(f"Unexpected time unit: {unit}. Expected s, m, or h")

    number = int(number)

    # Convert to seconds
    multipliers = {"s": 1, "m": 60, "h": 3600}

    return number * multipliers[unit]


def estimate_tokens(messages: list[dict[str, str]]) -> int:
    foo = len(str(messages)) // 4
    s = 0
    for m in messages:
        if m.get("content"):
            s += len(m["content"]) // 3
        if m.get("tool_calls"):
            s += 1000
    return s
