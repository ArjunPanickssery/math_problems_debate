from pathlib import Path
from pydantic import BaseModel
import json
import jsonlines
import aiofiles
import random as rnd
import hashlib


def config(instance):
    result = {}
    if hasattr(instance, "__class__") and hasattr(instance.__class__, "__name__"):
        result["class"] = instance.__class__.__name__
    for key, value in instance.__dict__.items():
        if hasattr(value, "__dict__"):
            result[key] = config(value)
        elif isinstance(value, list):
            result[key] = [
                config(item) if hasattr(item, "__dict__") else item for item in value
            ]
        elif isinstance(value, dict):
            result[key] = {
                k: config(v) if hasattr(v, "__dict__") else v for k, v in value.items()
            }
        else:
            result[key] = value
    return result


def seed(*args, **kwargs):
    user_seed = kwargs.get("user_seed", 0)
    input_str = "".join(map(str, args))
    hash_object = hashlib.md5(input_str.encode())
    seed = int(hash_object.hexdigest(), 16) % 2**32
    rnd.seed(seed + user_seed)


def random(*args, **kwargs):
    seed(*args, **kwargs)
    return rnd.random()


def apply(func, *args, **kwargs):
    """Apply a function tolerant to extra parameters"""
    import inspect

    allowed_params = inspect.signature(func).parameters.keys()
    filtered_params = {k: v for k, v in kwargs.items() if k in allowed_params}
    return func(*args, **filtered_params)


async def apply_async(func, *args, **kwargs):
    """Apply a function tolerant to extra parameters (async version)"""
    import inspect

    allowed_params = inspect.signature(func).parameters.keys()
    filtered_params = {k: v for k, v in kwargs.items() if k in allowed_params}
    return await func(*args, **filtered_params)


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


AbstractionError = NotImplementedError("Must be implemented in subclass.")


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
