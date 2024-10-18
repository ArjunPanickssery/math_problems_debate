import json
from pathlib import Path
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


AbstractionError = NotImplementedError("Must be implemented in subclass.")

