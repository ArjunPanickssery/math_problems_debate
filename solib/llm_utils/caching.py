from functools import wraps
import io
import os
import cloudpickle
import inspect
from perscache import Cache
from perscache.cache import hash_it
from perscache.serializers import CloudPickleSerializer, Serializer
from typing import Callable, Iterable
from pydantic import BaseModel


CACHE_BREAKER = os.getenv("CACHE_BREAKER", 0)


class BetterCache(Cache):
    """A subclass of Cache that hashes types by their names."""

    @staticmethod
    def _get_hash(
        fn: Callable,
        args: tuple,
        kwargs: dict,
        serializer: Serializer,
        ignore: Iterable[str],
    ) -> str:
        from solib.globals import SIMULATE

        # Get the argument dictionary by binding the function signature with args and kwargs
        arg_dict = inspect.signature(fn).bind(*args, **kwargs).arguments

        # Remove ignored arguments from the argument dictionary
        if ignore is not None:
            arg_dict = {k: v for k, v in arg_dict.items() if k not in ignore}

        # Convert types in the argument dictionary to their names
        for key, value in arg_dict.items():
            if isinstance(value, type):
                arg_dict[key] = (
                    value.__name__
                )  # Use type name instead of the actual type object

        # Include global variables in the cache hash because Python handles default
        # variables a bit differently than you might expect
        arg_dict["simulate"] = SIMULATE  # Add to the hash key
        arg_dict["cache_breaker"] = CACHE_BREAKER

        # Hash the function source, serializer type, and the argument dictionary
        return hash_it(inspect.getsource(fn), type(serializer).__name__, arg_dict)


class SafeCloudPickleSerializer(CloudPickleSerializer):
    # https://github.com/pydantic/pydantic/issues/8232#issuecomment-2189431721
    @classmethod
    def dumps(cls, obj):
        model_namespaces = {}

        with io.BytesIO() as f:
            pickler = cloudpickle.CloudPickler(f)

            for ModelClass in BaseModel.__subclasses__():
                model_namespaces[ModelClass] = ModelClass.__pydantic_parent_namespace__
                ModelClass.__pydantic_parent_namespace__ = None

            try:
                pickler.dump(obj)
                return f.getvalue()
            finally:
                for ModelClass, namespace in model_namespaces.items():
                    ModelClass.__pydantic_parent_namespace__ = namespace


cache = BetterCache(serializer=SafeCloudPickleSerializer())


# HACK. I have no idea why this works but just manually adding 'self' to
# @cache(ignore=...) doesn't.
def method_cache(ignore=None):
    if ignore is None:
        ignore = []
    # Ensure 'self' is always ignored
    if "self" not in ignore:
        ignore = ["self"] + ignore

    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            return cache(ignore=ignore)(method)(self, *args, **kwargs)

        return wrapper

    return decorator