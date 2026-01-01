import asyncio
import functools
import logging
import time
from litellm import Message
import requests
from traceback import format_exc
from typing import Callable
import json
from datetime import datetime
from pathlib import Path

import tiktoken

from solib.utils.globals import OPENROUTER_API_KEY, ENABLE_PROMPT_HISTORY, RATE_LIMITER_FRAC_RATE_LIMIT
from solib.utils.rate_limits.rate_limit_utils import DEFAULT_RATES
from solib.utils.utils import parse_time_interval, rand_suffix

LOGGER = logging.getLogger(__name__)


class Resource:
    """
    A resource that is consumed over time and replenished at a constant rate.
    """

    def __init__(self, refresh_rate, total=0, throughput=0):
        self.refresh_rate = refresh_rate
        self.total = total
        self.throughput = throughput
        self.last_update_time = time.time()
        self.start_time = time.time()
        self.value = self.refresh_rate

    def _replenish(self):
        """
        Updates the value of the resource based on the time since the last update.
        """
        curr_time = time.time()
        self.value = min(
            self.refresh_rate,
            self.value + (curr_time - self.last_update_time) * self.refresh_rate / 60,
        )
        self.last_update_time = curr_time
        self.throughput = self.total / (curr_time - self.start_time) * 60

    def geq(self, amount: float) -> bool:
        self._replenish()
        return self.value >= amount

    def consume(self, amount: float):
        """
        Consumes the given amount of the resource.
        """
        assert self.geq(
            amount
        ), f"Resource does not have enough capacity to consume {amount} units"
        self.value -= amount
        self.total += amount


class RateLimiter:
    def __init__(
        self,
        frac_rate_limit: float = 0.9,
        prompt_dir: str = "prompt_dir",
        enable_prompt_history: bool = False,
    ):
        self.frac_rate_limit = frac_rate_limit
        self.model_ids = set()

        self.token_capacity = dict()
        self.request_capacity = dict()
        self.lock_add = asyncio.Lock()
        self.lock_consume = asyncio.Lock()
        if enable_prompt_history:
            self.prompt_dir = prompt_dir
        else:
            self.prompt_dir = None
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def update_openrouter_ratelimit(self, model_id: str):
        try:
            resp = requests.get(
                "https://openrouter.ai/api/v1/auth/key",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            )
            resp_json = resp.json()
            # Handle different API response formats
            if "data" in resp_json and "rate_limit" in resp_json["data"]:
                resp_ = resp_json["data"]["rate_limit"]
                rl = resp_["requests"] / parse_time_interval(resp_["interval"])
            else:
                # Fallback to reasonable defaults if API format changed
                print(f"OpenRouter rate limit API format changed, using defaults. Response: {resp_json}")
                rl = 60  # 60 requests per second as fallback
            tl = 1e5  # meh
            return tl, rl
        except Exception as e:
            print(f"Error getting OpenRouter rate limit: {e}, using defaults")
            return 1e5, 60  # fallback defaults

    async def add_model_id(self, model_id: str):
        if model_id in self.model_ids:
            return

        self.model_ids.add(model_id)

        # make dummy request to get token and request capacity
        if model_id.startswith("openrouter/"):
            token_capacity, request_capacity = self.update_openrouter_ratelimit(
                model_id
            )
        elif model_id.startswith("ollama"):
            token_capacity = 1e20  # arbitrary large
            request_capacity = 80  # don't make this too big
        else:
            amts = DEFAULT_RATES.get(model_id)
            token_capacity = amts["tpm"]
            request_capacity = amts["rpm"]

        print(
            f"got capacities for model {model_id}: {token_capacity}, {request_capacity}"
        )
        token_cap = token_capacity * self.frac_rate_limit
        request_cap = request_capacity * self.frac_rate_limit

        token_capacity = Resource(token_cap)
        request_capacity = Resource(request_cap)
        token_capacity.consume(token_cap)
        request_capacity.consume(request_cap)
        self.token_capacity[model_id] = token_capacity
        self.request_capacity[model_id] = request_capacity

    @staticmethod
    def _count_prompt_token_capacity(messages: list[dict[str, str]], **kwargs) -> int:
        # The magic formula is: .25 * (total number of characters) + (number of messages) + (max_tokens, or 15 if not specified)
        BUFFER = 5  # A bit of buffer for some error margin
        MIN_NUM_TOKENS = 20

        num_tokens = 0
        for message in messages:
            num_tokens += 1
            if message.get("content"):
                num_tokens += len(message["content"]) / 4
            else:  # tool call
                num_tokens += 2000

        return max(
            MIN_NUM_TOKENS,
            int(num_tokens + BUFFER)
            + kwargs.get("n", 1) * kwargs.get("max_tokens", 15),
        )

    async def call(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        max_attempts: int,
        call_function: functools.partial,
        is_valid=lambda x: True,
        write: Path | str | None = None,
        **kwargs,
    ):
        uniq_id = rand_suffix()
        async def attempt_api_call():
            while True:
                async with self.lock_consume:
                    request_capacity, token_capacity = (
                        self.request_capacity[model_id],
                        self.token_capacity[model_id],
                    )
                    if request_capacity.geq(1) and token_capacity.geq(token_count):
                        request_capacity.consume(1)
                        token_capacity.consume(token_count)
                    else:
                        await asyncio.sleep(0.01)
                        continue  # Skip this iteration if the condition isn't met

                # Make the API call outside the lock
                return await asyncio.wait_for(
                    call_function(model_id, messages, **kwargs),
                    timeout=222,  # this needs to be longer for ollama
                )

        async with self.lock_add:
            await self.add_model_id(model_id)

        token_count = self._count_prompt_token_capacity(messages, **kwargs)

        response = None
        for i in range(max_attempts):
            try:
                response = await attempt_api_call()

                if not is_valid(response):
                    raise RuntimeError(
                        f"Invalid response according to is_valid {response}"
                    )
            except Exception as e:
                error_info = f"Exception Type: {type(e).__name__}, Error Details: {str(e)}, Traceback: {format_exc()}"
                LOGGER.warn(
                    f"Encountered API error: {error_info}.\nRetrying now. (Attempt {i})"
                )
                await asyncio.sleep(1.5**i)
            else:
                break

        if response is None:
            raise RuntimeError(
                f"Failed to get a response from the API after {max_attempts} attempts."
            )

        if ENABLE_PROMPT_HISTORY and write is not None:

            prompt_dir = Path(write) / "prompt_history"
            prompt_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{model_id.replace('/', '_')}_{timestamp}_{rand_suffix()}.json"

            history_entry = {
                "timestamp": timestamp,
                "model": model_id,
                "messages": [
                    m.model_dump() if hasattr(m, "model_dump") else m
                    for m in messages
                ],
                "response": (
                    response.model_dump()
                    if hasattr(response, "model_dump")
                    else response
                ),
                "completion_kwargs": str(call_function.keywords | kwargs),
            }
            try:
                with open(prompt_dir / filename, "w") as f:
                    json.dump(history_entry, f, indent=2)
            except Exception as e:
                LOGGER.warn(
                    f"Failed to write prompt history to {prompt_dir / filename}: {e}"
                )

        return response


RATE_LIMITER = (
    RateLimiter(frac_rate_limit=RATE_LIMITER_FRAC_RATE_LIMIT)
)  # initialize here instead of in solib.utils.globals to avoid circular import