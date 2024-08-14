import numpy as np
from contextlib import contextmanager
from warnings import warn
from typing import Callable


class CostItem:

    # input_tokens: dollar per input token
    # output_tokens: dollar per output token
    # time: seconds per output token
    prices = {
        "gpt-4o": {
            "input_tokens": 5.0e-6,
            "output_tokens": 15.0e-6,
            "time": 18e-3,
        },
        "gpt-4o-mini": {
            "input_tokens": 0.15e-6,
            "output_tokens": 0.6e-6,
            "time": 9e-3,
        },
        "claude-3-5-sonnet": {
            "input_tokens": 3.0e-6,
            "output_tokens": 15.0e-6,
            "time": 18e-3,
        },
        "claude-3-opus": {
            "input_tokens": 15.0e-6,
            "output_tokens": 75.0e-6,
            "time": 18e-3,
        },
        "claude-3-haiku": {
            "input_tokens": 0.25e-6,
            "output_tokens": 1.25e-6,
            "time": 9e-3,
        },
    }

    def __init__(
        self,
        cost_range: list[float] = None,
        time_range: list[float] = None,
        model: str = None,
        input_tokens: int = None,
        output_tokens_range: list[int] = None,
        input_string: str = None,
        description: str = None,
        input_tokens_estimator: Callable[[str], int] = None,
        output_tokens_estimator: Callable[[str, int], list[int]] = None,
    ):
        self.cost_range = cost_range
        self.time_range = time_range
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens_range = output_tokens_range
        self.input_string = input_string
        self.description = description
        if input_tokens_estimator is None:
            input_tokens_estimator = lambda input_string: int(len(input_string) // 4.5)
        if output_tokens_estimator is None:
            output_tokens_estimator = lambda input_string, input_tokens: [1, 2048]
        self.input_tokens_estimator = input_tokens_estimator
        self.output_tokens_estimator = output_tokens_estimator
        self.calc()

    def calc(self):
        if self.cost_range is not None and self.time_range is not None:
            return
        if self.model is None or self.model not in self.prices:
            warn(
                "Model {self.model} is None or its price is not known; "
                "ignoring LLM call {self}."
            )
            return
        if self.input_tokens is None:
            if self.input_string is None:
                warn("Input tokens and string are None; ignoring LLM call {self}.")
                return
            self.input_tokens = self.input_tokens_estimator(self.input_string)
        if self.output_tokens_range is None:
            self.output_tokens_range = self.output_tokens_estimator(
                self.input_string, self.input_tokens
            )
        input_cost = self.input_tokens * self.prices[self.model]["input_tokens"]
        output_cost_range = [
            self.output_tokens * self.prices[self.model]["output_tokens"]
            for self.output_tokens in self.output_tokens_range
        ]
        if self.cost_range is None:
            self.cost_range = [
                input_cost + output_cost for output_cost in output_cost_range
            ]
        if self.time_range is None:
            self.time_range = self.time_range or [
                self.output_tokens * self.prices[self.model]["time"]
                for self.output_tokens in self.output_tokens_range
            ]
        if self.cost_range is None:
            print('BROOO???')
        if self.time_range is None:
            print('GAAHHHH???')
    
    def __repr__(self):
        return (
            f"CostItem("
            f"cost_range={self.cost_range}, "
            f"time_range={self.time_range}, "
            f"model={self.model}, "
            f"input_tokens={self.input_tokens}, "
            f"output_tokens_range={self.output_tokens_range}, "
            f"input_string={self.input_string}, "
            f"description={self.description}"
            ")"
        )


class CostEstimator:

    def __init__(self):
        self.log:list[CostItem] = []
        self.num_calls = 0
        self.cost_range = [0.0, 0.0]
        self.time_range = [0.0, 0.0]

    @contextmanager
    def append(self, item: CostItem):
        item.calc()
        if item.cost_range is None:
            print('HUHHHH???')
        self.log.append(item)
        self.num_calls += 1
        self.cost_range[0] += item.cost_range[0]
        self.cost_range[1] += item.cost_range[1]
        self.time_range[0] += item.time_range[0]
        self.time_range[1] += item.time_range[1]
        yield

    def __repr__(self):
        return (
            f"Total cost: {self.cost_range[0]:.2f} - {self.cost_range[1]:.2f} USD\n"
            f"Total time: {self.time_range[0]:.2f} - {self.time_range[1]:.2f} sec\n"
            f"Total calls: {self.num_calls}\n"
            f"Breakdown:\n {self.log}"
        )