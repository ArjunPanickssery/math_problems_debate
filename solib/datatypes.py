import json
import logging
import inspect
import warnings
import numpy as np
from functools import wraps
from typing import Any, Literal
from pydantic import BaseModel, field_validator, computed_field

logger = logging.getLogger(__name__)


class Prob(BaseModel):
    prob: float

    @field_validator("prob")
    @classmethod
    def validate_prob(cls, v):
        assert 0.0 <= v <= 1.0, "Probability must be between 0 and 1."
        return v

    def __float__(self):
        return self.prob

    def __add__(self, other):
        return float(self) + float(other)

    def __mul__(self, other):
        return float(self) * float(other)

    def __truediv__(self, other):
        return float(self) / float(other)

    def __sub__(self, other):
        return float(self) - float(other)

    def __neg__(self):
        return -float(self)

    def __radd__(self, other):
        return float(other) + float(self)

    def __rmul__(self, other):
        return float(other) * float(self)

    def __rtruediv__(self, other):
        return float(other) / float(self)

    def __rsub__(self, other):
        return float(other) - float(self)

    def __array__(self):
        return np.array(float(self))

    def __lt__(self, other):
        return float(self) < float(other)

    def __le__(self, other):
        return float(self) <= float(other)

    def __gt__(self, other):
        return float(self) > float(other)

    def __ge__(self, other):
        return float(self) >= float(other)

    def __eq__(self, other):
        return float(self) == float(other)


class Score(BaseModel):
    """
    Data class to store scores based on different metrics.
    """

    log: float | None = None
    logodds: float | None = None
    accuracy: float | None = None

    @classmethod
    def calc(cls, question: "Question", answer_case: "Answer") -> "Score" | None:
        """Score for answer_case based on probability assigned to it."""
        assert answer_case in question.answer_cases
        if not all(a.judge_prob is not None for a in question.answer_cases):
            return None
        if not question.is_normalized:
            warnings.warn("Calculating score based on unnormalized probabilities.")
        score_log = np.log(answer_case.judge_prob)
        score_logodds = np.log(answer_case.judge_prob) - np.log(
            question.neg(answer_case).judge_prob
        )
        score_accuracy = float(
            answer_case.judge_prob > question.neg(answer_case).judge_prob
        )
        return Score(log=score_log, logodds=score_logodds, accuracy=score_accuracy)

    # Helper method for pointwise operations
    def _operate(self, other, op):
        # Function to apply an operation between two values
        def apply(x, y):
            if x is None or y is None:
                return None
            return op(x, y)

        if isinstance(other, Score):
            return Score(
                log=apply(self.log, other.log),
                logodds=apply(self.logodds, other.logodds),
                accuracy=apply(self.accuracy, other.accuracy),
            )
        elif isinstance(other, (int, float)):  # Scalar operations
            return Score(
                log=apply(self.log, other),
                logodds=apply(self.logodds, other),
                accuracy=apply(self.accuracy, other),
            )
        else:
            return NotImplemented

    # Implement arithmetic magic methods
    def __add__(self, other):
        return self._operate(other, lambda x, y: x + y)

    def __sub__(self, other):
        return self._operate(other, lambda x, y: x - y)

    def __mul__(self, other):
        return self._operate(other, lambda x, y: x * y)

    def __truediv__(self, other):
        return self._operate(other, lambda x, y: x / y)

    def __floordiv__(self, other):
        return self._operate(other, lambda x, y: x // y)

    def __pow__(self, other):
        return self._operate(other, lambda x, y: x**y)

    # Implement reversed operators for commutative operations
    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return self.__truediv__(other)


class Answer(BaseModel):
    """
    A potential answer, meant to be included inside a Question. Can optionally contain
    a `value` (e.g. 1.0 for true, -1.0 for false), which hidden via the .censor() method
    (which in particular is used in .to_prompt()).

    Can also be used to store an elicited `judge_prob`.

    Can also be used to store `case_probs`, indicating the probabilities elicited from a
    judge for each answer_case in a question after an agent argues for this answer.
    """

    short: str
    long: Any
    value: float | None = None
    judge_prob: Prob | None = None
    case_probs: "Question" | None = None

    def censor(self) -> "Answer":
        return Answer(short=self.short, long=self.long)

    @property
    def is_censored(self) -> bool:
        return (
            self.value is None and self.judge_prob is None and self.case_probs is None
        )

    @property
    def is_grounded(self) -> bool:
        return self.value is not None

    @property
    def is_elicited(self) -> bool:
        return self.judge_prob is not None

    @property
    def is_argued(self) -> bool:
        return self.case_probs is not None

    def to_prompt(self):
        return f"({self.short}): {self.long}"

    @computed_field
    @property
    def agent_score(self) -> Score:
        """Score for this answer case according to self.case_probs."""
        return Score.calc(self.case_probs, self)


class Question(BaseModel):
    """Question.

    Arguments:
        question (str): The question to be answered.
        answer_cases (list[Answer]): A list of answers that may be argued for.
    """

    question: str
    answer_cases: list[Answer]

    def censor(self) -> "Question":
        return Question(
            question=self.question,
            answer_cases=[answer.censor() for answer in self.answer_cases],
        )

    @property
    def is_censored(self) -> bool:
        return all(answer.is_censored for answer in self.answer_cases)

    @property
    def is_grounded(self) -> bool:
        return all(answer.is_grounded for answer in self.answer_cases)

    @property
    def is_elicited(self) -> bool:
        return all(answer.is_elicited for answer in self.answer_cases)

    @property
    def is_argued(self) -> bool:
        return all(answer.is_argued for answer in self.answer_cases)

    def to_prompt(self):
        return (
            f"QUESTION: {self.question}\n"
            + "POSSIBLE ANSWERS:\n"
            + "\n".join(answer.to_prompt() for answer in self.answer_cases)
        )

    @property
    def answer_cases_short(self) -> list[str]:
        return [answer.short for answer in self.answer_cases]

    @property
    def answer_cases_dict(self) -> dict[str, Answer]:
        return {answer.short: answer for answer in self.answer_cases}

    def neg(self, answer: Answer) -> Answer:
        assert len(self.answer_cases) == 2
        assert answer in self.answer_cases
        ans_index = self.answer_cases.index(answer)
        return self.answer_cases[1 - ans_index]

    # following methods only applies for value not None

    @property
    def best_answer(self) -> Answer:
        return max(self.answer_cases, key=lambda x: x.value)

    @property
    def worst_answer(self) -> Answer:
        return min(self.answer_cases, key=lambda x: x.value)

    @property
    def true_answer(self) -> Answer:
        true_answers = [answer for answer in self.answer_cases if answer.value == 1.0]
        assert len(true_answers) == 1
        return true_answers[0]

    @property
    def false_answer(self) -> Answer:
        false_answers = [answer for answer in self.answer_cases if answer.value == -1.0]
        assert len(false_answers) == 1
        return false_answers[0]

    # following methods only apply for judge_prob not None

    @property
    def total_prob(self) -> float:
        return sum(answer.judge_prob for answer in self.answer_cases)

    @property
    def is_normalized(self) -> bool:
        return abs(self.total_prob - 1.0) < 1e-3

    def normalize_probs(self):
        total = self.total_prob
        return Question(
            question=self.question,
            answer_cases=[
                Answer(
                    short=answer.short,
                    long=answer.long,
                    judge_prob=answer.judge_prob / total,
                )
                for answer in self.answer_cases
            ],
        )

    @computed_field
    @property
    def judge_score(self) -> Score:
        """
        Score judge_probs based on ground truth
        """
        return Score.calc(self, self.true_answer)

    @computed_field
    @property
    def agent_score_diff(self) -> Score:
        """Advantage in answering for the truth."""
        return self.true_answer.agent_score - self.false_answer.agent_score


class BetterJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        elif isinstance(obj, BaseModel):
            return obj.model_dump()
        return super().default(obj)


def censor(*args_to_censor):
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get function signature to manage defaults
            sig = inspect.signature(func)
            bound_args = sig.bind_partial(*args, **kwargs)
            bound_args.apply_defaults()

            # Apply censor() to specified arguments
            for arg_name in args_to_censor:
                if arg_name in bound_args.arguments and hasattr(
                    bound_args.arguments[arg_name], "censor"
                ):
                    bound_args.arguments[arg_name] = bound_args.arguments[
                        arg_name
                    ].censor()

            # Call the original async function
            return await func(*bound_args.args, **bound_args.kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Get function signature to manage defaults
            sig = inspect.signature(func)
            bound_args = sig.bind_partial(*args, **kwargs)
            bound_args.apply_defaults()

            # Apply censor() to specified arguments
            for arg_name in args_to_censor:
                if arg_name in bound_args.arguments and hasattr(
                    bound_args.arguments[arg_name], "censor"
                ):
                    bound_args.arguments[arg_name] = bound_args.arguments[
                        arg_name
                    ].censor()

            # Call the original sync function
            return func(*bound_args.args, **bound_args.kwargs)

        # Determine if the function is async or not and choose the appropriate wrapper
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
