import json
import logging
from dataclasses import dataclass
from typing import Any
from pathlib import Path
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)


@dataclass
class Answer:
    short: str
    long: Any

    def to_prompt(self):
        return f"({self.short}): {self.long}"

    def __eq__(self, other):
        return self.short == other.short and self.long == other.long

    def __hash__(self):
        return hash((self.short, self.long))

    def to_dict(self):
        return {"short": self.short, "long": self.long}


@dataclass
class Question_stripped:
    """Question, with answer values (e.g. truth) censored.

    Arguments:
        question (str): The question to be answered.
        answer_cases (list[str]): A list of answers that may be argued for.
    """

    question: str
    answer_cases: list[Answer]

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
        assert (
            len(self.answer_cases) == 2
        ), "question must have two answer cases to negate"
        found_neg = False
        for a in self.answer_cases:
            if a != answer:
                assert not found_neg, "both question.answer_cases different from answer"
                negated_answer = a
                found_neg = True
        assert found_neg, "both question.answer_cases equal to answer"
        return negated_answer

    def strip(self) -> "Question_stripped":
        return self

    def __eq__(self, other):
        return (
            self.question == other.question and self.answer_cases == other.answer_cases
        )

    def __hash__(self):
        return hash((self.question, tuple(self.answer_cases)))

    def to_dict(self):
        return {
            "question": self.question,
            "answer_cases": [answer.to_dict() for answer in self.answer_cases],
        }


class Question(Question_stripped):
    """
    A question with a set of possible answers that may be argued for.

    Arguments:
        question (str): The question to be answered.
        answer_cases_and_values (list[tuple[Answer, float]]): A list of answers that may be argued for, with
            their associated "values" (e.g. 1 for the true answer, -1 for the false answer).
            NOTE: the values should NEVER be shown to any AI, or bad things will happen. Only answer_cases
            may be shown to any AI.
    """

    def __init__(
        self, question: str, answer_cases_and_values: list[tuple[Answer, float]]
    ):
        self.question = question
        self.answer_cases_and_values = answer_cases_and_values

    @property
    def answer_cases(self) -> list[Answer]:
        return [a for a, _ in self.answer_cases_and_values]

    @property
    def values_by_answer(self) -> dict[Answer, float]:
        return {a: v for a, v in self.answer_cases_and_values}

    @property
    def answers_by_value(self) -> dict[float, Answer]:
        assert len(set(self.values_by_answer.values())) == len(
            self.values_by_answer
        ), "values_by_answer must have unique values to invert"
        return {v: a for a, v in self.values_by_answer.items()}

    @property
    def best_answer(self) -> Answer:
        return self.answers_by_value[max(self.values_by_answer)]

    @property
    def worst_answer(self) -> Answer:
        return self.answers_by_value[min(self.values_by_answer)]

    @property
    def true_answer(self) -> Answer:
        return self.answers_by_value[1.0]

    @property
    def false_answer(self) -> Answer:
        return self.answers_by_value[-1.0]

    def strip(self) -> Question_stripped:
        return Question_stripped(question=self.question, answer_cases=self.answer_cases)

    def __eq__(self, other):
        return (
            self.question == other.question
            and self.answer_cases_and_values == other.answer_cases_and_values
        )

    def __hash__(self):
        return hash((self.question, tuple(self.answer_cases_and_values)))

    def to_dict(self):
        return {
            "question": self.question,
            "answer_cases_and_values": [
                (a.to_dict(), v) for a, v in self.answer_cases_and_values
            ],
        }


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
        import numpy as np

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


class BetterJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        elif isinstance(obj, BaseModel):
            return obj.model_dump()
        return super().default(obj)
