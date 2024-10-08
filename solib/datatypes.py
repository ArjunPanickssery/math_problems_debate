import json
from dataclasses import dataclass
from typing import Any
from pathlib import Path
from pydantic import BaseModel, field_validator


@dataclass(frozen=True)
class Answer:
    short: str
    long: Any

    def to_prompt(self):
        return f"({self.short}): {self.long}"


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class Question(Question_stripped):
    """
    A question with a set of possible answers that may be argued for.

    Arguments:
        question (str): The question to be answered.
        answer_cases (dict[Answer, float]): A dictionary of answers that may be argued for, with
            their associated "values" (e.g. 1 for the true answer, -1 for the false answer).
            NOTE: self.answer_cases.values() should NEVER be shown to any AI, or bad things will happen.
    """

    question: str
    answer_cases: dict[Answer, float]

    @property
    def answers_by_value(self) -> dict[float, Answer]:
        assert len(set(self.answer_cases.values())) == len(
            self.answer_cases
        ), "answer_cases must have unique values to invert"
        return {v: a for a, v in self.answer_cases.items()}

    @property
    def best_answer(self) -> Answer:
        return self.answers_by_value[max(self.answer_cases)]

    @property
    def worst_answer(self) -> Answer:
        return self.answers_by_value[min(self.answer_cases)]

    @property
    def true_answer(self) -> Answer:
        return self.answers_by_value[1.0]

    @property
    def false_answer(self) -> Answer:
        return self.answers_by_value[-1.0]

    def strip(self) -> Question_stripped:
        return Question_stripped(
            question=self.question, answer_cases=self.answer_cases.keys()
        )


class Prob(BaseModel):
    prob: float

    @field_validator("prob")
    @classmethod
    def validate_prob(cls, v):
        assert 0.0 <= v <= 1.0, "Probability must be between 0 and 1."
        return v
