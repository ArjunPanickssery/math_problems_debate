import json
from dataclasses import dataclass
from typing import Any
from pathlib import Path
from pydantic import BaseModel, field_validator

@dataclass
class Answer:
    symbol: str
    value: Any

    def __str__(self):
        return f"{self.symbol}: {self.value}"


@dataclass
class Question:
    question: str
    possible_answers: list[Answer]
    correct_answer: Answer  # make sure this doesn't appear in __str__

    def __str__(self):
        return (
            f"QUESTION: {self.question}\n"
            + "POSSIBLE ANSWERS:\n"
            + "\n".join(str(answer) for answer in self.possible_answers)
        )

    @property
    def possible_answer_symbols(self) -> list[str]:
        return [answer.symbol for answer in self.possible_answers]

    @property
    def possible_answers_dict(self) -> dict[str, Answer]:
        return {answer.symbol: answer for answer in self.possible_answers}

class Prob(BaseModel):
    prob: float

    @field_validator("prob")
    @classmethod
    def validate_prob(cls, v):
        assert 0.0 <= v <= 1.0, "Probability must be between 0 and 1."
        return v