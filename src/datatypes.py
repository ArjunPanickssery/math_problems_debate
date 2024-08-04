import json
from dataclasses import dataclass
from typing import Any
from pathlib import Path

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

def load_data(data_path: Path) -> list[Question]:
    with open(data_path, "r") as file:
        data = json.load(file)
    