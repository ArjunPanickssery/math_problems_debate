import json
from solib.utils import seed
from solib.datatypes import Question, Answer
import random


def transform(data_item: dict, **kwargs) -> Question:
    question, answer_correct, answer_incorrect = (
        data_item["question"],
        data_item["answer_correct"],
        data_item["answer_incorrect"],
    )
    if not isinstance(answer_correct, str):
        answer_correct = f"{answer_correct['numeric']}\n{answer_correct['proof']}"
    if not isinstance(answer_incorrect, str):
        answer_incorrect = f"{answer_incorrect['numeric']}\n{answer_incorrect['proof']}"

    correct_answer = Answer(symbol="A", value=answer_correct)
    incorrect_answer = Answer(symbol="B", value=answer_incorrect)

    seed(data_item, **kwargs)
    if random.random() > 0.5:
        correct_answer.symbol, incorrect_answer.symbol = "B", "A"

    return Question(
        question=question,
        possible_answers=[correct_answer, incorrect_answer],
        correct_answer=correct_answer,
    )


def train_data(**kwargs) -> list[Question]:
    with open("src/data/math/train.json", "r") as file:
        data = json.load(file)
    return [transform(item, **kwargs) for item in data]


def test_data(**seed_kwargs) -> list[Question]:
    with open("src/data/math/test.json", "r") as file:
        data = json.load(file)
    return [transform(item, **seed_kwargs) for item in data]
