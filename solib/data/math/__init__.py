import json
import os.path as osp

from solib.utils import random
from solib.datatypes import Question, Answer


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

    if random(data_item, **kwargs) > 0.5:
        correct_answer.symbol, incorrect_answer.symbol = "B", "A"

    return Question(
        question=question,
        possible_answers=[correct_answer, incorrect_answer],
        correct_answer=correct_answer,
    )

def load_data(path, **kwargs) -> list[Question]:
    with open(path, "r") as file:
        data = json.load(file)
    return [transform(item, **kwargs) for item in data]


def train_data(**kwargs) -> list[Question]:
    train_path = osp.join(osp.dirname(osp.abspath(__file__)), "train.json")
    return load_data(train_path, **kwargs)


def test_data(**kwargs) -> list[Question]:
    test_path = osp.join(osp.dirname(osp.abspath(__file__)), "test.json")
    return load_data(test_path, **kwargs)
