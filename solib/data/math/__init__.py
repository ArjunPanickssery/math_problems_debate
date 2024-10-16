import json
import os.path as osp
import logging
from solib.utils import random
from solib.datatypes import Question, Answer

logger = logging.getLogger(__name__)

# NOTE: **kwargs are passed everywhere to allow the random seed to be broken (see project README)


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

    if random(data_item, **kwargs) > 0.5:
        correct_answer = Answer(short="A", long=answer_correct)
        incorrect_answer = Answer(short="B", long=answer_incorrect)
    else:
        correct_answer = Answer(short="B", long=answer_incorrect)
        incorrect_answer = Answer(short="A", long=answer_correct)

    return Question(
        question=question,
        answer_cases={correct_answer: 1.0, incorrect_answer: -1.0},
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
