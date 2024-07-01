import json
import os
from typing import TypedDict


class Answer(TypedDict):
    numeric: float
    proof: str


class DatasetItem(TypedDict):
    question: str
    answer_correct: Answer
    answer_incorrect: Answer


def save_to_json(dictionary, file_name):
    # Create directory if not present
    directory = os.path.dirname(file_name)
    if directory != "" and not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_name, "w") as f:
        json.dump(dictionary, f)


def load_from_json(file_name) -> dict:
    with open(file_name, "r") as f:
        return json.load(f)


def load_data():
    return load_from_json("data/train_data.json"), load_from_json("data/test_data.json")
