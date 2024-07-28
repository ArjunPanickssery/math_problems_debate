from collections import defaultdict
import dataclasses
import json
import os
from dataclasses import dataclass
from typing import List, TypedDict


@dataclass
class Answer:
    numeric: float
    proof: str


@dataclass
class DatasetItem:
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


def transform_to_dataset_item(data: List[dict]) -> List[DatasetItem]:
    """Transforms a list of dicts into a list of DatasetItems"""
    return [
        DatasetItem(
            question=item["question"],
            answer_correct=Answer(
                proof=item["answer_correct"]["proof"],
                numeric=item["answer_correct"]["numeric"],
            ),
            answer_incorrect=Answer(
                proof=item["answer_incorrect"]["proof"],
                numeric=item["answer_incorrect"]["numeric"],
            ),
        )
        for item in data
        if type(item["answer_incorrect"]) == dict
    ]


def load_data() -> tuple[List[DatasetItem], List[DatasetItem]]:
    train_data_raw = load_from_json("data/train_data.json")
    test_data_raw = load_from_json("data/test_data.json")
    train_data = transform_to_dataset_item(train_data_raw)
    test_data = transform_to_dataset_item(test_data_raw)
    return train_data, test_data

def load_cache(cache_path: str):
    with open(cache_path, 'r') as f:
        cache = json.load(f)
    # Convert nested dictionaries to defaultdict
    return defaultdict(lambda: defaultdict(dict), {
        k: defaultdict(dict, v) for k, v in cache.items()
    })

def save_cache(cache, cache_path: str):
    with open(cache_path, 'w') as f:
        # Convert defaultdict to regular dict for JSON serialization
        json.dump({k: dict(v) for k, v in cache.items()}, f, indent=4)