from typing import List, Tuple
import json
import os.path as osp
import logging
from datasets import load_dataset

from solib.datatypes import Question, Answer
from solib.utils import random

LOGGER = logging.getLogger(__name__)

class Dataset:
    def __init__(self, correct_val=1.0, incorrect_val=-1.0):
        self.correct_val = correct_val
        self.incorrect_val = incorrect_val
        self.questions: List[Question] = []

    def to_question(self, question: str, answer_correct: str, answer_incorrect: str, user_seed=0) -> Question:
        if random(question, user_seed=user_seed).random() > 0.5:
            correct_answer = Answer(short="A", long=answer_correct, value=self.correct_val)
            incorrect_answer = Answer(short="B", long=answer_incorrect, value=self.incorrect_val)
        else:
            correct_answer = Answer(short="B", long=answer_correct, value=self.correct_val)
            incorrect_answer = Answer(short="A", long=answer_incorrect, value=self.incorrect_val)

        return Question(question=question, answer_cases=[correct_answer, incorrect_answer])

    def extract_info(self, data_item: dict, user_seed=0) -> Tuple[str, str, str]:
        """Should return a tuple of (question, correct_answer, incorrect_answer)"""
        raise NotImplementedError

    @classmethod
    def train_data(cls, user_seed=0):
        raise NotImplementedError

    def transform(self, data_item: dict, user_seed=0) -> Question:
        question, answer_correct, answer_incorrect = self.extract_info(data_item, user_seed=user_seed)
        return self.to_question(question, answer_correct, answer_incorrect, user_seed=user_seed)

    def from_json(self, path, user_seed=0) -> list[Question]:
        with open(path, "r") as file:
            data = json.load(file)
        self.questions = [self.transform(item, user_seed=user_seed) for item in data]

    def __getitem__(self, index):
        return self.questions[index]

    def __len__(self):
        return len(self.questions)

    def __iter__(self):
        return iter(self.questions)


class GPQA(Dataset):
    def extract_info(self, data_item: dict, user_seed=0) -> Tuple[str, str, str]:
        return (
            data_item["Question"],
            data_item["Correct Answer"],
            data_item[f"Incorrect Answer {random(data_item, user_seed=user_seed).randint(1, 3)}"]
            )

    @classmethod
    def train_data(cls, user_seed=0):
        dset = load_dataset("Idavidrein/gpqa", "gpqa_main")
        inst = cls()
        inst.questions = [inst.transform(item, user_seed=user_seed) for item in dset["train"]]
        return inst


class GSM8K(Dataset):
    def extract_info(self, data_item: dict, user_seed=0) -> Tuple[str, str, str]:
        question, answer_correct, answer_incorrect = (
            data_item["question"],
            data_item["answer_correct"],
            data_item["answer_incorrect"],
        )
        if not isinstance(answer_correct, str):
            answer_correct = f"{answer_correct['numeric']}\n{answer_correct['proof']}"
        if not isinstance(answer_incorrect, str):
            answer_incorrect = f"{answer_incorrect['numeric']}\n{answer_incorrect['proof']}"
        return question, answer_correct, answer_incorrect


    @classmethod
    def train_data(cls, user_seed=0):
        train_path = osp.join(osp.dirname(osp.abspath(__file__)), 'math', "train.json")
        inst = cls()
        inst.from_json(train_path, user_seed=user_seed)
        return inst


    @classmethod
    def test_data(cls, user_seed=0):
        test_path = osp.join(osp.dirname(osp.abspath(__file__)), 'math', "test.json")
        inst = cls()
        inst.from_json(test_path, user_seed=user_seed)
        return inst


class MMLU(Dataset):
    def extract_info(self, data_item: dict, user_seed=0) -> Tuple[str]:
        choices = data_item["choices"]
        wrong = list(range(len(choices)))
        del wrong[data_item["answer"]]  # remove correct answer
        incorrect = random(data_item, user_seed=user_seed).choice(wrong)
        return data_item["question"], choices[data_item["answer"]], choices[incorrect]

    def preprocess(self, dset):
        return dset.filter(lambda x: x['subjet'] != 'business_ethics')

    @classmethod
    def train_data(cls, user_seed=0):
        dset = load_dataset("cais/mmlu", "all")['test']
        inst = cls()
        dset = inst.preprocess(dset)
        inst.questions = [inst.transform(item, user_seed=user_seed) for item in dset["train"]]
        return inst
