from typing import List, Tuple
import json
import os.path as osp
import os
import logging
from datasets import load_dataset
import requests
import zipfile
import glob
import re


from solib.datatypes import Question, Answer
from solib.utils import random

LOGGER = logging.getLogger(__name__)


def file_path():
    return osp.dirname(osp.abspath(__file__))

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
    def data(cls, user_seed=0):
        raise NotImplementedError

    def transform(self, data_item: dict, user_seed=0) -> Question:
        question, answer_correct, answer_incorrect = self.extract_info(data_item, user_seed=user_seed)
        return self.to_question(question, answer_correct, answer_incorrect, user_seed=user_seed)

    def from_json(self, path, user_seed=0) -> list[Question]:
        with open(path, "r") as file:
            data = json.load(file)
        self.set_questions(data, user_seed=user_seed)

    def set_questions(self, data: list[dict], user_seed: int):
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
    def data(cls, user_seed=0):
        dset = load_dataset("Idavidrein/gpqa", "gpqa_main")
        inst = cls()
        inst.set_questions(dset["train"], user_seed)
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
    def data(cls, user_seed=0):
        train_path = osp.join(file_path(), 'math', "train.json")
        inst = cls()
        inst.from_json(train_path, user_seed=user_seed)
        return inst


    @classmethod
    def test_data(cls, user_seed=0):
        test_path = osp.join(file_path(), 'math', "test.json")
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
        return dset.filter(lambda x: x['subject'] != 'business_ethics')

    @classmethod
    def data(cls, user_seed=0):
        dset = load_dataset("cais/mmlu", "all")['test']
        inst = cls()
        dset = inst.preprocess(dset)
        inst.set_questions(dset, user_seed)
        return inst


class PrOntoQA(Dataset):
    output_dir = osp.join(file_path(), 'prontoqa')
    output_path = osp.join(output_dir, "generated_ood_data.zip")
    url = "https://github.com/asaparov/prontoqa/raw/refs/heads/main/generated_ood_data.zip"
    consolidated_path = osp.join(output_dir, "consolidated.json")

    @staticmethod
    def download():
        os.makedirs(PrOntoQA.output_dir, exist_ok=True)

        # Download the file
        response = requests.get(PrOntoQA.url)
        with open(PrOntoQA.output_path, "wb") as file:
            file.write(response.content)

        # Unzip the file
        with zipfile.ZipFile(PrOntoQA.output_path, 'r') as zip_ref:
            zip_ref.extractall(PrOntoQA.output_dir)

    @staticmethod
    def consolidate_json_files():
        # Read all json files and append them to a list
        json_files = glob.glob(osp.join(PrOntoQA.output_dir, "*.json"))
        if not json_files:
            # Download the data if it does not exist
            PrOntoQA.download()
            json_files = glob.glob(osp.join(PrOntoQA.output_dir, "*.json"))

        data = []
        for file in json_files:
            with open(file, "r") as f:
                f_data = json.load(f).values()
                questions = [question for example in f_data
                                    for name,question in example.items()
                                        if name.startswith('in_context_example')]
                data.extend(questions)

        # Write the list to a single json file
        with open(PrOntoQA.consolidated_path, "w") as f:
            json.dump(data, f, indent=4)

    def negate(self, text: str) -> str:
        # First, replace "is not" with a temporary placeholder
        text = re.sub(r'\bis not\b', '__TEMP_IS_NOT__', text)
        # Then, replace "is" with "is not"
        text = re.sub(r'\bis\b', 'is not', text)
        # Finally, replace the placeholder with "is"
        text = re.sub(r'__TEMP_IS_NOT__', 'is', text)
        return text

    def extract_info(self, data_item: dict, user_seed=0) -> Tuple[str, str, str]:
        context = data_item['question']
        query = data_item['query'].split(":")[1].strip()  # strip out the Prove: prefix
        correct_answer = True
        if random(query, user_seed=user_seed).random() > 0.5:
            correct_answer = False
            query = self.negate(query)
        question = f"{context} \n\n True or False?: {query}"
        return question, str(correct_answer), str(not correct_answer)

    @classmethod
    def data(cls, user_seed=0):
        # if consolidated file does not exist, download and consolidate
        if not osp.exists(cls.consolidated_path):
            cls.consolidate_json_files()

        inst = cls()
        inst.from_json(cls.consolidated_path, user_seed=user_seed)
        return inst


class TruthfulQA(Dataset):
    def extract_info(self, data_item: dict, user_seed=0) -> Tuple[str]:
        question = data_item["question"]
        correct = random(data_item, user_seed=user_seed).choice(data_item['correct_answers'])
        incorrect = random(data_item, user_seed=user_seed+1).choice(data_item['incorrect_answers'])  # +1 to give different seed
        return question, correct, incorrect

    @classmethod
    def data(cls, user_seed=0):
        dset = load_dataset("truthfulqa/truthful_qa", "generation")['validation']
        inst = cls()
        inst.set_questions(dset, user_seed)
        return inst

