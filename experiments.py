import dataclasses
import random
from collections import defaultdict
from typing import List

from tqdm import tqdm

from data import (
    DatasetItem,
    load_argument_cache,
    load_data,
    load_naive_judge_cache,
    save_argument_cache,
    save_naive_judge_cache,
    save_to_json,
)
from model_wrappers import (
    ClaudeWrapper,
    GPTWrapper,
    Llama2Wrapper,
    Llama3Wrapper,
    ModelWrapper,
)

LETTERS = ["A", "B"]


def run_consultancy(
    consultant: ModelWrapper,
    judge: ModelWrapper,
    dataset: List[DatasetItem],
    output_path: str,
    argument_cache_path: str = None,
    naive_judge_cache_path: str = None,
):
    results = []
    for dataset_item in tqdm(dataset):
        argument_cache = (
            load_argument_cache(argument_cache_path)
            if argument_cache_path
            else defaultdict(lambda: defaultdict(dict))
        )
        naive_judge_cache = (
            load_naive_judge_cache(naive_judge_cache_path)
            if naive_judge_cache_path
            else defaultdict(dict)
        )

        # Randomize the a/b order of the proofs
        proof_a, proof_b = (
            dataset_item.answer_correct.proof,
            dataset_item.answer_incorrect.proof,
        )
        correct_letter, incorrect_letter = LETTERS
        is_answer_a_correct = True

        if random.random() > 0.5:
            is_answer_a_correct = False
            proof_a, proof_b = proof_b, proof_a
            correct_letter, incorrect_letter = incorrect_letter, correct_letter

        # Randomize which debater gets which proof
        consultant_letter = LETTERS[0]
        consultant_answer_numeric = (
            dataset_item.answer_correct.numeric
            if is_answer_a_correct
            else dataset_item.answer_incorrect.numeric
        )
        if random.random() > 0.5:
            consultant_letter = LETTERS[1]
            consultant_answer_numeric = (
                dataset_item.answer_incorrect.numeric
                if is_answer_a_correct
                else dataset_item.answer_correct.numeric
            )

        # Get argument
        if (
            argument_cache
            and "a" in argument_cache[consultant.model_id][dataset_item.question]
        ):
            argument = argument_cache[consultant.model_id][dataset_item.question]["a"]
        else:
            argument = consultant.get_debater_argument(
                dataset_item.question,
                consultant_letter,
                consultant_answer_numeric,
                proof_a,
                proof_b,
            )

        # Get judge confidence & naive-judge confidence
        judge_confidence = judge.get_judge_confidence_for_consultancy(
            dataset_item,
            is_answer_a_correct,
            consultant_letter,
            argument,
            [correct_letter, incorrect_letter],
        )

        if (
            naive_judge_cache
            and dataset_item.question in naive_judge_cache[judge.model_id]
        ):
            naive_judge_confidence = naive_judge_cache[judge.model_id][
                dataset_item.question
            ]
        else:
            naive_judge_confidence = judge.get_naive_judge_confidence(
                dataset_item,
                is_answer_a_correct,
                [correct_letter, incorrect_letter],
            )
            naive_judge_cache[judge.model_id][
                dataset_item.question
            ] = naive_judge_confidence

        results.append(
            {
                "item": dataclasses.asdict(dataset_item),
                "correct_letter": "a" if is_answer_a_correct else "b",
                "consultant": consultant.model_id,
                "consultant_letter": consultant_letter.lower(),
                "argument": argument,
                "judge": judge.model_id,
                "naive_judge_confidence": naive_judge_confidence,
                "judge_confidence": judge_confidence,
            }
        )
        save_to_json(results, output_path)
        if argument_cache_path:
            save_argument_cache(argument_cache, argument_cache_path)
        if naive_judge_cache_path:
            save_naive_judge_cache(naive_judge_cache, naive_judge_cache_path)


def run_debate(
    debater_one: ModelWrapper,
    debater_two: ModelWrapper,
    judge: ModelWrapper,
    dataset: List[DatasetItem],
    output_path: str,
    argument_cache_path: str = None,
    naive_judge_cache_path: str = None,
):
    results = []
    for dataset_item in tqdm(dataset):
        argument_cache = (
            load_argument_cache(argument_cache_path)
            if argument_cache_path
            else defaultdict(lambda: defaultdict(dict))
        )
        naive_judge_cache = (
            load_naive_judge_cache(naive_judge_cache_path)
            if naive_judge_cache_path
            else defaultdict(dict)
        )

        # Randomize the a/b order of the proofs
        proof_a, proof_b = (
            dataset_item.answer_correct.proof,
            dataset_item.answer_incorrect.proof,
        )
        correct_letter, incorrect_letter = LETTERS
        is_answer_a_correct = True

        if random.random() > 0.5:
            is_answer_a_correct = False
            proof_a, proof_b = proof_b, proof_a
            correct_letter, incorrect_letter = incorrect_letter, correct_letter

        # Randomize which debater gets which proof
        debater_a, debater_b = debater_one, debater_two
        if random.random() > 0.5:
            debater_a, debater_b = debater_two, debater_one

        # Get arguments
        if (
            argument_cache
            and "a" in argument_cache[debater_a.model_id][dataset_item.question]
        ):
            response_a = argument_cache[debater_a.model_id][dataset_item.question]["a"]
        else:
            response_a = debater_a.get_debater_argument(
                dataset_item.question,
                LETTERS[0],
                (
                    dataset_item.answer_correct.numeric
                    if is_answer_a_correct
                    else dataset_item.answer_incorrect.numeric
                ),
                proof_a,
                proof_b,
            )
            argument_cache[debater_a.model_id][dataset_item.question]["a"] = response_a
        if (
            argument_cache
            and "b" in argument_cache[debater_b.model_id][dataset_item.question]
        ):
            response_b = argument_cache[debater_b.model_id][dataset_item.question]["b"]
        else:
            response_b = debater_b.get_debater_argument(
                dataset_item.question,
                LETTERS[1],
                (
                    dataset_item.answer_incorrect.numeric
                    if is_answer_a_correct
                    else dataset_item.answer_correct.numeric
                ),
                proof_a,
                proof_b,
            )
            argument_cache[debater_b.model_id][dataset_item.question]["b"] = response_b

        # Get judge confidence & naive-judge confidence
        judge_confidence = judge.get_judge_confidence_for_debate(
            dataset_item,
            response_a,
            response_b,
            is_answer_a_correct,
            [correct_letter, incorrect_letter],
        )

        if (
            naive_judge_cache
            and dataset_item.question in naive_judge_cache[judge.model_id]
        ):
            naive_judge_confidence = naive_judge_cache[judge.model_id][
                dataset_item.question
            ]
        else:
            naive_judge_confidence = judge.get_naive_judge_confidence(
                dataset_item,
                is_answer_a_correct,
                [correct_letter, incorrect_letter],
            )
            naive_judge_cache[judge.model_id][
                dataset_item.question
            ] = naive_judge_confidence

        results.append(
            {
                "item": dataclasses.asdict(dataset_item),
                "correct_letter": "a" if is_answer_a_correct else "b",
                "debater_a": debater_a.model_id,
                "debater_b": debater_b.model_id,
                "response_a": response_a,
                "response_b": response_b,
                "judge": judge.model_id,
                "naive_judge_confidence": naive_judge_confidence,
                "judge_confidence": judge_confidence,
            }
        )
        save_to_json(results, output_path)
        if argument_cache_path:
            save_argument_cache(argument_cache, argument_cache_path)
        if naive_judge_cache_path:
            save_naive_judge_cache(naive_judge_cache, naive_judge_cache_path)


def debate_script():
    train_data, test_data = load_data()
    # llama3_8b = Llama3Wrapper("llama3_8b", "meta-llama/Meta-Llama-3-8B-Instruct")
    # llama2_7b = Llama2Wrapper("llama2_7b", "meta-llama/Llama-2-7b-chat-hf")
    # llama2_13b = Llama2Wrapper("llama2_13b", "meta-llama/Llama-2-13b-chat-hf")
    claude3_sonnet = ClaudeWrapper("claude3_sonnet", "claude-3-sonnet-20240229")
    claude35_sonnet = ClaudeWrapper("claude35_sonnet", "claude-3-5-sonnet-20240620")
    gpt4o = GPTWrapper("gpt4o", "gpt-4o-2024-05-13")
    gpt35_turbo = GPTWrapper("gpt35_turbo", "gpt-3.5-turbo-0125")

    def run(debater_one, debater_two, judge):
        if debater_two.model_id < debater_one.model_id:
            debater_one, debater_two = debater_two, debater_one
        print(
            f"Running debate between {debater_one.model_id} and {debater_two.model_id}\nJudge: {judge.model_id}"
        )
        run_debate(
            debater_one,
            debater_two,
            judge,
            train_data,
            f"results_debate/{debater_one.model_id}-{debater_two.model_id}-{judge.model_id}.json",
            argument_cache_path=f"argument_cache.json",
            naive_judge_cache_path=f"naive_judge_cache.json",
        )

    fake_llama3_8b = GPTWrapper("llama3_8b", "fake")
    fake_llama2_7b = GPTWrapper("llama2_7b", "fake")
    fake_llama2_13b = GPTWrapper("llama2_13b", "fake")

    models = [
        claude35_sonnet,
        claude3_sonnet,
        gpt4o,
        gpt35_turbo,
        fake_llama3_8b,
        fake_llama2_7b,
        fake_llama2_13b,
    ]
    judge = gpt4o
    for i in range(7):
        debater = models[i]
        for opponent in models[i + 1 :]:
            run(debater, opponent, judge)


def consultancy_script():
    train_data, test_data = load_data()
    claude3_sonnet = ClaudeWrapper("claude3_sonnet", "claude-3-sonnet-20240229")
    claude35_sonnet = ClaudeWrapper("claude35_sonnet", "claude-3-5-sonnet-20240620")
    gpt_4o = GPTWrapper("gpt4o", "gpt-4o-2024-05-13")
    gpt_35_turbo = GPTWrapper("gpt35_turbo", "gpt-3.5-turbo-0125")
    # judge = GPTWrapper("gpt35_turbo", "gpt-3.5-turbo-0125")

    def run(consultant, judge):
        print(f"Running consultancy with {consultant.model_id} and {judge.model_id}")
        run_consultancy(
            consultant,
            judge,
            train_data,
            f"results_consultancy/{consultant.model_id}-{judge.model_id}.json",
            argument_cache_path=f"argument_cache.json",
            naive_judge_cache_path=f"naive_judge_cache.json",
        )

    llama3_8b = Llama3Wrapper("llama3_8b", "meta-llama/Meta-Llama-3-8B-Instruct")
    # llama2_7b = Llama2Wrapper("llama2_7b", "meta-llama/Llama-2-7b-chat-hf")
    # llama2_13b = Llama2Wrapper("llama2_13b", "meta-llama/Llama-2-13b-chat-hf")
    fake_llama = GPTWrapper("llama2_13b", "fake")
    run(fake_llama, llama3_8b)


if __name__ == "__main__":
    debate_script()
