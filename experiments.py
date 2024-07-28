import dataclasses
import random
from collections import defaultdict
from typing import List

from tqdm import tqdm

from data import DatasetItem, load_cache, load_data, save_cache, save_to_json
from model_wrappers import (
    ClaudeWrapper,
    GPTWrapper,
    Llama2Wrapper,
    Llama3Wrapper,
    ModelWrapper,
)

LETTERS = ["A", "B"]


def run_debate(
    debater_one: ModelWrapper,
    debater_two: ModelWrapper,
    judge: ModelWrapper,
    dataset: List[DatasetItem],
    output_path: str,
    cache_path: str = None,
):
    results = []
    for dataset_item in tqdm(dataset):
        if cache_path:
            cache = load_cache(cache_path)
        else:
            cache = defaultdict(lambda: defaultdict(dict))

        # Randomize the a/b order of the proofs so that the judge doesn't learn to always pick the first one
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
        if cache and "a" in cache[debater_a.model_id][dataset_item.question]:
            response_a = cache[debater_a.model_id][dataset_item.question]["a"]
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
            cache[debater_a.model_id][dataset_item.question]["a"] = response_a
        if cache and "b" in cache[debater_b.model_id][dataset_item.question]:
            response_b = cache[debater_b.model_id][dataset_item.question]["b"]
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
            cache[debater_b.model_id][dataset_item.question]["b"] = response_b

        # Get judge confidence & naive-judge confidence
        judge_confidence = judge.get_judge_confidence(
            dataset_item,
            response_a,
            response_b,
            is_answer_a_correct,
            [correct_letter, incorrect_letter],
        )

        naive_judge_confidence = judge.get_judge_confidence(
            dataset_item,
            response_a,
            response_b,
            is_answer_a_correct,
            [correct_letter, incorrect_letter],
            is_judge_naive=True,
        )

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
        if cache_path:
            save_cache(cache, cache_path)


if __name__ == "__main__":
    train_data, test_data = load_data()
    # debater_one = Llama3Wrapper("llama3_8b", "meta-llama/Meta-Llama-3-8B-Instruct")
    debater_one = Llama2Wrapper("llama2_7b", "meta-llama/Llama-2-7b-chat-hf")
    # debater_one = Llama2Wrapper("llama2_13b", "meta-llama/Llama-2-13b-chat-hf")
    # debater_two = ClaudeWrapper("claude3_sonnet", "claude-3-sonnet-20240229")
    # debater_two = ClaudeWrapper("claude35_sonnet", "claude-3-5-sonnet-20240620")
    debater_two = GPTWrapper("gpt_4o", "gpt-4o-2024-05-13")
    # debater_two = GPTWrapper('gpt_35_turbo', 'gpt-3.5-turbo-0125')
    judge = GPTWrapper("gpt35_turbo", "gpt-3.5-turbo-0125")
    # judge = Llama3Wrapper("llama3_8b", "meta-llama/Meta-Llama-3-8B-Instruct")

    print(
        f"Running debate between {debater_one.model_id} and {debater_two.model_id}\nJudge: {judge.model_id}"
    )
    run_debate(
        debater_one,
        debater_two,
        judge,
        train_data,
        f"results/{debater_one.model_id}-{debater_two.model_id}-{judge.model_id}.json",
        cache_path=f"cache.json",
    )
