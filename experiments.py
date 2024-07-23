import dataclasses
import random
from typing import List

from tqdm import tqdm

from data import DatasetItem, load_data, save_to_json
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
):
    results = []
    for dataset_item in tqdm(dataset):
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

        # Get judge confidence & blind-judge confidence
        judge_confidence = judge.get_judge_confidence(
            dataset_item,
            response_a,
            response_b,
            is_answer_a_correct,
            [correct_letter, incorrect_letter],
        )

        blind_judge_confidence = judge.get_judge_confidence(
            dataset_item,
            None,
            None,
            is_answer_a_correct,
            [correct_letter, incorrect_letter],
            is_judge_blind=True,
        )

        results.append(
            {
                "item": dataclasses.asdict(dataset_item),
                "a_debater": debater_a.model_id,
                "b_debater": debater_b.model_id,
                debater_a.model_id: response_a,
                debater_b.model_id: response_b,
                "blind_judge_confidence": blind_judge_confidence,
                "judge_confidence": judge_confidence,
            }
        )
        save_to_json(results, output_path)


if __name__ == "__main__":
    train_data, test_data = load_data()
    debater_one = debater_two = judge = GPTWrapper("gpt-4o", "gpt-4o-2024-05-13")
    # judge = Llama3Wrapper("llama3_8b", "meta-llama/Meta-Llama-3-8B-Instruct")
    run_debate(
        debater_one,
        debater_two,
        judge,
        train_data,
        f"results/{debater_one.model_id}-{debater_two.model_id}-{judge.model_id}.json",
    )
