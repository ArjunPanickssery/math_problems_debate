import json
import os.path as osp
from pathlib import Path
import asyncio
from typing import Dict, List, Optional
import jinja2
from solib.data.loading import GSM8K
from solib.datatypes import Question
from solib.utils.llm_utils import acompletion_ratelimited
from tqdm import tqdm

# Set up Jinja environment
env = jinja2.Environment(
    loader=jinja2.FileSystemLoader("solib/prompts/data_generation")
)


def load_prompt_template(template_name: str) -> str:
    """Load a prompt template from the prompts directory."""
    return env.get_template(template_name).render()


def extract_numeric_answer(solution: str) -> Optional[float]:
    """Extract numeric answer from a solution string that ends with #### {number}."""
    hash_idx = solution.find("####")
    if hash_idx == -1:
        return None

    try:
        numeric_str = solution[hash_idx:].split("####")[1].strip()
        return float(numeric_str.replace(",", ""))
    except (ValueError, IndexError):
        return None


async def generate_incorrect_solution(
    question: str,
    solution: str,
    model: str = "claude-3-5-sonnet-20241022",
    max_retries: int = 30,
) -> Optional[Dict[str, str]]:
    """Generate an incorrect solution for a given math problem."""

    # Load prompt templates
    system_prompt = load_prompt_template("gsm8k_incorrect_system.jinja")
    user_template = env.get_template("gsm8k_incorrect_user.jinja")

    # Render user prompt
    user_prompt = user_template.render(question=question, solution=solution)

    # Build messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(max_retries):
        try:
            # Make request to Claude with increased max_tokens using ratelimited version
            response = await acompletion_ratelimited(
                model=model, messages=messages, max_tokens=5000
            )

            # Extract response content
            content = response.choices[0].message.content

            # Try to parse alternate solution using rfind to get the last instance
            start_tag = "<alternate_solution>"
            end_tag = "</alternate_solution>"

            start_idx = content.rfind(start_tag)  # Changed from find to rfind
            end_idx = content.rfind(end_tag)  # Changed from find to rfind

            if start_idx == -1 or end_idx == -1:
                print(
                    f"Warning: Could not parse solution on attempt {attempt + 1}. Retrying..."
                )
                if attempt == max_retries - 1:
                    print(f"Failed question: {question}")
                continue

            incorrect_solution = content[start_idx + len(start_tag) : end_idx].strip()

            # Extract numeric answer
            numeric_answer = extract_numeric_answer(incorrect_solution)
            if numeric_answer is None:
                print(
                    f"Warning: Could not parse numeric answer on attempt {attempt + 1}. Retrying..."
                )
                if attempt == max_retries - 1:
                    print(f"Failed question: {question}")
                continue

            return {
                "incorrect_solution": incorrect_solution,
                "transcript": content,
                "numeric": numeric_answer,
            }

        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                print(f"Failed to generate solution after {max_retries} attempts")
                print(f"Failed question: {question}")
                return None


async def process_single_problem(problem: Question) -> Optional[Dict]:
    """Process a single problem and return the result."""
    # Extract question and solution
    question = problem.question
    solution = problem.best_answer.long

    # Extract numeric answer from correct solution
    correct_numeric = extract_numeric_answer(solution)
    if correct_numeric is None:
        print(
            f"Warning: Could not parse numeric answer from correct solution: {solution}"
        )
        return None

    # Generate incorrect solution
    result = await generate_incorrect_solution(question, solution)

    if result:
        new_problem = {}
        new_problem["question"] = question
        new_problem["answer_correct"] = {
            "numeric": correct_numeric,
            "proof": solution,
        }
        new_problem["answer_incorrect"] = {
            "numeric": result["numeric"],
            "proof": result["incorrect_solution"],
        }
        new_problem["transcript"] = result["transcript"]
        return new_problem
    return None


async def process_dataset(problems: List[Question], output_path: str) -> List[Dict]:
    """Process a list of problems in parallel and save results to output file."""
    # Create tasks for all problems
    tasks = [process_single_problem(problem) for problem in problems]

    # Process all problems in parallel with progress bar
    results = []
    for coro in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc="Generating incorrect solutions",
    ):
        result = await coro
        if result:
            results.append(result)

    return results


async def main():
    # Load original datasets
    train_data = GSM8K.data().questions
    test_data = GSM8K.test_data().questions

    # Create output directory if it doesn't exist
    Path("solib/data/math").mkdir(parents=True, exist_ok=True)

    # Define output paths
    train_output = osp.join("solib/data/math", "train_expanded.json")
    train_output_full = osp.join("solib/data/math", "train_expanded_full.json")
    test_output = osp.join("solib/data/math", "test_expanded.json")
    test_output_full = osp.join("solib/data/math", "test_expanded_full.json")

    # Load existing expanded datasets if they exist
    existing_train = []
    existing_test = []
    if osp.exists(train_output_full):
        with open(train_output_full, "r") as f:
            existing_train = json.load(f)
    if osp.exists(test_output_full):
        with open(test_output_full, "r") as f:
            existing_test = json.load(f)

    # Create sets of existing questions
    existing_train_qs = {q["question"] for q in existing_train}
    existing_test_qs = {q["question"] for q in existing_test}

    # Filter out questions that already have solutions
    remaining_train = [q for q in train_data if q.question not in existing_train_qs]
    remaining_test = [q for q in test_data if q.question not in existing_test_qs]

    print(f"Found {len(existing_train)} existing train solutions")
    print(f"Found {len(existing_test)} existing test solutions")
    print(f"Remaining train questions to process: {len(remaining_train)}")
    print(f"Remaining test questions to process: {len(remaining_test)}")

    input("Press Enter to continue...")
    if not remaining_train and not remaining_test:
        print("No new questions to process!")
        return

    # Process remaining questions
    train_results = []
    test_results = []
    if remaining_train:
        print("\nProcessing remaining training questions...")
        train_results = await process_dataset(remaining_train, train_output)
    if remaining_test:
        print("\nProcessing remaining test questions...")
        test_results = await process_dataset(remaining_test, test_output)

    # Function to create compact version without transcripts
    def create_compact_version(data):
        return [{k: v for k, v in item.items() if k != "transcript"} for item in data]

    # Combine with existing results and save both versions
    if remaining_train:
        # Full version with transcripts
        all_train_results = existing_train + train_results
        print(f"\nSaving {len(all_train_results)} total train results")
        with open(train_output_full, "w") as f:
            json.dump(all_train_results, f, indent=2)

        # Compact version without transcripts
        compact_train_results = create_compact_version(all_train_results)
        with open(train_output, "w") as f:
            json.dump(compact_train_results, f, indent=2)

    if remaining_test:
        # Full version with transcripts
        all_test_results = existing_test + test_results
        print(f"\nSaving {len(all_test_results)} total test results")
        with open(test_output_full, "w") as f:
            json.dump(all_test_results, f, indent=2)

        # Compact version without transcripts
        compact_test_results = create_compact_version(all_test_results)
        with open(test_output, "w") as f:
            json.dump(compact_test_results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
