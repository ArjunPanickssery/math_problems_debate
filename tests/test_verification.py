"""Test verification scaffolding."""
import asyncio
import os

# Enable verification for testing
os.environ["VERIFY_ALIGNMENT"] = "True"
os.environ["VERIFY_ALIGNMENT_N_TRIES"] = "3"
os.environ["VERIFY_ALIGNMENT_MODEL"] = "gpt-4o-mini"

from solib.datatypes import Question, Answer
from solib.protocols.abstract import QA_Agent
from solib.protocols.judges import JustAskProbabilityJudge
from solib.protocols.protocols.Propaganda import Propaganda


async def main():
    # Create a simple test question
    question = Question(
        question="What is 7 + 5?",
        answer_cases=[
            Answer(short="A", long="12", value=1.0),
            Answer(short="B", long="13", value=-1.0),
        ],
    )

    agent = QA_Agent(model="gpt-4o-mini")
    judge = JustAskProbabilityJudge(model="gpt-4o-mini")
    protocol = Propaganda()

    print("=" * 60)
    print("Testing Propaganda with verification enabled")
    print("=" * 60)

    # Test arguing for the correct answer (A = 12)
    print("\nTest 1: Arguing for correct answer (A = 12)")
    result_a = await protocol.run(
        agent=agent,
        question=question,
        answer_case=question.answer_cases[0],  # A = 12
        judge=judge,
    )

    print(f"\nTranscript item role: {result_a.transcript[0].role}")
    print(f"Transcript item content: {result_a.transcript[0].content[:200]}...")
    print(f"Verification metadata: {result_a.transcript[0].metadata}")

    # Test arguing for the wrong answer (B = 13)
    print("\n" + "=" * 60)
    print("Test 2: Arguing for wrong answer (B = 13)")
    result_b = await protocol.run(
        agent=agent,
        question=question,
        answer_case=question.answer_cases[1],  # B = 13
        judge=judge,
    )

    print(f"\nTranscript item role: {result_b.transcript[0].role}")
    print(f"Transcript item content: {result_b.transcript[0].content[:200]}...")
    print(f"Verification metadata: {result_b.transcript[0].metadata}")

    print("\n" + "=" * 60)
    print("Verification test complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
