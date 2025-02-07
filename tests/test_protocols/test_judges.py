import pytest
from unittest.mock import AsyncMock
from solib.datatypes import Prob, Answer, Question
from solib.protocols.judges.JustAskProbabilityJudge import JustAskProbabilityJudge
from solib.protocols.judges.TipOfTongueJudge import TipOfTongueJudge



@pytest.fixture
def basic_question():
    return Question(
        question="Test question?",
        answer_cases=[
            Answer(short="A", long="Option A"),
            Answer(short="B", long="Option B"),
        ],
    )


@pytest.mark.asyncio
async def test_just_ask_probability_judge(basic_question):
    judge = JustAskProbabilityJudge(model="gpt-4o-mini")
    judge.get_response = AsyncMock(return_value=Prob(prob=0.7))

    result = await judge(
        question=basic_question,
        context="Test context",
    )

    assert result.is_elicited
    assert result.is_normalized
    assert result.total_prob == 1.0
    assert (
        result.answer_cases[0].judge_prob.prob + result.answer_cases[1].judge_prob.prob
        == 1.0
    )


@pytest.mark.asyncio
async def test_tip_of_tongue_judge(basic_question):
    judge = TipOfTongueJudge(model="gpt-4o-mini")
    judge.get_probs = AsyncMock(
        return_value={
            "A": 0.7,
            "B": 0.3,
        }
    )

    result = await judge(
        question=basic_question,
        context="Test context",
    )

    assert result.is_elicited
    assert result.is_normalized
    assert result.total_prob == 1.0
    assert result.answer_cases[0].judge_prob.prob == 0.7
    assert result.answer_cases[1].judge_prob.prob == 0.3
