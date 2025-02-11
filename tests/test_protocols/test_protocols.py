import pytest
from unittest.mock import AsyncMock
from solib.datatypes import Prob, Answer, Question
from solib.protocols.protocols.Blind import Blind
from solib.protocols.protocols.Consultancy import Consultancy
from solib.protocols.protocols.Propaganda import Propaganda
from solib.protocols.protocols.Debate import Debate
from solib.protocols.abstract import QA_Agent
from solib.protocols.judges import JustAskProbabilityJudge


@pytest.fixture
def basic_question():
    return Question(
        question="Test question?",
        answer_cases=[
            Answer(short="A", long="Option A"),
            Answer(short="B", long="Option B"),
        ],
    )


@pytest.fixture
def mock_agent():
    agent = QA_Agent(model="gpt-4o-mini")
    agent.get_response = AsyncMock(return_value="Test response")
    return agent


@pytest.fixture
def mock_judge():
    judge = JustAskProbabilityJudge(model="gpt-4o-mini")

    async def mock_call(question, context):
        return Question(
            question=question.question,
            answer_cases=[
                Answer(
                    short=a.short,
                    long=a.long,
                    judge_prob=Prob(prob=0.7 if i == 0 else 0.3),
                )
                for i, a in enumerate(question.answer_cases)
            ],
            transcript=question.transcript,
        )

    judge.__call__ = mock_call
    return judge


@pytest.mark.asyncio
async def test_blind_protocol(basic_question, mock_agent, mock_judge):
    protocol = Blind()
    result = await protocol.run(
        agent=mock_agent,
        question=basic_question,
        answer_case=basic_question.answer_cases[0],
        judge=mock_judge,
    )

    assert result.is_elicited
    assert result.is_normalized


@pytest.mark.asyncio
async def test_propaganda_protocol(basic_question, mock_agent, mock_judge):
    protocol = Propaganda()
    result = await protocol.run(
        agent=mock_agent,
        question=basic_question,
        answer_case=basic_question.answer_cases[0],
        judge=mock_judge,
    )

    assert result.is_elicited
    assert result.is_normalized
    assert result.has_transcript
    assert len(result.transcript) == 1
    assert result.transcript[0].role == basic_question.answer_cases[0].short


@pytest.mark.asyncio
async def test_consultancy_protocol(basic_question, mock_agent, mock_judge):
    protocol = Consultancy(num_turns=2)
    result = await protocol.run(
        agent=mock_agent,
        question=basic_question,
        answer_case=basic_question.answer_cases[0],
        judge=mock_judge,
    )

    assert result.is_elicited
    assert result.is_normalized
    assert result.has_transcript
    assert len(result.transcript) == 2


@pytest.mark.asyncio
async def test_debate_protocol(basic_question, mock_agent, mock_judge):
    protocol = Debate(num_turns=2)
    result = await protocol.run(
        agent=mock_agent,
        question=basic_question,
        answer_case=basic_question.answer_cases[0],
        judge=mock_judge,
        adversary=mock_agent,  # Using same agent as adversary for simplicity
    )

    assert result.is_elicited
    assert result.is_normalized
    assert result.has_transcript
    assert len(result.transcript) == 2
    # Test that both debaters spoke
    assert result.transcript[0].role == basic_question.answer_cases[0].short
    assert result.transcript[1].role == basic_question.answer_cases[1].short
