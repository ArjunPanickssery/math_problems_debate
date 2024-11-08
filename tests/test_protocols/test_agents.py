import pytest
from unittest.mock import AsyncMock, patch
from solib.datatypes import Question, Answer, Prob
from solib.protocols.agents.BestOfN_Agent import BestOfN_Agent
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
def mock_qa_agent():
    agent = QA_Agent(model="gpt-4o-mini")
    agent.get_response = AsyncMock(return_value="Test response")
    return agent


@pytest.fixture
def mock_judge():
    judge = JustAskProbabilityJudge(model="gpt-4o-mini")
    judge.get_response = AsyncMock(return_value=Prob(prob=0.7))
    return judge


@pytest.mark.asyncio
async def test_qa_agent(mock_qa_agent, basic_question):
    response = await mock_qa_agent(
        question=basic_question,
        answer_case=basic_question.answer_cases[0],
    )
    assert response == "Test response"
    mock_qa_agent.get_response.assert_called_once()


@pytest.mark.asyncio
async def test_best_of_n_agent(mock_qa_agent, mock_judge, basic_question):
    with patch(
        "solib.protocols.agents.BestOfN_Agent.parallelized_call", new_callable=AsyncMock
    ) as mock_parallel:
        # Mock the parallel calls to return a list of (response, score) tuples
        mock_parallel.return_value = [
            ("Response 1", -1.0),
            ("Response 2", -0.5),  # This should be chosen as best
            ("Response 3", -2.0),
        ]

        bon_agent = BestOfN_Agent(
            n=3,
            agent=mock_qa_agent,
            judge=mock_judge,
        )

        response = await bon_agent(
            question=basic_question,
            answer_case=basic_question.answer_cases[0],
        )

        assert response == "Response 2"
        mock_parallel.assert_called_once()
