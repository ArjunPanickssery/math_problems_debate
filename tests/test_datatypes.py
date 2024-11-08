import pytest
from pydantic import ValidationError
from solib.datatypes import Prob, Score, Answer, Question, TranscriptItem


def test_prob():
    # Test valid probability creation
    p = Prob(prob=0.5)
    assert float(p) == 0.5

    # Test arithmetic operations
    p2 = Prob(prob=0.3)
    assert float(p + p2) == 0.8
    assert float(p * p2) == 0.15
    assert float(p / p2) == 0.5 / 0.3

    # Test comparison operations
    assert p > p2
    assert p2 < p
    assert Prob(prob=0.5) == p

    # Test padding
    p_small = Prob(prob=0.0001)
    padded = p_small.pad(eps=0.01)
    assert float(padded) == 0.01

    # Test invalid probability
    with pytest.raises(ValidationError):
        Prob(prob=1.5)
    with pytest.raises(ValidationError):
        Prob(prob=-0.5)


def test_score():
    # Test score creation
    s = Score(log=-1.0, logodds=0.5, accuracy=1.0)

    # Test arithmetic operations
    s2 = Score(log=-2.0, logodds=1.0, accuracy=0.0)
    s_sum = s + s2
    assert s_sum.log == -3.0
    assert s_sum.logodds == 1.5
    assert s_sum.accuracy == 1.0

    # Test scalar operations
    s_scaled = s * 2
    assert s_scaled.log == -2.0
    assert s_scaled.logodds == 1.0
    assert s_scaled.accuracy == 2.0

    # Test array operations
    scores = [s, s2]
    mean_score = Score.mean(scores)
    assert mean_score.log == -1.5
    assert mean_score.logodds == 0.75
    assert mean_score.accuracy == 0.5


def test_answer():
    # Test censored answer
    ans = Answer(short="A", long="Option A")
    assert ans.is_censored
    assert not ans.is_grounded
    assert not ans.is_elicited
    assert not ans.is_argued

    # Test grounded answer
    ans_grounded = Answer(short="A", long="Option A", value=1.0)
    assert ans_grounded.is_grounded

    # Test elicited answer
    ans_elicited = Answer(short="A", long="Option A", judge_prob=Prob(prob=0.7))
    assert ans_elicited.is_elicited

    # Test prompt representation
    assert ans.to_prompt() == "(A): Option A"


def test_question():
    # Create basic answers
    ans_a = Answer(short="A", long="Option A")
    ans_b = Answer(short="B", long="Option B")

    # Test basic question creation
    q = Question(question="Test question?", answer_cases=[ans_a, ans_b])
    assert q.is_censored
    assert not q.is_grounded
    assert not q.is_elicited

    # Test grounded question
    q_grounded = Question(
        question="Test question?",
        answer_cases=[
            Answer(short="A", long="Option A", value=1.0),
            Answer(short="B", long="Option B", value=-1.0),
        ],
    )
    assert q_grounded.is_grounded
    assert q_grounded.true_answer.short == "A"
    assert q_grounded.false_answer.short == "B"

    # Test elicited question
    q_elicited = Question(
        question="Test question?",
        answer_cases=[
            Answer(short="A", long="Option A", judge_prob=Prob(prob=0.7)),
            Answer(short="B", long="Option B", judge_prob=Prob(prob=0.3)),
        ],
    )
    assert q_elicited.is_elicited
    assert q_elicited.is_normalized
    assert q_elicited.total_prob == 1.0


def test_transcript_item():
    # Test transcript item creation
    item = TranscriptItem(
        role="client", content="Test message", metadata={"timestamp": "2024-01-01"}
    )
    assert item.role == "client"
    assert item.content == "Test message"
    assert item.metadata["timestamp"] == "2024-01-01"

    # Test question with transcript
    q = Question(
        question="Test question?",
        answer_cases=[
            Answer(short="A", long="Option A"),
            Answer(short="B", long="Option B"),
        ],
    )
    q.append(item)
    assert q.has_transcript
    assert len(q.transcript) == 1
    assert q.transcript[0].role == "client"
