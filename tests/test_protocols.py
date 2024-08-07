import sys

sys.path.append("src")


import pytest
from src.utils import *
from src.llm_utils import *
from src.datatypes import Answer, Question
from src.protocols.common import Protocol, Judge
from src.protocols.debate import Debater, SequentialDebate
from src.protocols.consultancy import Consultant, Client, Consultancy
from src.protocols.blind import BlindJudgement
from src.protocols.variants.common import (
    COTJudge,
    JustAskProbabilityJudge,
    COTJustAskProbabilityJudge,
    RandomJudge,
    HumanJudge,
)
from src.protocols.variants.debate import SimultaneousDebate
from src.protocols.variants.consultancy import OpenConsultancy

judges = [
    RandomJudge(),
    HumanJudge(),
    Judge(model="gpt-4o-mini"),
    Judge(model="hf:meta-llama/Llama-2-7b-chat-hf"),
    COTJudge(model="gpt-4o-mini"),
    JustAskProbabilityJudge(model="gpt-4o-mini"),
    COTJustAskProbabilityJudge(model="gpt-4o-mini"),
    COTJudge(model="hf:meta-llama/Llama-2-7b-chat-hf"),
    JustAskProbabilityJudge(model="hf:meta-llama/Llama-2-7b-chat-hf"),
    COTJustAskProbabilityJudge(model="hf:meta-llama/Llama-2-7b-chat-hf"),
]
protocols = {
    "blind": [BlindJudgement],
    "debate": [SequentialDebate, SimultaneousDebate],
    "consultancy": [Consultancy, OpenConsultancy],
}
debaters = [
    Debater(model="gpt-4o-mini"),
    Debater(model="hf:meta-llama/Llama-2-7b-chat-hf"),
]
debater_2s = [
    Debater(model="gpt-4o-mini"),
    Debater(model="hf:meta-llama/Llama-2-7b-chat-hf"),
]
consultants = [
    Consultant(model="gpt-4o-mini"),
    Consultant(model="hf:meta-llama/Llama-2-7b-chat-hf"),
]
clients = [
    Client(model="gpt-4o-mini"),
    Client(model="hf:meta-llama/Llama-2-7b-chat-hf"),
]

ques = Question(
    question="Who will be the 2024 presidential winner?",
    possible_answers=[Answer("A", "Donald Trump"), Answer("B", "not Donald Trump")],
    correct_answer=Answer("A", "Donald Trump"),
)


@pytest.mark.parametrize(
    "protocol,  judge, debater_1, debater_2, num_turns",
    [
        (protocol, judge, debater, debater, num_turns)
        for protocol in protocols["debate"]
        for judge in judges
        for debater in debaters
        for num_turns in [2, 4]
    ],
)
def test_protocols_debate(protocol, judge, debater_1, debater_2, num_turns):
    if any(
        model.startswith("hf:")
        for model in [judge.model, debater_1.model, debater_2.model]
    ) and not pytest.config.getoption("--runslow"):
        pytest.skip("skipping test with hf model, add --runslow option to run")
    if any(
        isinstance(judge, (COTJudge, JustAskProbabilityJudge)) for judge in [judge]
    ) and not pytest.config.getoption("--runmore"):
        pytest.skip(
            "skipping redundant test with COTJudge and JustAskProbabilityJudge, "
            "add --runmore option to run"
        )

    sosetup = protocol(
        judge=judge, debater_1=debater_1, debater_2=debater_2, num_turns=num_turns
    )
    transcript = sosetup.run(ques)
    assert transcript.question == ques
    assert transcript.protocol == protocol
    assert isinstance(transcript.transcript, list)
    assert len(transcript.transcript) == num_turns
    assert isinstance(transcript.judgement, judge.TranscriptItem)
    print(transcript)
