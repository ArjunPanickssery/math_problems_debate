import pytest
import asyncio
from copy import deepcopy
from solib.utils import *
from solib.llm_utils import *
from solib.datatypes import Answer, Question
from solib.protocols.abstract import Protocol, Judge
from solib.protocols.debate import Debater, SequentialDebate
from solib.protocols.consultancy import Consultant, Client, Consultancy
from solib.protocols.blind import BlindJudgement
from solib.protocols.variants.common import (
    COTJudge,
    JustAskProbabilityJudge,
    COTJustAskProbabilityJudge,
    RandomJudge,
    HumanJudge,
)
from solib.protocols.variants.debate import SimultaneousDebate
from solib.protocols.variants.consultancy import OpenConsultancy

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
debaters = [
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
sosetups = {
    "blind": {
        "protocols": [BlindJudgement],
        "kwargs": [{"judge": judge} for judge in judges],
    },
    "debate": {
        "protocols": [SequentialDebate, SimultaneousDebate],
        "kwargs": [
            {
                "judge": judge,
                "debater_1": debater,
                "debater_2": deepcopy(debater),
                "num_turns": num_turns,
            }
            for judge in judges
            for debater in debaters
            for num_turns in [2, 4]
        ],
    },
    "consultancy": {
        "protocols": [Consultancy, OpenConsultancy],
        "kwargs": [
            {
                "judge": judge,
                "client": client,
                "consultant": consultant,
                "num_turns": num_turns,
            }
            for judge in judges
            for client in clients
            for consultant in consultants
            for num_turns in [2, 4]
        ],
    },
}


def generate_param_sets(sosetups):
    params = []
    for protocol_type, item in sosetups.items():
        params_protocol_types = []
        for protocol in item["protocols"]:
            params_protocol = []
            for kwargs in item["kwargs"]:
                params_protocol.append((protocol, kwargs))
            print(
                f"\nNumber of parameter sets for {protocol_type} / {protocol}: "
                f"{len(params_protocol)}"
            )
            params_protocol_types.extend(params_protocol)
        print(
            f"\nTotal number of parameter sets for {protocol_type}: "
            f"{len(params_protocol_types)}"
        )
        params.extend(params_protocol_types)
    print(
        f"\nTotal number of parameter sets: {len(params)}"
    )
    return params


param_sets = generate_param_sets(sosetups)

ques = Question(
    question="Who will be the 2024 presidential winner?",
    possible_answers=[Answer("A", "Donald Trump"), Answer("B", "not Donald Trump")],
    correct_answer=Answer("A", "Donald Trump"),
)

@pytest.mark.asyncio
@pytest.mark.parametrize("protocol, kwargs", param_sets)
async def test_protocols(protocol, kwargs, request):
    models = [
        v.model
        for k, v in kwargs.items()
        if hasattr(v, "model") and v.model is not None
    ]
    judge = kwargs.get("judge", None)
    if any(
        model.startswith("hf:") for model in models
    ) and not request.config.getoption("--runhf", False):
        pytest.skip("skipping test with hf model, add --runhf option to run")
    if isinstance(
        judge, (COTJudge, JustAskProbabilityJudge)
    ) and not request.config.getoption("--runredundant", False):
        pytest.skip(
            "skipping redundant test with COTJudge and JustAskProbabilityJudge, "
            "add --runredundant option to run"
        )
    instance = protocol(**kwargs)
    assert isinstance(instance, Protocol)
    transcript = await instance.run(ques)
    assert transcript.question == ques
    # assert transcript.protocol == protocol # actually no because subclasses
    assert isinstance(transcript.transcript, list)
    num_turns = kwargs.get("num_turns", None)
    # assert num_turns is None or len(transcript.transcript) == num_turns # actually no because +1
    assert isinstance(transcript.judgement, judge.TranscriptItem)
    print(transcript)
