from typing import Self, Any
from dataclasses import dataclass
import numpy as np
from src.llm_utils import get_llm_response


@dataclass
class Answer:
    symbol: str
    value: Any

    def __str__(self):
        return f"{self.symbol}: {self.value}"


@dataclass
class Question:
    question: str
    possible_answers: list[Answer]
    correct_answer: Answer  # make sure this doesn't appear in __str__

    def __str__(self):
        return (
            f"QUESTION: {self.question}\n"
            + "POSSIBLE ANSWERS:\n"
            + "\n".join(str(answer) for answer in self.possible_answers)
        )

    @property
    def possible_answer_symbols(self) -> list[str]:
        return [answer.symbol for answer in self.possible_answers]


class Transcript:
    """A transcript of any possible conversation-based protocol for a task,
    like debate, consulting or (the degenerate case of) a blind judge. Designed
    for question-answering tasks, but could potentially be extended."""

    def __init__(self, question: Question, protocol=None):
        self.question = question
        self.judgement = None
        self.transcript = []
        self.protocol = protocol

    def append(self, item: Any):
        if isinstance(item, Judge.TranscriptItem):
            self.judgement = item
        self.transcript.append(item)

    def __str__(self):
        if self.protocol is None:
            return str(self.question) + (
                str(self.judgement) if self.judgement is not None else ""
            )
        else:
            return self.protocol.str_transcript(self)


# ABC
class Judge:

    @dataclass
    class TranscriptItem:
        probabilities: dict[str, float]

        def __str__(self):
            return f"\n## JUDGE PROBABILITY ESTIMATES:\n{self.probabilities}"

    def __call__(self, transcript: Transcript) -> "Self.TranscriptItem":
        words_in_mouth = " I judge that the answer is:\n\n("
        probabilities = get_llm_response(
            prompt=self.prompt.format(transcript=transcript),
            model=self.model,
            return_probs_for=transcript.question.possible_answer_symbols,
            max_tokens=max(
                len(answer_symbol)
                for answer_symbol in transcript.question.possible_answer_symbols
            ),
            words_in_mouth=words_in_mouth,
        )
        return self.TranscriptItem(probabilities=probabilities)

    def __init__(self, prompt: str = None, model: str = None):
        """Initialize basic / default Judge. Can be overriden in subclasses.

        Args:
            prompt (str): prompt for the judge. Default None.
            model (str): model for the judge. Default None.
        """
        if prompt is None:
            prompt = (
                "I want you to give me your most informed judgment on which answer "
                "is correct, based on any information I provide. Your response "
                "should simply be the option symbol (e.g. A, B) of one of the items "
                "from POSSIBLE ANSWERS, and nothing else.\n\n"
                "{transcript}\n\n"
            )
        self.prompt = prompt
        self.model = model


# ABC
class Protocol:
    """General class for a judge endowed with any (multi-agent) protocol for
    question-answering, e.g. debate, consultancy, blindjudge. Methods for things "
    "like accuracy and testing should be implemented here."""

    def __init__(self, judge: Judge):
        raise NotImplementedError

    def run(self, question: Question) -> Transcript:
        raise NotImplementedError

    def __call__(self, question: Question) -> dict[str, float]:
        transcript = self.run(question)
        return transcript.judgement.probabilities

    def score(self, question: Question) -> float:
        return np.log(
            self(question)[question.possible_answers.index(question.correct_answer)]
        )

    @classmethod
    def subtranscript(cls, transcript: Transcript) -> str:
        transcript_item_classes = [
            participant_class.TranscriptItem
            for participant_class in cls.participant_classes
        ]
        items = [
            item
            for item in transcript.transcript
            if any(isinstance(item, cls) for cls in transcript_item_classes)
        ]
        return items

    @classmethod
    def str_transcript(cls, transcript: Transcript) -> str:
        return (
            str(transcript.question)
            + cls.transcript_heading
            + "".join(str(item) for item in cls.subtranscript(transcript))
            + (str(transcript.judgement) if transcript.judgement is not None else "")
        )

class BlindJudge(Protocol):
    """General class for BlindJudge.
    - Judge asks questions, debater responds.
    - Ends when Judge has asked num_turns questions.
    """

    def __init__(
        self,
        judge: Judge,
    ):
        self.judge = judge

    def run(self, question: Question) -> Transcript:
        transcript = Transcript(question)
        judge_item = self.judge(transcript)
        transcript.append(judge_item)
        return transcript