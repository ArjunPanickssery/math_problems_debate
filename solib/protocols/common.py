import json
from typing import Self, Any
from dataclasses import dataclass
import numpy as np
from solib.utils import config
from solib.llm_utils import get_llm_response
from solib.datatypes import Answer, Question


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
        return self.protocol.str_transcript(self)

    def __dict__(self):
        return {
            "question": self.question,
            "judgement": self.judgement,
            "transcript": self.transcript,
            "protocol": self.protocol.__name__,
        }


# ABC
class Judge:

    @dataclass
    class TranscriptItem:
        probabilities: dict[str, float]

        def __str__(self):
            return f"\n## JUDGE PROBABILITY ESTIMATES:\n{self.probabilities}"

    def __call__(self, transcript: Transcript, **kwargs) -> "Self.TranscriptItem":
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
        if prompt is not None:
            self.prompt = prompt
        self.model = model

    prompt = (
        "I want you to give me your most informed judgment on which answer "
        "is correct, based on any information I provide. Your response "
        "should simply be the option symbol (e.g. A, B) of one of the items "
        "from POSSIBLE ANSWERS, and nothing else.\n\n"
        "{transcript}\n\n"
    )


# ABC
class Protocol:
    """General class for a judge endowed with any (multi-agent) protocol for
    question-answering, e.g. debate, consultancy, blindjudge. Methods for things "
    "like accuracy and testing should be implemented here."""

    def __init__(self, judge: Judge):
        raise NotImplementedError

    def run(self, question: Question, **kwargs) -> Transcript:
        raise NotImplementedError

    def __call__(self, question: Question, **kwargs) -> dict[str, float]:
        transcript = self.run(question, **kwargs)
        return transcript.judgement.probabilities

    def score(
        self, question: Question, probabilities: list[float] = None, **kwargs
    ) -> float:
        if probabilities is None:
            probabilities = self(question, **kwargs)
        return np.log(
            probabilities[question.possible_answers.index(question.correct_answer)]
        )

    def test(
        self,
        questions: list[Question],
        verbose: bool = False,
        write: str = None,
        **kwargs,
    ):
        results = []
        for question in questions:
            transcript = self.run(question, **kwargs)
            probabilities = transcript.judgement.probabilities
            score = self.score(question, probabilities, **kwargs)
            result = {
                "transcript": transcript,
                "score": score,
            }
            if verbose:
                print(result)
            results.append(result)
        scores = [result["score"] for result in results]
        stats = {
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
        }
        if verbose:
            print(stats)
        if write:
            with open(write, "w") as f:
                json.dump(
                    {"results": results, "stats": stats, "config": config(self)}, f
                )

        return results, stats

    @classmethod
    def str_transcript(cls, transcript: Transcript) -> str:
        return (
            str(transcript.question)
            + cls.transcript_heading
            + "".join(str(item) for item in transcript.transcript)
            + (str(transcript.judgement) if transcript.judgement is not None else "")
        )
