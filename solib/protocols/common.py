import json
import asyncio
from typing import Self, Any
from dataclasses import dataclass
import numpy as np
from solib.utils import config
from solib.llm_utils import (
    get_llm_response,
    get_llm_response_async,
    get_llm_probs,
    get_llm_probs_async,
    parallelized_call,
)
from solib.datatypes import Answer, Question
from solib.tool_use.tool_rendering import TOOL_RESULT_START_TAG, TOOL_RESULT_END_TAG


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

    async def __call__(self, transcript: Transcript, **kwargs) -> "Self.TranscriptItem":
        words_in_mouth = " I judge that the answer is:\n\n("
        probabilities = await get_llm_probs_async(
            prompt=self.prompt.format(transcript=transcript),
            model=self.model,
            return_probs_for=transcript.question.possible_answer_symbols,
            words_in_mouth=words_in_mouth,
            **kwargs,
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
        "from POSSIBLE ANSWERS, and nothing else. Please note that anything "
        f"that is included in {TOOL_RESULT_START_TAG}{TOOL_RESULT_END_TAG} "
        "tags can be guaranteed to be true.\n\n"
        "{transcript}\n\n"
    )


# ABC
class Protocol:
    """General class for a judge endowed with any (multi-agent) protocol for
    question-answering, e.g. debate, consultancy, blindjudge. Methods for things "
    "like accuracy and testing should be implemented here."""

    def __init__(self, judge: Judge):
        raise NotImplementedError

    async def run(self, question: Question, **kwargs) -> Transcript:
        raise NotImplementedError

    async def __call__(self, question: Question, **kwargs) -> dict[str, float]:
        transcript = await self.run(question, **kwargs)
        return transcript.judgement.probabilities

    async def score(
        self, question: Question, probabilities: list[float] = None, **kwargs
    ) -> float:
        if probabilities is None:
            transcript = await self.run(question, **kwargs)
            probabilities = transcript.judgement.probabilities
        return np.log(
            probabilities[question.possible_answers.index(question.correct_answer)]
        )

    async def test(
        self,
        questions: list[Question],
        verbose: bool = False,
        write: str = None,
        **kwargs,
    ):
        results = []

        async def process_question(question):
            transcript = await self.run(question, **kwargs)
            probabilities = transcript.judgement.probabilities
            score = await self.score(question, probabilities, **kwargs)
            result = {
                "transcript": transcript,
                "score": score,
            }
            if verbose:
                print(result)
            return result

        results = await parallelized_call(process_question, questions)

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
