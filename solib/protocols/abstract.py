import json
import asyncio
from functools import partial
from typing import Self, Any
from dataclasses import dataclass
import numpy as np
from solib import logger
from solib.utils import config, AbstractionError
from solib.llm_utils import parallelized_call, get_llm_response_async
from solib.datatypes import Answer, Question, Prob


class Judge:

    async def __call__(
        self, question: Question, context: str | None = None
    ) -> dict[Answer, Prob]:
        """Give a probability for each answer_case, based on any context provided."""
        raise AbstractionError


class Agent:

    def __init__(
        self,
        model: str,
        tools: list[callable] | None = None,
        max_tokens: int | None = None,
    ):
        """
        Args:
            model (str): e.g. "gpt-4o-mini".
            tools (list[callable]): tools for the agent.
            max_tokens (int | None): e.g. 2048.
        """
        self.model = model
        self.tools = tools
        self.max_tokens = max_tokens

    async def __call__(
        self,
        prompt: str,
        question: Question,
        answer_case: Answer,
        context: str | None = None,
        words_in_mouth: str | None = None,
    ) -> str:
        """Simulate an AI arguing to convince the judge in favour of answer_case.

        Args:
            prompt (str): formattable prompt that takes in question, answer_case, and context.
            words_in_mouth (str | None): e.g. " Sure, here's my response:\n\n"
            context (str | None): context e.g. transcript of the conversation so far.
            question (Question): question.
            answer_case (Answer): answer case to argue for.
        """
        prompt = prompt.format(
            question=question,
            answer_case=answer_case,
            context=context,
        )
        return await get_llm_response_async(
            model=self.model,
            prompt=prompt,
            words_in_mouth=words_in_mouth,
            tools=self.tools,
            max_tokens=self.max_tokens,
        )


class Protocol:
    """General class for a judge endowed with any (multi-agent) protocol for
    question-answering, e.g. debate, consultancy, blindjudge. Methods for things "
    "like accuracy and testing should be implemented here."""

    @dataclass
    class TranscriptItem:
        answer_case: Answer
        message: str

        def to_prompt(self):
            return (
                f"### (Argument in favour of {self.answer_case.short})\n"
                f"{self.message}\n"
            )

    @dataclass
    class Transcript:
        question: Question
        answer_case: Answer
        transcript: list["Protocol.TranscriptItem"]
        judgement: Prob | None = None

        def __dict__(self):
            return {
                "question": self.question,
                "answer_case": self.answer_case,
                "transcript": self.transcript,
                "judgement": self.judgement,
            }

        def to_prompt(self):
            return "\n".join(item.to_prompt() for item in self.transcript)

        def append(self, x: "Protocol.TranscriptItem"):
            self.transcript.append(x)

    async def run(
        self, agent: Agent, question: Question, answer_case: Answer, **other_components
    ) -> Self.Transcript:
        """Let agent argue for answer_case, and get the probability for answer_case
        based on that.

        Args:
            agent (Agent): agent to oversee/judge.
            question (Question): question.
            answer_case (Answer): answer case the agent argues for.
            other_components (dict): other components e.g. adversary, judge.
        """
        raise AbstractionError

    def judge_score(self, question: Question, answer_case: Answer, prob: Prob):
        if answer_case == question.true_answer:
            return np.log(prob)
        elif answer_case == question.false_answer:
            return np.log(1 - prob)
        else:
            return 0.0

    async def run_and_score(
        self, agent: Agent, question: Question, answer_case: Answer, **other_components
    ) -> float:
        """Score of the judge after agent has argued for answer_case."""
        transcript = await self.run(agent, question, answer_case, **other_components)
        prob = transcript.judgement.prob
        return self.judge_score(question, answer_case, prob)

    async def run_on_all_answer_cases(
        self, agent: Agent, question: Question, **other_components
    ) -> dict[Answer, Self.Transcript]:
        """Judge probability after hearing each answer case."""
        # we'd like to just do this, but it's less efficient:
        # return {
        #     answer_case: await self.run(
        #         agent, question, answer_case, **other_components
        #     )
        #     for answer_case in question.answer_cases
        # }
        transcripts = await parallelized_call(
            partial(self.run, agent=agent, question=question, **other_components),
            question.answer_cases,
        )
        return {
            answer_case: transcript
            for answer_case, transcript in zip(question.answer_cases, transcripts)
        }

    def reward_difference(self, question: Question, probs: dict[Answer, Prob]) -> float:
        return sum(
            np.log(probs[a]) * question.answer_cases[a] for a in question.answer_cases
        )

    async def run_and_reward_difference(
        self, agent: Agent, question: Question, **other_components
    ) -> float:
        """Reward difference for the agent between arguing for true_answer and false_answer."""
        transcripts = await self.run_on_all_answer_cases(
            agent, question, **other_components
        )
        probs = {a: t.judgement.prob for a, t in transcripts.items()}
        return self.reward_difference(question, probs)

    async def test(
        self,
        agent: Agent,
        questions: list[Question],
        write: str = None,
        **other_components,
    ):
        results = []

        async def process_question(question):
            transcripts: dict[Answer, Self.Transcript] = (
                await self.run_on_all_answer_cases(agent, question, **other_components)
            )
            probs: dict[Answer, float] = {
                a: t.judgement.prob for a, t in transcripts.items()
            }
            logger.info(probs)
            judge_scores: dict[Answer, float] = {
                a: self.judge_score(question, a, probs[a])
                for a in question.answer_cases
            }
            logger.info(judge_scores)
            reward_difference: float = self.reward_difference(question, probs)
            logger.info(reward_difference)
            result = {
                "question": question,
                "transcripts": transcripts,
                "probs": probs,
                "judge_scores": judge_scores,
                "reward_difference": reward_difference,
            }
            return result

        results = await parallelized_call(process_question, questions)
        stats = self.compute_stats(results)

        logger.info(stats)

        if write:
            with open(write, "w") as f:
                json.dump(
                    {"results": results, "stats": stats, "config": config(self)}, f
                )

        return results, stats

    def compute_stats(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        reward_differences = [r["reward_difference"] for r in results]
        mean_reward_difference = np.mean(reward_differences)
        std_reward_difference = np.std(reward_differences)
        avg_judge_scores = [np.mean(r["judge_scores"].values()) for r in results]
        mean_avg_judge_score = np.mean(avg_judge_scores)
        std_avg_judge_score = np.std(avg_judge_scores)
        return {
            "mean_reward_difference": mean_reward_difference,
            "std_reward_difference": std_reward_difference,
            "mean_avg_judge_score": mean_avg_judge_score,
            "std_avg_judge_score": std_avg_judge_score,
        }
