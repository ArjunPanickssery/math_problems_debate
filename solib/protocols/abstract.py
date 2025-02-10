import copy
import logging
from pathlib import Path
from solib.datatypes import Answer, Question, TranscriptItem
from solib.utils import (
    dump_config,
    AbstractionError,
    parallelized_call,
    write_jsonl_async,
    write_json,
)
from solib.utils import LLM_Agent
from solib.data.loading import Dataset
from solib.utils.llm_utils import jinja_env

LOGGER = logging.getLogger(__name__)


class Judge(LLM_Agent):
    async def __call__(
        self,
        question: Question,
        context: str | None = None,
        caching: bool = None,
    ) -> Question:
        """Add probabilities to each answer_case."""
        raise AbstractionError

    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model}, tools={self.tools})"


class QA_Agent(LLM_Agent):
    """A generally convenient wrapper around an LLM, that answers a given prompt,
    optionally formatted with a context, question and answer case."""

    def __init__(
        self,
        model: str = None,
        tools: list[callable] | None = None,
        system_prompt_file: str = "qa_agent/qa_agent_system.jinja",
        user_prompt_file: str = "qa_agent/qa_agent_user.jinja",
        words_in_mouth: str | None = None,
    ):
        super().__init__(model=model, tools=tools)
        self.system_template = jinja_env.get_template(system_prompt_file)
        self.user_template = jinja_env.get_template(user_prompt_file)
        self.words_in_mouth = words_in_mouth
        self.dict = {
            "model": self.model,
            "tools": self.tools,
            "system_prompt": jinja_env.get_source(system_prompt_file),
            "user_prompt": jinja_env.get_source(user_prompt_file),
        }

    async def __call__(
        self,
        question: Question = None,
        answer_case: Answer = None,
        context: str | None = None,
        words_in_mouth: str | None = None,
        max_tokens: int = 2048,
        caching: bool = None,
        temperature: float = 0.4,
        extra_user_renders: dict | None = None,
        system_prompt_template: str | None = None,
        user_prompt_template: str | None = None,
        write: Path | str | None = None,
    ) -> str:
        """Simulate an AI arguing to convince the judge in favour of answer_case.

        Args:
            words_in_mouth (str | None): e.g. " Sure, here's my response:\n\n". Only supported for HF / local models.
            context (str | None): context e.g. transcript of the conversation so far.
            question (Question): question.
            answer_case (Answer): answer case to argue for.
            max_tokens (int): max tokens for the LLM.
            caching (bool): to disable caching.
            temperature (float): temperature for sampling.
            write (Path | str | None): Path to write prompt history to.
        """
        words_in_mouth = words_in_mouth or self.words_in_mouth
        if isinstance(question, Question):
            question_ = question.to_prompt()
            trans_len = (
                len(question.transcript) if question.transcript is not None else 0
            )
        else:
            question_ = question
        if isinstance(answer_case, Answer):
            answer_case_ = answer_case.to_prompt()
        else:
            answer_case_ = answer_case

        sys_prompt = system_prompt_template or self.system_template
        user_prompt = user_prompt_template or self.user_template

        if isinstance(question, Question):
            assert (
                len(question.transcript) if question.transcript is not None else 0
            ) == trans_len

        messages = [
            {
                "role": "system",
                "content": sys_prompt.render(),
            },
            {
                "role": "user",
                "content": user_prompt.render(
                    question=question_,
                    answer_case=answer_case_,
                    context=context,
                    **(extra_user_renders or {}),
                ),
            },
        ]

        if isinstance(question, Question):
            assert (
                len(question.transcript) if question.transcript is not None else 0
            ) == trans_len

        for m in messages:
            assert isinstance(m["content"], str)

        question_original = copy.deepcopy(question)

        resp = await self.get_response(
            messages=messages,
            words_in_mouth=words_in_mouth,
            max_tokens=max_tokens,
            caching=caching,
            temperature=temperature,
            write=write,
        )

        if isinstance(question, Question):
            DEBUG_INFO = (
                f"{type(self)}\n"
                f"question.transcript: {question.transcript}\n"
                f"original question.transcript: {question_original.transcript}\n"
                f"messages: {messages}\n"
                f"words_in_mouth: {words_in_mouth}\n"
                f"max_tokens: {max_tokens}\n"
                f"caching: {caching}\n"
                f"temperature: {temperature}\n"
                f"write: {write}\n"
                f"response: {resp}\n"
            )
            assert (
                len(question.transcript) if question.transcript is not None else 0
            ) == trans_len, DEBUG_INFO

        return resp

    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model}, tools={self.tools})"


class Protocol:
    """General class for a judge endowed with any (multi-agent) protocol for
    question-answering, e.g. debate, consultancy, blindjudge.
    """

    def __init__(self, **kwargs):
        self.dict = kwargs

    def tsitem_to_prompt(self, item: TranscriptItem) -> str:
        return f"## Argument in favour of {item.role}\n{item.content}\n"

    def ts_to_prompt(self, transcript: list[TranscriptItem] | Question | None) -> str:
        if isinstance(transcript, Question):
            transcript = transcript.transcript
        if transcript is None:
            return ""
        return "\n".join(self.tsitem_to_prompt(item) for item in transcript)

    async def end_communication(self, question: Question) -> bool:
        raise AbstractionError

    async def step(
        self,
        agent: QA_Agent,
        question: Question,
        answer_case: Answer,
        judge: Judge,
        write: Path | str | None = None,
        **other_components,
    ) -> Question:
        """You can subclass step() and end_communication() and have run() defined
        automatically, or you can define run() and step() will just be run().
        Or you can subclass both."""
        # raise AbstractionError
        return await self.run(
            agent=agent,
            question=question,
            answer_case=answer_case,
            judge=judge,
            write=write,
            **other_components,
        )

    async def run(
        self,
        agent: QA_Agent,
        question: Question,
        answer_case: Answer,
        judge: Judge,
        write: Path | str | None = None,
        **other_components,
    ) -> Question:
        """Let agent argue for answer_case, and get the probability for answer_case
        based on that. Can be subclassed to take a different form than "run self.step()
        until self.end_communication()".

        Args:
            agent (QA_Agent): agent to oversee/judge.
            question (Question): question.
            answer_case (Answer): answer case the agent argues for.
            judge (Judge): judge to elicit a probability.
            write (Path | str | None): Path to write prompt history to.
            other_components (dict): other components e.g. adversary.

        Returns:
            Question with added transcript + probs for each answer_case.
        """
        while not self.end_communication(question):
            t = len(question.transcript) if question.transcript is not None else 0
            question = await self.step(
                agent=agent,
                question=question,
                answer_case=answer_case,
                judge=judge,
                write=write,
                **other_components,
            )
            assert len(question.transcript) > t
        result = await judge(
            question=question, context=self.ts_to_prompt(question), write=write
        )
        assert result.is_elicited
        assert result.transcript is not None
        return result

    async def run_on_all_answer_cases(
        self,
        agent: QA_Agent,
        question: Question,
        judge: Judge,
        write: Path | str | None = None,
        **other_components,
    ) -> Question:
        """For each answer_case in Question, run the protocol with that answer_case,
        and update that answer_case with the Question produced by Protocol.run()"""
        # we'd like to just do this, but it's less efficient:
        # return {
        #     answer_case: await self.run(
        #         agent, question, answer_case, **other_components
        #     )
        #     for answer_case in question.answer_cases
        # }

        case_probss = await parallelized_call(
            lambda answer_case: self.run(
                agent=agent,
                question=question,
                judge=judge,
                answer_case=answer_case,
                write=write,
                **other_components,
            ),
            question.answer_cases,
        )
        result = Question(
            question=question.question,
            answer_cases=[
                Answer(**(a.model_dump() | {"case_probs": case_probs}))
                for a, case_probs in zip(question.answer_cases, case_probss)
            ],
        )
        assert result.is_argued
        return result

    def get_experiment_config(
        self, agent: QA_Agent, judge: Judge, **other_components
    ) -> dict:
        return dump_config(
            {
                "protocol": self,
                "agent": agent,
                "judge": judge,
                **other_components,
            }
        )

    async def experiment(
        self,
        agent: QA_Agent,
        questions: Dataset,
        judge: Judge,
        write: Path | str | None = None,
        continue_from_results: dict[tuple, Question] | None = None,
        **other_components,
    ) -> tuple[list[Question], dict]:
        results = []

        if write:
            Path(write).mkdir(parents=True, exist_ok=True)
            write_config = Path(write) / "config.json"
            write_results = Path(write) / "results.jsonl"
            write_stats = Path(write) / "stats.json"
        else:
            write_results = write_stats = write_config = None

        write_json(
            self.get_experiment_config(agent=agent, judge=judge, **other_components),
            path=write_config,
        )

        async def process_question(question: Question):
            result = None
            if continue_from_results is not None:
                result = continue_from_results.get(question.id, None)
                LOGGER.debug(f"Continuing from result: {result}")

            if result is None:
                # censor question before sending to any AI
                question_censored = question.censor()

                result = await self.run_on_all_answer_cases(
                    agent=agent,
                    question=question_censored,
                    judge=judge,
                    write=write,
                    **other_components,
                )

                # reattach ground truth values to trigger computation of scores
                result = result.uncensor(question)

            assert result.is_argued
            assert result.is_grounded

            await write_jsonl_async(
                result.model_dump(exclude_none=True), path=write_results, append=True
            )

            return result

        results = await parallelized_call(process_question, questions)
        stats = Question.compute_stats(results)

        write_json(stats, path=write_stats)
        return results, stats
