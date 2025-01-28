import logging
from pathlib import Path
from solib.utils import (
    dump_config,
    AbstractionError,
    parallelized_call,
    write_jsonl_async,
    write_json,
)
from solib.utils import LLM_Agent
from solib.datatypes import Answer, Question, TranscriptItem
from solib.data.loading import Dataset
from solib.utils.llm_utils import jinja_env

LOGGER = logging.getLogger(__name__)


class Judge(LLM_Agent):
    async def __call__(
        self, question: Question, context: str | None = None, cache_breaker: int = 0
    ) -> Question:
        """Add probabilities to each answer_case."""
        raise AbstractionError

    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model}, tools={self.tools})"


class QA_Agent(LLM_Agent):
    """A generally convenient wrapper around an LLM, that answers a given prompt,
    optionally formatted with a context, question and answer case."""

    async def __call__(
        self,
        prompt_file: str = None,
        question: Question = None,
        answer_case: Answer = None,
        context: str | None = None,
        words_in_mouth: str | None = None,
        max_tokens: int = 2048,
        cache_breaker: int = 0,
        temperature: float = 0.4,
    ) -> str:
        """Simulate an AI arguing to convince the judge in favour of answer_case.

        Args:
            prompt_file (str): path to a jinja template file that overrides the default prompt template.
            words_in_mouth (str | None): e.g. " Sure, here's my response:\n\n". Only supported for HF / local models.
            context (str | None): context e.g. transcript of the conversation so far.
            question (Question): question.
            answer_case (Answer): answer case to argue for.
            max_tokens (int): max tokens for the LLM.
            cache_breaker (int): dummy integer that is used to invalidate cache entries.
            temperature (float): temperature for sampling.
        """
        prompt_template = self.prompt_template
        words_in_mouth = words_in_mouth or self.words_in_mouth
        if isinstance(question, Question):
            question = question.to_prompt()
        if isinstance(answer_case, Answer):
            answer_case = answer_case.to_prompt()
        prompt = prompt_template.render(
            question=question,
            answer_case=answer_case,
            context=context,
        )
        return await self.get_response(
            prompt=prompt,
            words_in_mouth=words_in_mouth,
            max_tokens=max_tokens,
            cache_breaker=cache_breaker,
            temperature=temperature,
        )

    def __init__(
        self,
        model: str = None,
        tools: list[callable] | None = None,
        hf_quantization_config=None,
        prompt_file: str = "qa_agent.jinja",
        words_in_mouth: str = None,
    ):
        super().__init__(
            model=model, tools=tools, hf_quantization_config=hf_quantization_config
        )
        self.prompt_template = jinja_env.get_template(prompt_file)
        self.words_in_mouth = words_in_mouth
        self.dict = {
            "model": self.model,
            "tools": self.tools,
            "prompt": jinja_env.get_source(prompt_file),
            "words_in_mouth": self.words_in_mouth,
        }

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
            **other_components,
        )

    async def run(
        self,
        agent: QA_Agent,
        question: Question,
        answer_case: Answer,
        judge: Judge,
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
                **other_components,
            )
            assert len(question.transcript) > t
        result = await judge(question=question, context=self.ts_to_prompt(question))
        assert result.is_elicited
        assert result.transcript is not None
        return result

    async def run_on_all_answer_cases(
        self,
        agent: QA_Agent,
        question: Question,
        judge: Judge,
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

    async def experiment(
        self,
        agent: QA_Agent,
        questions: Dataset,
        judge: Judge,
        write: Path | str | None = None,
        **other_components,
    ) -> tuple[list[Question], dict]:
        results = []

        if write:
            Path(write).mkdir(parents=True, exist_ok=True)
            write_results = Path(write) / "results.jsonl"
            write_stats = Path(write) / "stats.json"
            write_config = Path(write) / "config.json"
        else:
            write_results = write_stats = write_config = None

        async def process_question(question: Question):
            # censor question before sending to any AI
            question_censored = question.censor()

            result = await self.run_on_all_answer_cases(
                agent=agent,
                question=question_censored,
                judge=judge,
                **other_components,
            )

            # reattach ground truth values to trigger computation of scores
            result = result.uncensor(question)
            await write_jsonl_async(
                result.model_dump(exclude_none=True), path=write_results, append=True
            )
            assert result.is_argued
            assert result.is_grounded
            return result

        results = await parallelized_call(process_question, questions, use_tqdm=True)
        stats = Question.compute_stats(results)

        write_json(stats, path=write_stats)
        write_json(
            dump_config(
                {
                    "protocol": self,
                    "agent": agent,
                    "judge": judge,
                    **other_components,
                }
            ),
            path=write_config,
        )

        return results, stats
