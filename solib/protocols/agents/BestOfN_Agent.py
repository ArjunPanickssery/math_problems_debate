import logging
from pathlib import Path
from solib.datatypes import Question, Answer, Score
from solib.utils import parallelized_call
from solib.utils.llm_utils import DEFAULT_BON_MODEL
from solib.protocols.abstract import QA_Agent, Protocol, Judge
from solib.protocols.judges import JustAskProbabilityJudge
from solib.protocols.protocols import Propaganda

LOGGER = logging.getLogger(__name__)


class BestOfN_Agent(QA_Agent):
    def __init__(
        self,
        n: int,
        agent: QA_Agent,
        judge: Judge = None,
        protocol: Protocol = None,
        **other_components,
    ):
        self.n = n
        self.agent = agent
        self.judge = judge or JustAskProbabilityJudge(DEFAULT_BON_MODEL)
        self.protocol = protocol or Propaganda()
        self.other_components = other_components
        # inherit other stuff
        self.model = self.agent.model
        self.tools = self.agent.tools
        self.dict = {
            "n": self.n,
            "agent": self.agent.dict,
            "judge": self.judge.dict,
            "protocol": self.protocol.dict,
        }

    async def __call__(
        self,
        question: Question = None,
        answer_case: Answer = None,
        context: str | None = None,
        words_in_mouth: str | None = None,
        max_tokens: int = 2048,
        temperature: float = None,
        extra_user_renders: dict | None = None,
        cache_breaker: str | int | None = None,
        write: Path | str | None = None,
        return_prompt: bool = False,
        **rendering_components,
    ) -> str | tuple[str, str]:
        async def run_agent(kwargs: dict):
            transcript = await self.protocol.step(
                agent=self.agent,
                question=kwargs["question"],
                answer_case=kwargs["answer_case"],
                judge=self.judge,
                temperature=(
                    temperature if temperature is not None else kwargs["temperature"]
                ),
                extra_user_renders=extra_user_renders,
                cache_breaker=kwargs["cache_breaker"], # trick to break cache, but still save BoN results
                write=write,
                **self.other_components,
            )
            response = transcript.transcript[-1].content
            prompt_str = transcript.transcript[-1].prompt
            result = await self.judge(
                question=transcript, context=self.protocol.ts_to_prompt(transcript),
                cache_breaker=kwargs["cache_breaker"],
                write=write,
            )
            agent_score = Score.calc(result, kwargs["answer_case"]).log
            return response, agent_score, prompt_str

        results = await parallelized_call(
            run_agent,
            [
                {
                    "question": question,
                    "answer_case": answer_case,
                    "temperature": (0.4 if self.n == 0 else 0.8),
                    "cache_breaker": str(i),
                }
                for i in range(self.n)
            ],
        )

        best_response, best_score, best_prompt = max(results, key=lambda x: x[1])

        if return_prompt:
            return best_response, best_prompt
        return best_response
