import logging
from solib.llm_utils import parallelized_call
from solib.datatypes import Question, Answer, Score
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
        self.judge = judge or JustAskProbabilityJudge()
        self.protocol = protocol or Propaganda()
        self.other_components = other_components
        # inherit other stuff
        self.model = self.agent.model
        self.tools = self.agent.tools
        self.prompt = self.agent.prompt
        self.words_in_mouth = self.agent.words_in_mouth
        self.dict = self.agent.dict

    async def __call__(
        self,
        prompt: str = None,
        question: Question = None,
        answer_case: Answer = None,
        context: str | None = None,
        words_in_mouth: str | None = None,
        max_tokens: int = 2048,
    ) -> str:
        async def run_agent(kwargs):
            i = kwargs.pop("i")
            LOGGER.debug(f"local cache_breaker during BON: {i}")
            transcript = await self.protocol.step(
                agent=self.agent,
                question=kwargs["question"],
                answer_case=kwargs["answer_case"],
                judge=self.judge,
                cache_breaker=i,
                **self.other_components,
            )
            response = transcript.transcript[-1].content
            result = await self.judge(
                question=transcript, context=self.protocol.ts_to_prompt(transcript)
            )
            agent_score = Score.calc(result, kwargs["answer_case"]).log
            return response, agent_score

        results = await parallelized_call(
            run_agent,
            [
                {
                    "i": i,
                    "question": question,
                    "answer_case": answer_case,
                }
                for i in range(self.n)
            ],
        )

        best_response, best_score = max(results, key=lambda x: x[1])

        return best_response
