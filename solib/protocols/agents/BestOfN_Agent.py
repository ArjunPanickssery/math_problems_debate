import logging
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
        self.user_template = self.agent.user_template
        self.system_template = self.agent.system_template
        self.dict = self.agent.dict

    async def __call__(
        self,
        system_prompt_template: str | None = None,
        user_prompt_template: str | None = None,
        question: Question = None,
        answer_case: Answer = None,
        context: str | None = None,
        words_in_mouth: str | None = None,
        max_tokens: int = 2048,
        caching: bool = False,
        temperature: float = None,
    ) -> str:
        async def run_agent(kwargs: dict):
            transcript = await self.protocol.step(
                agent=self.agent,
                question=kwargs["question"],
                answer_case=kwargs["answer_case"],
                judge=self.judge,
                caching=False,  # IMPORTANT! want all n responses to be different
                temperature=(
                    temperature if temperature is not None else kwargs["temperature"]
                ),
                **self.other_components,
            )
            response = transcript.transcript[-1].content
            result = await self.judge(
                question=transcript, context=self.protocol.ts_to_prompt(transcript)
            )
            agent_score = Score.calc(result, kwargs["answer_case"]).log
            return response, agent_score


        # icky hack to avoid having to pass in system_prompt_template and user_prompt_template everywhere
        system_prompt_stored = self.system_template
        user_prompt_stored = self.user_template

        self.agent.system_template = system_prompt_template or self.system_template
        self.agent.user_template = user_prompt_template or self.user_template

        results = await parallelized_call(
            run_agent,
            [
                {
                    "question": question,
                    "answer_case": answer_case,
                    "temperature": (0.4 if self.n == 0 else 0.8),
                }
                for i in range(self.n)
            ],
        )

        self.agent.system_template = system_prompt_stored
        self.agent.user_template = user_prompt_stored
        
        best_response, best_score = max(results, key=lambda x: x[1])

        return best_response
