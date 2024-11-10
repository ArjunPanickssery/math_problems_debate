import logging
from solib.datatypes import Question, Answer, TranscriptItem
from solib.protocols.abstract import Protocol, QA_Agent, Judge

LOGGER = logging.getLogger(__name__)


class Propaganda(Protocol):
    async def run(
        self,
        agent: QA_Agent,
        question: Question,
        answer_case: Answer,
        judge: Judge,
        cache_breaker: int = 0,
        temperature: float = 0.4,
    ) -> Question:
        assert isinstance(question, Question)
        agent_response = await agent(
            prompt=self.prompt,
            question=question,
            answer_case=answer_case,
            context=self.ts_to_prompt(question),
            cache_breaker=cache_breaker,
            temperature=temperature,
        )
        question.append(TranscriptItem(role=answer_case.short, content=agent_response))
        assert question.transcript is not None
        result = await judge(
            question=question,
            context=self.ts_to_prompt(question),
            cache_breaker=cache_breaker,
        )
        assert result.transcript is not None
        return result
