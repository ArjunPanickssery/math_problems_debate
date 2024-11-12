import logging
from solib.datatypes import Question, Answer
from solib.protocols.abstract import Protocol, QA_Agent, Judge

#LOGGER = logging.get#LOGGER(__name__)


class Blind(Protocol):
    """Absolute baseline: Judge unassisted by AI."""

    async def run(
        self,
        agent: QA_Agent,
        question: Question,
        answer_case: Answer,
        judge: Judge,
        cache_breaker: int = 0,
    ) -> Question:
        result = await judge(
            question=question,
            context=self.ts_to_prompt(question),
            cache_breaker=cache_breaker,
        )
        return result
