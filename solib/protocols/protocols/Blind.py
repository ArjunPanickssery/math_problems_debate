import logging
from pathlib import Path
from solib.datatypes import Question, Answer
from solib.protocols.abstract import Protocol, QA_Agent, Judge

LOGGER = logging.getLogger(__name__)


class Blind(Protocol):
    """Absolute baseline: Judge unassisted by AI."""

    async def run(
        self,
        agent: QA_Agent,
        question: Question,
        answer_case: Answer,
        judge: Judge,
        cache_breaker: str | int | None = None,
        write: Path | str | None = None,
        **rendering_components,
    ) -> Question:
        result = await judge(
            question=question,
            context=self.ts_to_prompt(question),
            cache_breaker=cache_breaker,
            write=write,
        )
        return result
