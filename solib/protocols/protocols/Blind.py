import logging
from solib.datatypes import Question, Answer, censor
from solib.protocols.abstract import Protocol, QA_Agent, Judge

logger = logging.getLogger(__name__)


class Blind(Protocol):
    """Absolute baseline: Judge unassisted by AI."""

    @censor("question", "answer_case", reattach_from="question")
    async def run(
        self,
        agent: QA_Agent,
        question: Question,
        answer_case: Answer,
        judge: Judge,
    ) -> Question:
        result = await judge(question=question, context=self.ts_to_prompt(question))
        return result
