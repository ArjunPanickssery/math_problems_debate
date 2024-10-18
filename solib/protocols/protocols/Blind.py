import logging
from solib.datatypes import Question_stripped, Answer
from solib.protocols.abstract import Protocol, QA_Agent, Judge

logger = logging.getLogger(__name__)


class Blind(Protocol):
    """Absolute baseline: Judge unassisted by AI."""

    async def run(
        self,
        agent: QA_Agent,
        question: Question_stripped,
        answer_case: Answer,
        judge: Judge,
    ) -> "Blind.Transcript":
        transcript = self.Transcript()
        judgement = await judge(question=question, context=transcript.to_prompt())
        transcript.judgement = judgement
        return transcript
