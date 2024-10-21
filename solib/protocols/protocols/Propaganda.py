import logging
from solib.datatypes import Question, Answer, TranscriptItem
from solib.protocols.abstract import Protocol, QA_Agent, Judge

logger = logging.getLogger(__name__)


class Propaganda(Protocol):

    async def run(
        self,
        agent: QA_Agent,
        question: Question,
        answer_case: Answer,
        judge: Judge,
    ) -> Question:
        agent_response = await agent(
            prompt=self.prompt,
            question=question.to_prompt(),
            answer_case=answer_case.to_prompt(),
            context=self.ts_to_prompt(question),
        )
        question.append(TranscriptItem(role=answer_case.short, content=agent_response))
        result = await judge(question=question, context=self.ts_to_prompt(question))
        return result
