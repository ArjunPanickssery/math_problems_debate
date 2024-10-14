import logging
from solib.protocols.abstract import Protocol, QA_Agent, Judge
from solib.datatypes import Question_stripped, Answer
from solib.protocols.abstract import Protocol, QA_Agent, Judge

logger = logging.getLogger(__name__)


class Propaganda(Protocol):
    async def run(
        self,
        agent: QA_Agent,
        question: Question_stripped,
        answer_case: Answer,
        judge: Judge,
    ) -> "Propaganda.Transcript":
        transcript = self.Transcript()
        agent_answer = await agent(
            prompt=self.prompt,
            question=question.to_prompt(),
            answer_case=answer_case.to_prompt(),
            context=transcript.to_prompt(),
        )
        transcript.append(self.TranscriptItem(answer_case, context=agent_answer))
        judgement = await judge(question=question, context=transcript.to_prompt())
        transcript.judgement = judgement
        return transcript
