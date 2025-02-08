import logging
from pathlib import Path
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
        caching: bool = None,
        temperature: float = 0.4,
        write: Path | str | None = None,
        **rendering_components,
    ) -> Question:
        assert isinstance(question, Question)
        agent_response = await agent(
            question=question,
            answer_case=answer_case,
            extra_user_renders={"answer_case_short": answer_case.short},
            context=self.ts_to_prompt(question),
            caching=caching,
            temperature=temperature,
            write=write,
            **rendering_components,
        )
        question.append(TranscriptItem(role=answer_case.short, content=agent_response))
        assert question.transcript is not None
        result = await judge(
            question=question,
            context=self.ts_to_prompt(question),
            caching=caching,
            write=write,
        )
        assert result.transcript is not None
        return result
