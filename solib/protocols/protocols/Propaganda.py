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
        temperature: float = 0.4,
        write: Path | str | None = None,
        cache_breaker: str | int | None = None,
        **rendering_components,
    ) -> Question:
        assert isinstance(question, Question)
        if "extra_user_renders" in rendering_components:
            if rendering_components["extra_user_renders"] is None:
                rendering_components["extra_user_renders"] = {}
            rendering_components["extra_user_renders"] |= {
                "answer_case_short": answer_case.short
            }
        agent_response = await agent(
            question=question,
            answer_case=answer_case,
            # extra_user_renders={"answer_case_short": answer_case.short},
            context=self.ts_to_prompt(question),
            cache_breaker=cache_breaker,
            temperature=temperature,
            write=write,
            **rendering_components,
        )
        question = question.append(
            TranscriptItem(role=answer_case.short, content=agent_response)
        )
        assert question.transcript is not None
        result = await judge(
            question=question,
            context=self.ts_to_prompt(question),
            cache_breaker=cache_breaker,
            write=write,
        )
        assert result.transcript is not None
        return result
