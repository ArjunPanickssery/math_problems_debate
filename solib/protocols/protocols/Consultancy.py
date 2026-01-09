import functools
import logging
from pathlib import Path
from solib.datatypes import Question, Answer, TranscriptItem
from solib.protocols.abstract import Protocol, QA_Agent, Judge
from solib.utils.llm_utils import jinja_env
from solib.utils.verification import generate_argument_with_verification

LOGGER = logging.getLogger(__name__)


class Consultancy(Protocol):
    def __init__(
        self,
        consultant_system_file: str = "consultancy/consultant_system.jinja",
        consultant_user_file: str = "consultancy/consultant_user.jinja",
        client_system_file: str = "consultancy/client_system.jinja",
        client_user_file: str = "consultancy/client_user.jinja",
        num_turns: int = 2,
        consultant_goes_first: bool = False,
    ):
        self.consultant_system_template = jinja_env.get_template(consultant_system_file)
        self.consultant_user_template = jinja_env.get_template(consultant_user_file)
        self.client_system_template = jinja_env.get_template(client_system_file)
        self.client_user_template = jinja_env.get_template(client_user_file)

        self.num_turns = num_turns
        self.consultant_goes_first = consultant_goes_first
        super().__init__(
            consultant_system=jinja_env.get_source(consultant_system_file),
            consultant_user=jinja_env.get_source(consultant_user_file),
            client_system=jinja_env.get_source(client_system_file),
            client_user=jinja_env.get_source(client_user_file),
            num_turns=num_turns,
            consultant_goes_first=consultant_goes_first,
        )

    def end_communication(self, question: Question) -> bool:
        """Default end_communication method: return True when consultancy transcript
        exceeds length num_turns. Override in subclass."""
        return (
            question.transcript is not None
            and len(question.transcript) >= self.num_turns
        )

    def tsitem_to_prompt(self, item: TranscriptItem) -> str:
        if item.role == "client":
            return f"### Enquiry from judge:\n{item.content}\n"
        else:
            return super().tsitem_to_prompt(item)

    async def step(
        self,
        agent: QA_Agent,
        question: Question,
        answer_case: Answer,
        judge: Judge,
        cache_breaker: str | int | None = None,
        temperature: float = 0.4,
        write: Path | str | None = None,
        **rendering_components,
    ) -> Question:
        # Create callable for consultant that accepts feedback
        async def consultant_callable(feedback: str = None):
            return await agent(
                system_prompt_template=self.consultant_system_template,
                user_prompt_template=self.consultant_user_template,
                question=question,
                answer_case=answer_case,
                context=self.ts_to_prompt(question),
                feedback=feedback,
                cache_breaker=cache_breaker,
                temperature=temperature,
                write=write,
            )

        client_agent = QA_Agent(
            model=judge.model,
            tools=judge.tools,
        )
        client = functools.partial(
            client_agent,
            question=question,
            system_prompt_template=self.client_system_template,
            user_prompt_template=self.client_user_template,
            cache_breaker=cache_breaker,
            temperature=temperature,
            write=write,
        )

        if (question.transcript in [None, []] and self.consultant_goes_first) or (
            question.transcript and question.transcript[-1].role == "client"
        ):
            # Consultant's turn - use verification
            cons_resp, verification_metadata = await generate_argument_with_verification(
                agent_callable=consultant_callable,
                question=question,
                answer_case=answer_case,
            )
            question_ = question.append(TranscriptItem(
                role=answer_case.short,
                content=cons_resp,
                metadata=verification_metadata if verification_metadata else None,
            ))
        elif (question.transcript in [None, []] and not self.consultant_goes_first) or (
            question.transcript and question.transcript[-1].role != "client"
        ):
            # Client's turn - no verification needed (client is asking questions, not arguing)
            client_resp = await client(context=self.ts_to_prompt(question))
            question_ = question.append(TranscriptItem(role="client", content=client_resp))
        else:
            raise ValueError("Logic is no longer valid in this universe. Please try another one.")
        return question_
