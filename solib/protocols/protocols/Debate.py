import functools
import logging
import asyncio
from pathlib import Path
from solib.datatypes import Question, Answer, TranscriptItem
from solib.protocols.abstract import Protocol, QA_Agent, Judge
from solib.utils.llm_utils import jinja_env

LOGGER = logging.getLogger(__name__)


class Debate(Protocol):
    def __init__(
        self,
        debater_system_file: str = "debate/qa_agent_system.jinja",
        debater_user_file: str = "debate/qa_agent_user.jinja",
        num_turns: int = 2,
        simultaneous: bool = True,
    ):
        self.debater_system_template = jinja_env.get_template(debater_system_file)
        self.debater_user_template = jinja_env.get_template(debater_user_file)

        self.num_turns = num_turns
        self.simultaneous = simultaneous
        super().__init__(
            debater_system=jinja_env.get_source(debater_system_file),
            debater_user=jinja_env.get_source(debater_user_file),
            num_turns=num_turns,
            simultaneous=simultaneous,
        )

    async def step(
        self,
        agent: QA_Agent,
        question: Question,
        answer_case: Answer,
        adversary: QA_Agent,
        judge: Judge,
        caching: bool = None,
        temperature: float = 0.4,
        write: Path | str | None = None,
        **rendering_components,
    ):
        opp_case = question.neg(answer_case)
        assert answer_case is not None and opp_case is not None
        assert answer_case.short != opp_case.short
        if question.transcript is not None:
            for i, ti in enumerate(question.transcript):
                ti: TranscriptItem
                role: str = ti.role
                if i % 2 == 0:
                    assert role == answer_case.short, f"{[ti.role for ti in question.transcript]}"
                else:
                    assert role == opp_case.short, f"{[ti.role for ti in question.transcript]}"
        trans_len = len(question.transcript) if question.transcript is not None else 0

        debater_pro = functools.partial(
            agent,
            question=question,
            answer_case=answer_case,
            system_prompt_template=self.debater_system_template,
            user_prompt_template=self.debater_user_template,
            extra_user_renders={
                "answer_case_short": answer_case.short,
                "answer_opposite_short": opp_case.short,
            },
            caching=caching,
            temperature=temperature,
            write=write,
        )
        debater_con = functools.partial(
            adversary,
            question=question,
            answer_case=opp_case,
            system_prompt_template=self.debater_system_template,
            user_prompt_template=self.debater_user_template,
            extra_user_renders={
                "answer_case_short": opp_case.short,
                "answer_opposite_short": answer_case.short,
            },
            caching=caching,
            temperature=temperature,
            write=write,
        )
        if question.transcript is not None:
            assert len(question.transcript) == trans_len, f"{len(question.transcript)}=={trans_len}"

        if self.simultaneous:
            tasks = [debater_pro(context=self.ts_to_prompt(question)), debater_con(context=self.ts_to_prompt(question))]
            debater_pro_arg, debater_con_arg = await asyncio.gather(*tasks)
            # debater_pro_arg = await debater_pro(context=self.ts_to_prompt(question))
            # debater_con_arg = await debater_con(context=self.ts_to_prompt(question))
            if question.transcript is not None:
                assert len(question.transcript) == trans_len, f"{len(question.transcript)}=={trans_len}"
            question = question.append(TranscriptItem(role=answer_case.short, content=debater_pro_arg))
            if question.transcript is not None:
                assert len(question.transcript) == trans_len+1, f"{len(question.transcript)}=={trans_len}+1"
            question = question.append(TranscriptItem(role=opp_case.short, content=debater_con_arg))
            assert len(question.transcript) == trans_len + 2, f"Simultaneous debate, {len(question.transcript)}=={trans_len}+2"
        else:
            debater_pro_arg = await debater_pro(context=self.ts_to_prompt(question))
            if question.transcript is not None:
                assert len(question.transcript) == trans_len, f"Sequential debate, {len(question.transcript)}=={trans_len}"
            question = question.append(TranscriptItem(role=answer_case.short, content=debater_pro_arg))
            assert len(question.transcript) == trans_len + 1, f"Sequential debate, {len(question.transcript)}=={trans_len}+1"
            debater_con_arg = await debater_con(context=self.ts_to_prompt(question))
            assert len(question.transcript) == trans_len + 1, f"Sequential debate, {len(question.transcript)}=={trans_len}+1"
            question = question.append(TranscriptItem(role=opp_case.short, content=debater_con_arg))
            assert len(question.transcript) == trans_len + 2, f"Sequential debate, {len(question.transcript)}=={trans_len}+2"
        return question

    async def run_on_all_answer_cases(
        self,
        agent: QA_Agent,
        question: Question,
        judge: Judge,
        adversary: QA_Agent,
        caching: bool = None,
        temperature: float = 0.4,
        write: Path | str | None = None,
    ) -> Question:
        """Debate specifically is symmetric, so we can subclass this to only run the
        debate once."""
        if agent != adversary:
            # it's not symmetric if agent and adversary are different
            return await super().run_on_all_answer_cases(
                agent=agent,
                question=question,
                judge=judge,
                adversary=adversary,
                caching=caching,
                temperature=temperature,
                write=write,
            )
        case_probs_0 = await self.run(
            agent=agent,
            question=question,
            answer_case=question.answer_cases[0],
            adversary=adversary,
            judge=judge,
            caching=caching,
            temperature=temperature,
            write=write,
        )  # elicited probs after arguing for answer_cases[0]
        # but this is the same as the elicited probs after arguing for answer_cases[1]
        # because the adversary is arguing for the opposite answer
        # this also contains the transcript, which is again identical for both
        return Question(
            question=question.question,
            answer_cases=[
                Answer(**(a.model_dump() | {"case_probs": case_probs_0}))
                for a in question.answer_cases
            ],
        )

    def end_communication(self, question: Question) -> bool:
        """Default end_communication method: return True when debate transcript
        exceeds length num_turns. Can also be generalized to a host function that
        e.g. decides which debater goes next etc. Override in subclass."""
        return (
            question.transcript is not None
            and len(question.transcript) >= self.num_turns
        )
