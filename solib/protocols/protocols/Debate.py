import functools
import logging
from solib.datatypes import Question, Answer, TranscriptItem, censor
from solib.protocols.abstract import Protocol, QA_Agent, Judge

logger = logging.getLogger(__name__)


class Debate(Protocol):

    def __init__(self, num_turns: int = 2, simultaneous=True, prompt: str = None):
        self.num_turns = num_turns
        self.simultaneous = simultaneous
        super().__init__(prompt=prompt)

    @censor("question", "answer_case")
    async def run(
        self,
        agent: QA_Agent,
        question: Question,
        answer_case: Answer,
        adversary: QA_Agent,
        judge: Judge,
    ) -> Question:
        opp_case = question.neg(answer_case)
        debater_pro = functools.partial(
            agent,
            prompt=self.prompt,
            question=question.to_prompt(),
            answer_case=answer_case.to_prompt(),
        )
        debater_con = functools.partial(
            adversary,
            prompt=self.prompt,
            question=question.to_prompt(),
            answer_case=opp_case.to_prompt(),
        )
        while not self.end_communication(question):
            if self.simultaneous:
                debater_pro_arg = await debater_pro(context=self.ts_to_prompt(question))
                debater_con_arg = await debater_con(context=self.ts_to_prompt(question))
                question.append(
                    TranscriptItem(role=answer_case.short, content=debater_pro_arg)
                )
                question.append(
                    TranscriptItem(role=opp_case.short, content=debater_con_arg)
                )
            else:
                debater_pro_arg = await debater_pro(context=self.ts_to_prompt(question))
                question.append(
                    TranscriptItem(role=answer_case.short, content=debater_pro_arg)
                )
                debater_con_arg = await debater_con(context=self.ts_to_prompt(question))
                question.append(
                    TranscriptItem(role=opp_case.short, content=debater_con_arg)
                )
        result = await judge(question=question, context=self.ts_to_prompt(question))
        return result

    async def run_on_all_answer_cases(
        self,
        agent: QA_Agent,
        question: Question,
        judge: Judge,
        adversary: QA_Agent,
    ) -> Question:
        """Debate specifically is symmetric, so we can subclass this to only run the
        debate once."""
        if agent != adversary:
            # it's not symmetric if agent and adversary are different
            return await super().run_on_all_answer_cases(
                agent=agent, question=question, judge=judge, adversary=adversary
            )
        case_probs_0 = await self.run(
            agent=agent,
            question=question,
            answer_case=question.answer_cases[0],
            adversary=adversary,
            judge=judge,
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
        return len(question.transcript) >= self.num_turns