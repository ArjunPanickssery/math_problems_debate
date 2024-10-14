import functools
import logging
from solib.datatypes import Question_stripped, Answer, Prob
from solib.protocols.abstract import Protocol, QA_Agent, Judge

logger = logging.getLogger(__name__)


class Debate(Protocol):

    def __init__(self, num_turns: int = 2, simultaneous=True, prompt: str = None):
        self.num_turns = num_turns
        self.simultaneous = simultaneous
        super().__init__(prompt=prompt)

    async def run(
        self,
        agent: QA_Agent,
        question: Question_stripped,
        answer_case: Answer,
        adversary: QA_Agent,
        judge: Judge,
    ) -> "Debate.Transcript":
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
        transcript = self.Transcript()
        while not self.end_communication(transcript):
            if self.simultaneous:
                debater_pro_arg = await debater_pro(context=transcript.to_prompt())
                debater_con_arg = await debater_con(context=transcript.to_prompt())
                transcript.append(self.TranscriptItem(answer_case, debater_pro_arg))
                transcript.append(self.TranscriptItem(opp_case, debater_con_arg))
            else:
                debater_pro_arg = await debater_pro(context=transcript.to_prompt())
                transcript.append(self.TranscriptItem(answer_case, debater_pro_arg))
                debater_con_arg = await debater_con(context=transcript.to_prompt())
                transcript.append(self.TranscriptItem(opp_case, debater_con_arg))
        judgement = await judge(question=question, context=transcript.to_prompt())
        transcript.judgement = judgement
        return transcript

    async def run_on_all_answer_cases(
        self,
        agent: QA_Agent,
        question: Question_stripped,
        judge: Judge,
        adversary: QA_Agent,
    ) -> dict[Answer, "Debate.Transcript"]:
        """Debate specifically is symmetric, so we can subclass this to only run the
        debate once."""
        answer_0_transcript = await self.run(
            agent=agent,
            question=question,
            answer_case=question.answer_cases[0],
            adversary=adversary,
            judge=judge,
        )
        answer_1_transcript = self.Transcript(
            question=question,
            answer_case=question.answer_cases[1],
            transcript=answer_0_transcript.transcript,
            judgement=Prob(1 - answer_0_transcript.judgement.prob),
        )
        return {
            question.answer_cases[0]: answer_0_transcript,
            question.answer_cases[1]: answer_1_transcript,
        }

    def end_communication(self, transcript: "Debate.Transcript"):
        """Default end_communication method: return True when debate transcript
        exceeds length num_turns. Can also be generalized to a host function that
        e.g. decides which debater goes next etc. Override in subclass."""
        return len(transcript.transcript) >= self.num_turns
