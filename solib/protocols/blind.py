from solib.datatypes import Question
from solib.protocols.abstract import Transcript, Judge, Protocol


class BlindJudgement(Protocol):
    """General class for BlindJudgement."""

    def __init__(
        self,
        judge: Judge,
    ):
        self.judge = judge

    async def run(self, question: Question, **kwargs) -> Transcript:
        transcript = Transcript(question)
        judge_item = await self.judge(transcript, **kwargs)
        transcript.append(judge_item)
        return transcript
