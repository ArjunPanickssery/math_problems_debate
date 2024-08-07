from src.datatypes import Question
from src.protocols.common import Transcript, Judge, Protocol


class BlindJudgement(Protocol):
    """General class for BlindJudgement."""

    def __init__(
        self,
        judge: Judge,
    ):
        self.judge = judge

    def run(self, question: Question) -> Transcript:
        transcript = Transcript(question)
        judge_item = self.judge(transcript)
        transcript.append(judge_item)
        return transcript
