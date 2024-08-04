from datatypes import Question
from src.protocols.common import Transcript, Judge, Protocol


class BlindJudge(Protocol):
    """General class for BlindJudge."""

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
