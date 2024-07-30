from src.solib.common import Transcript, Question, Judge, Protocol


class BlindJudge(Protocol):
    """General class for BlindJudge.
    - Judge asks questions, debater responds.
    - Ends when Judge has asked num_turns questions.
    """

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
