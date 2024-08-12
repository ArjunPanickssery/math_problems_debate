from typing import Self
from dataclasses import dataclass
import random
from solib.utils import seed
from solib.llm_utils import get_llm_response
from solib.datatypes import Answer, Question
from solib.protocols.common import Transcript, Judge
from solib.protocols.debate import SequentialDebate

"""
VARIANT DEBATES
"""

class SimultaneousDebate(SequentialDebate):
    
    def run(self, question: Question, **kwargs) -> Transcript:
        seed(question, **kwargs)
        assert len(question.possible_answers) == 2
        transcript = Transcript(question, protocol=SequentialDebate)
        debater_1_answer, debater_2_answer = question.possible_answers
        if random.random() > 0.5:  # randomize which debater argues for which answer
            debater_1_answer, debater_2_answer = (
                debater_2_answer,
                debater_1_answer,
            )
        while not self.end_communication(transcript):
            debater_1_item = self.debater_1(debater_1_answer, transcript)
            # transcript.append(debater_1_item) # <-- change
            debater_2_item = self.debater_2(debater_2_answer, transcript)
            transcript.append(debater_1_item) # <-- change
            transcript.append(debater_2_item)
        judge_item = self.judge(transcript)
        transcript.append(judge_item)
        return transcript
