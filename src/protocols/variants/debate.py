from typing import Self
from dataclasses import dataclass
from random import random
import re
from src.llm_utils import get_llm_response
from src.datatypes import Answer, Question
from src.protocols.common import Transcript, Judge
from src.protocols.debate import Debate

"""
VARIANT DEBATES
"""

class SimultaneousDebate(Debate):
    
    def run(self, question: Question) -> Transcript:
        assert len(question.possible_answers) == 2
        transcript = Transcript(question, protocol=Debate)
        debater_1_answer, debater_2_answer = question.possible_answers
        if random() > 0.5:  # randomize which debater argues for which answer
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
