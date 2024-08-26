from typing import Self
from dataclasses import dataclass
from solib.utils import random
from solib.llm_utils import get_llm_response, get_llm_response_async
from solib.datatypes import Answer, Question
from solib.protocols.common import Transcript, Judge
from solib.protocols.debate import SequentialDebate

"""
VARIANT DEBATES
"""

class SimultaneousDebate(SequentialDebate):
    
    async def run(self, question: Question, **kwargs) -> Transcript:
        assert len(question.possible_answers) == 2
        transcript = Transcript(question, protocol=SequentialDebate)
        debater_1_answer, debater_2_answer = question.possible_answers
        if random(question, **kwargs) > 0.5:  # randomize which debater argues for which answer
            debater_1_answer, debater_2_answer = (
                debater_2_answer,
                debater_1_answer,
            )
        while not self.end_communication(transcript):
            debater_1_item = await self.debater_1(debater_1_answer, transcript)
            # transcript.append(debater_1_item) # <-- change
            debater_2_item = await self.debater_2(debater_2_answer, transcript)
            transcript.append(debater_1_item) # <-- change
            transcript.append(debater_2_item)
        judge_item = await self.judge(transcript)
        transcript.append(judge_item)
        return transcript
