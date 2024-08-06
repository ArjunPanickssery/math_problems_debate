from datatypes import Answer, Question
from src.protocols.consultancy import Consultant, Client, Consultancy
from src.protocols.common import Judge, Transcript

class OpenConsultancy(Consultancy):
    
    def choose_answer(self, question: Question) -> Answer:
        auxillary_judge = Judge(model=self.consultant.model)
        transcript = Transcript(question)
        probabilities = auxillary_judge(transcript).judgement.probabilities
        ans_key = max(probabilities, key=probabilities.get)
        return question.possible_answers_dict[ans_key]