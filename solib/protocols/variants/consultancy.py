from solib.datatypes import Answer, Question
from solib.protocols.consultancy import Consultant, Client, Consultancy
from solib.protocols.common import Judge, Transcript

class OpenConsultancy(Consultancy):
    
    def choose_answer(self, question: Question, **kwargs) -> Answer:
        auxillary_judge = Judge(model=self.consultant.model)
        transcript = Transcript(question)
        probabilities = auxillary_judge(transcript, **kwargs).probabilities
        ans_key = max(probabilities, key=probabilities.get)
        return question.possible_answers_dict[ans_key]