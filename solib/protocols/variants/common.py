from typing import Self
from dataclasses import dataclass
from random import random
import re
from solib.llm_utils import get_llm_response
from solib.protocols.common import Transcript, Judge

"""
VARIANT JUDGES
"""

class JustAskProbabilityJudge(Judge):
    """Instead of asking a judge to make a binary choice and getting the token probabilities,
    we just ask the judge to give probabilities. We will ask it for probabilities for each
    possible answer separately, and then we will normalize them."""
    
    prompt = (
        "I will give you a question to answer, and a list of possible "
        "answers. I want you to give me your most informed judgment on which "
        "answer is correct, based on any information I provide. Your response "
        "should simply give your credence/probability for a possible answer.\n\n"
        "{transcript}\n\n"
        "What is your probability that the answer is {answer}? Just give me the "
        "numerical probability in decimal terms (NOT percentage), nothing else: \n\n"
    )
    
    def __call__(self, transcript: Transcript) -> "Self.TranscriptItem":
        probabilities = {}
        for answer in transcript.question.possible_answers:
            words_in_mouth = f" The probability that the answer is {answer.symbol} is:\n\n"
            response = get_llm_response(
                prompt=self.prompt.format(transcript=transcript, answer=answer),
                model=self.model,
                max_tokens=5,
                words_in_mouth=words_in_mouth,
            )
            # extract numeric value from response 
            # HACK. Maybe we should use instructor instead of regex
            print('PRINTING RESPONSE')
            print(response)
            probability = float(re.findall("\d+\.\d+", response)[0])
            probabilities[answer.symbol] = probability
        # normalize probabilities
        total = sum(probabilities.values())
        for answer in probabilities:
            probabilities[answer] /= total
        return self.TranscriptItem(probabilities=probabilities)        


class COTJudge(Judge):
    """Judge with chain-of-thought prompting."""
    
    @dataclass
    class TranscriptItem(Judge.TranscriptItem):
        cot: str
        
        def __str__(self):
            return f"\n\n### CHAIN OF THOUGHT:\n{self.cot}\n\n" + super().__str__()
    
    cot_prompt = (
        "I want you to give me your most informed judgment on which answer "
        "is correct, based on any information I provide. \n\n"
        "{transcript}\n\n"
        "Based on this, which answer do you think is correct?\n\n"
        "Your chain-of-thought reasoning: "
    )
    prompt = (
        "I want you to give me your most informed judgment on which answer "
        "is correct, based on any information I provide. Your response "
        "should simply be the option symbol (e.g. A, B) of one of the items "
        "from POSSIBLE ANSWERS, and nothing else.\n\n"
        "{transcript}\n\n"
        "Your chain-of-thought reasoning: {cot}\n\n"
        "Final Answer: "
    )
    
    def __call__(self, transcript: Transcript) -> "Self.TranscriptItem":
        cot = get_llm_response(
            prompt=self.cot_prompt.format(transcript=transcript),
            model=self.model,
            max_tokens=2048,
        )
        words_in_mouth = " I judge that the answer is:\n\n("
        probabilities = get_llm_response(
            prompt=self.prompt.format(transcript=transcript, cot=cot),
            model=self.model,
            return_probs_for=transcript.question.possible_answer_symbols,
            max_tokens=max(
                len(answer_symbol)
                for answer_symbol in transcript.question.possible_answer_symbols
            ),
            words_in_mouth=words_in_mouth,
        )
        return COTJudge.TranscriptItem(probabilities=probabilities, cot=cot)
    
class COTJustAskProbabilityJudge(COTJudge):
    """Judge with chain-of-thought prompting."""
    
    cot_prompt = (
        "I will give you a question to answer, and a list of possible "
        "answers. I want you to give me your most informed judgment on which "
        "answer is correct, based on any information I provide. \n\n"
        "{transcript}\n\n"
        "Based on this, which answer do you think is correct?\n\n"
        "Your chain-of-thought reasoning: "
    )
    prompt = (
        "I will give you a question to answer, and a list of possible "
        "answers. I want you to give me your most informed judgment on which "
        "answer is correct, based on any information I provide. Your response "
        "should simply give your credence/probability for a possible answer.\n\n"
        "{transcript}\n\n"
        "Your chain-of-thought reasoning: {cot}\n\n"
        "Final numerical (NOT percentage) probability for {answer}: "
    )
    
    def __call__(self, transcript: Transcript) -> "Self.TranscriptItem":
        cot = get_llm_response(
            prompt=self.cot_prompt.format(transcript=transcript),
            model=self.model,
            max_tokens=2048,
        )
        probabilities = {}
        for answer in transcript.question.possible_answers:
            words_in_mouth = f" The probability that the answer is {answer.symbol} is:\n\n"
            response = get_llm_response(
                prompt=self.prompt.format(transcript=transcript, answer=answer, cot=cot),
                model=self.model,
                max_tokens=5,
                words_in_mouth=words_in_mouth,
            )
            # extract numeric value from response
            # HACK. Maybe we should use instructor instead of regex
            probability = float(re.findall("\d+\.\d+", response)[0])
            probabilities[answer.symbol] = probability
        # normalize probabilities
        total = sum(probabilities.values())
        for answer in probabilities:
            probabilities[answer] /= total
        return COTJudge.TranscriptItem(probabilities=probabilities, cot=cot)
    
class HumanJudge(Judge):
    """A human judge that can be used in a protocol. It is not a real judge, but
    a placeholder for a human judge. It will simply ask the user for probabilities
    for each possible answer."""
    
    def __call__(self, transcript: Transcript) -> "Self.TranscriptItem":
        probabilities = {}
        print(transcript)
        for answer in transcript.question.possible_answers:
            probability = float(input(f"What is your probability that the answer is {answer.symbol}? "))
            probabilities[answer.symbol] = probability
        # normalize probabilities
        total = sum(probabilities.values())
        for answer in probabilities:
            probabilities[answer] /= total
        return self.TranscriptItem(probabilities=probabilities)

class RandomJudge(Judge):
    """A random judge that can be used in a protocol. It will randomly assign
    probabilities to each possible answer."""
    
    def __call__(self, transcript: Transcript) -> "Self.TranscriptItem":
        probabilities = {}
        for answer in transcript.question.possible_answers:
            probabilities[answer.symbol] = random()
        # normalize probabilities
        total = sum(probabilities.values())
        for answer in probabilities:
            probabilities[answer] /= total
        return self.TranscriptItem(probabilities=probabilities)
