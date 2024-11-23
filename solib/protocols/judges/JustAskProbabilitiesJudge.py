import logging
from solib.datatypes import Question, Answer, Prob
from solib.tool_use.tool_rendering import TRUST_TOOL_USE_PROMPT
from solib.protocols.judges.JustAskProbabilityJudge import JustAskProbabilityJudge
from pydantic import BaseModel, field_validator

LOGGER = logging.getLogger(__name__)


class JustAskProbabilitiesJudge(JustAskProbabilityJudge):
    """Like JustAskProbabilityJudge, but instead of asking for probabilities for each
    answer separately, we ask for probabilities for all answers at once, using
    instructor to ask for a response in a response_model."""

    async def __call__(
        self, question: Question, context: str, cache_breaker: int = 0
    ) -> Question:
        """Returns a .is_elicited Question."""
        # Create dynamic response model with fields for each answer's probability
        response_model = type(
            "ProbResponse",
            (BaseModel,),
            {
                answer.short: (float, field_validator(answer.short)(
                    classmethod(lambda cls, v: Prob(prob=v).prob)
                ))
                for answer in question.answer_cases
            }
        )

        prompt = self.prompt.format(
            question=question.to_prompt(),
            context=context,
        )
        
        response = await self.get_response(
            prompt=prompt,
            response_model=response_model,
            max_tokens=100,
            cache_breaker=cache_breaker,
            temperature=0.4
        )

        LOGGER.debug(f"response: {response}")

        # Convert response into Answer objects with probabilities
        result = Question(
            question=question.question,
            answer_cases=[
                Answer.model_validate(a.model_dump() | {"judge_prob": Prob(prob=getattr(response, a.short))})
                for a in question.answer_cases
            ],
            transcript=question.transcript,
        )
        
        assert result.is_elicited
        return result.normalize_probs()
    
    prompt = (
        "{question}\n\n"
        "I want you to give me your most informed judgment on the probability of each "
        "possible answer. Your response should be a JSON object with fields matching "
        "the answer codes, where each field contains your credence/probability for that "
        "answer expressed as a decimal number between 0 and 1. The probabilities should "
        "sum to 1.\n"
        f"{TRUST_TOOL_USE_PROMPT}\n\n"
        "{context}"
    )
