import logging
from solib.datatypes import Question, Answer, Prob
from solib.protocols.judges.JustAskProbabilityJudge import JustAskProbabilityJudge
from pydantic import BaseModel, Field

LOGGER = logging.getLogger(__name__)

class ProbResponse(BaseModel):
    """Response model for probability judgments."""
    probabilities: dict[str, Prob] = Field(description="Probabilities for each answer option")

class JustAskProbabilitiesJudge(JustAskProbabilityJudge):
    """Like JustAskProbabilityJudge, but instead of asking for probabilities for each
    answer separately, we ask for probabilities for all answers at once."""

    prompt_file = "just_ask_both_judge.jinja"

    async def __call__(
        self, question: Question, context: str, cache_breaker: int = 0
    ) -> Question:
        """Returns a .is_elicited Question."""
        prompt = self.prompt_template.render(
            question=question.to_prompt(),
            context=context,
        )

        response = await self.get_response(
            prompt=prompt,
            response_model=ProbResponse,
            max_tokens=100,
            cache_breaker=cache_breaker,
            temperature=0.4,
        )

        LOGGER.debug(f"response: {response}")

        # Convert response into Answer objects with probabilities
        result = Question(
            question=question.question,
            answer_cases=[
                Answer.model_validate(
                    a.model_dump()
                    | {"judge_prob": response.probabilities[a.short]}
                )
                for a in question.answer_cases
            ],
            transcript=question.transcript,
        )

        assert result.is_elicited
        return result.normalize_probs()