import logging
from pathlib import Path
from solib.datatypes import Question, Prob, Answer
from solib.protocols.judges.JustAskProbabilityJudge import JustAskProbabilityJudge
from pydantic import BaseModel, Field

LOGGER = logging.getLogger(__name__)


class ProbResponse(BaseModel):
    """Response model for probability judgments."""

    probabilities: dict[str, Prob] = Field(
        description="Probabilities for each answer option. Provide it as a Prob object."
    )


class JustAskProbabilitiesJudge(JustAskProbabilityJudge):
    """Like JustAskProbabilityJudge, but instead of asking for probabilities for each
    answer separately, we ask for probabilities for all answers at once."""

    prompt_file = "judges/just_ask_both_judge.jinja"

    async def __call__(
        self,
        question: Question,
        context: str,
        write: Path | str | None = None,
        cache_breaker: str | int | None = None,
    ) -> Question:
        """Returns a .is_elicited Question."""
        prompt = self.prompt_template.render(
            question=question.to_prompt(),
            context=context,
        )
        messages = [{"role": "user", "content": prompt}]

        response = await self.get_response(
            messages=messages,
            response_model=ProbResponse,
            max_tokens=100,
            temperature=0.4,
            write=write,
            cache_breaker=cache_breaker,
        )

        LOGGER.debug(f"response: {response}")
        # print(response, type(response))
        # print(response.probabilities)

        # Convert response into Answer objects with probabilities
        result = Question(
            question=question.question,
            answer_cases=[
                Answer.model_validate(
                    a.model_dump() | {"judge_prob": response.probabilities[a.short]}
                )
                for a in question.answer_cases
            ],
            transcript=question.transcript,
        )

        assert result.is_elicited
        return result.normalize_probs()
