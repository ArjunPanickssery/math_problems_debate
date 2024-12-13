import logging
from solib.datatypes import Question, Answer, Prob
from solib.protocols.judges.JustAskProbabilityJudge import JustAskProbabilityJudge
from pydantic import BaseModel, Field, create_model, field_validator

LOGGER = logging.getLogger(__name__)


class JustAskProbabilitiesJudge(JustAskProbabilityJudge):
    """Like JustAskProbabilityJudge, but instead of asking for probabilities for each
    answer separately, we ask for probabilities for all answers at once, using
    instructor to ask for a response in a response_model."""

    prompt_file = "just_ask_both_judge.jinja"

    async def __call__(
        self, question: Question, context: str, cache_breaker: int = 0
    ) -> Question:
        """Returns a .is_elicited Question."""
        # Create dynamic response model with fields for each answer's probability
        fields = {}
        for answer in question.answer_cases:
            fields[answer.short] = (
                float,
                Field(
                    default=...,
                    validation_alias=answer.short,
                    json_schema_extra={
                        "minimum": 0.0,
                        "maximum": 1.0,
                    }
                )
            )
        
        # Create the model class
        response_model = create_model(
            "ProbResponse",
            **fields
        )

        # Add validators after model creation
        for answer in question.answer_cases:
            @field_validator(answer.short, mode="after")
            def validate_prob(cls, v, field):
                return Prob(prob=v).prob

            setattr(response_model, f"validate_{answer.short}", classmethod(validate_prob))

        prompt = self.prompt_template.render(
            question=question.to_prompt(),
            context=context,
        )

        response = await self.get_response(
            prompt=prompt,
            response_model=response_model,
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
                    | {"judge_prob": Prob(prob=getattr(response, a.short))}
                )
                for a in question.answer_cases
            ],
            transcript=question.transcript,
        )

        assert result.is_elicited
        return result.normalize_probs()