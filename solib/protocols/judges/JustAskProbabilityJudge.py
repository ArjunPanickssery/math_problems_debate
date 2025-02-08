import logging
from pathlib import Path
from solib.datatypes import Question, Answer, Prob
from solib.utils import parallelized_call
from solib.protocols.abstract import Judge
from solib.utils.llm_utils import jinja_env

LOGGER = logging.getLogger(__name__)


class JustAskProbabilityJudge(Judge):
    """Instead of asking a judge to make a binary choice and getting the token probabilities,
    we just ask the judge to give probabilities. We will ask it for probabilities for each
    possible answer separately, and then we will normalize them."""

    def __init__(
        self,
        model: str = None,
        tools: list[callable] | None = None,
        user_prompt_file: str = "judges/just_ask_judge.jinja",
    ):
        """Initialize basic / default Judge. Can be overriden in subclasses.

        Args:
            system_prompt_file (str): file for the system prompt.
            user_prompt_file (str): file for the user prompt.
            model (str): model for the judge. Default None.
        """
        super().__init__(model=model, tools=tools)
        self.user_template = jinja_env.get_template(user_prompt_file)
        self.dict = {
            "model": model,
            "tools": tools,
            "user_prompt": jinja_env.get_source(user_prompt_file),
        }

    async def __call__(
        self,
        question: Question,
        context: str,
        caching: bool = None,
        write: Path | str | None = None,
    ) -> Question:
        # we don't pass in temperature here since ToT Judge always uses 0.0, and
        # we don't distinguish between judge type in the code
        """Returns a .is_elicited Question."""

        async def get_prob(answer_case: Answer) -> Prob:
            messages = [
                {
                    "role": "user",
                    "content": self.user_template.render(
                        question=question.to_prompt(),
                        answer_case=answer_case.short,
                        context=context,
                    ),
                },
            ]

            response = await self.get_response(
                messages=messages,
                response_model=Prob,
                max_tokens=20,
                caching=caching,
                temperature=0.4,
                write=write,
            )

            LOGGER.debug(f"response: {response}")
            return response

        probs_ = await parallelized_call(get_prob, question.answer_cases)
        result = Question(
            question=question.question,
            answer_cases=[
                Answer.model_validate(a.model_dump() | {"judge_prob": p})
                for a, p in zip(question.answer_cases, probs_)
            ],
            transcript=question.transcript,
        )
        assert result.is_elicited
        return result.normalize_probs()
