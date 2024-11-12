import logging
from solib.utils import parallelized_call
from solib.datatypes import Question, Answer, Prob
from solib.tool_use.tool_rendering import TRUST_TOOL_USE_PROMPT
from solib.protocols.abstract import Judge

#LOGGER = logging.get#LOGGER(__name__)


class JustAskProbabilityJudge(Judge):
    """Instead of asking a judge to make a binary choice and getting the token probabilities,
    we just ask the judge to give probabilities. We will ask it for probabilities for each
    possible answer separately, and then we will normalize them."""

    async def __call__(
        self,
        question: Question,
        context: str,
        cache_breaker: int = 0,
    ) -> Question:
        # we don't pass in temperature here since ToT Judge always uses 0.0, and
        # we don't distinguish between judge type in the code
        """Returns a .is_elicited Question."""

        async def get_prob(answer_case: Answer) -> Prob:
            prompt = self.prompt.format(
                question=question.to_prompt(),
                context=context,
                answer_case=answer_case.short,
            )
            response = await self.get_response(
                prompt=prompt,
                response_model=Prob,
                max_tokens=20,
                cache_breaker=cache_breaker,
                temperature=0.4
            )

            #LOGGER.debug(f"response: {response}")
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

    def __init__(
        self,
        model: str = None,
        tools: list[callable] | None = None,
        hf_quantization_config=None,
        prompt: str = None,
        words_in_mouth: str = None,
    ):
        """Initialize basic / default Judge. Can be overriden in subclasses.

        Args:
            prompt (str): prompt for the judge. Default None.
            model (str): model for the judge. Default None.
        """
        self.prompt = prompt or self.prompt
        self.words_in_mouth = words_in_mouth
        self.dict = {
            "model": model,
            "tools": tools,
            "prompt": self.prompt,
            "words_in_mouth": self.words_in_mouth,
        }
        super().__init__(
            model=model, tools=tools, hf_quantization_config=hf_quantization_config
        )

    prompt = (
        "{question}\n\n"
        "I want you to give me your most informed judgment on whether the answer "
        "is {answer_case}. Your response should simply be your credence/probability "
        "for the answer {answer_case}, expressed as a decimal number between 0 and 1, "
        "and nothing else. "
        f"{TRUST_TOOL_USE_PROMPT}\n\n"
        "{context}"
    )
