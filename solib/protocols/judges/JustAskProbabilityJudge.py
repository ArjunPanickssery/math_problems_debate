import logging
from solib.llm_utils import parallelized_call
from solib.datatypes import Question_stripped, Answer, Prob
from solib.tool_use.tool_rendering import TRUST_TOOL_USE_PROMPT
from solib.protocols.abstract import Judge

logger = logging.getLogger(__name__)


class JustAskProbabilityJudge(Judge):
    """Instead of asking a judge to make a binary choice and getting the token probabilities,
    we just ask the judge to give probabilities. We will ask it for probabilities for each
    possible answer separately, and then we will normalize them."""

    async def __call__(
        self,
        question: Question_stripped,
        context: str,
    ) -> dict[Answer, Prob]:
        async def get_prob(answer_case: Answer) -> Prob:
            words_in_mouth = self.words_in_mouth.format(answer_case=answer_case.short)
            prompt = self.prompt.format(
                question=question.to_prompt(),
                context=context,
                answer_case=answer_case.short,
            )
            response = await self.get_response_async(
                prompt=prompt,
                response_model=Prob,
                max_tokens=20,
                words_in_mouth=words_in_mouth,
            )
            return response

        probs_ = await parallelized_call(get_prob, question.answer_cases)
        total = sum(p.prob for p in probs_)  # for normalization
        return {a: Prob(p.prob / total) for a, p in zip(question.answer_cases, probs_)}

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
        self.words_in_mouth = words_in_mouth or self.words_in_mouth
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

    words_in_mouth = " The probability that the answer is {answer_case} is:\n\n"
