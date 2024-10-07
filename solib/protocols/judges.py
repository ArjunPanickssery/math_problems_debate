from solib.llm_utils import (
    get_llm_response,
    get_llm_response_async,
    get_llm_probs,
    get_llm_probs_async,
    parallelized_call,
)
from solib.datatypes import Question, Question_stripped, Answer, Prob
from solib.tool_use.tool_rendering import TOOL_RESULT_START_TAG, TOOL_RESULT_END_TAG
from solib.protocols.abstract import Judge

TRUST_TOOL_USE_PROMPT = (
    "Please note that anything that is included in "
    f"{TOOL_RESULT_START_TAG}{TOOL_RESULT_END_TAG} "
    "tags is guaranteed to be true."
)


class TipOfTongueJudge(Judge):
    """TipOfTongueJudge's probabilities are extracted by asking it what the correct
    answer is and extracting the probability of it answering each option from the logits
    layer of the LLM."""

    async def __call__(
        self,
        question: Question_stripped,
        context: str,
    ) -> dict[Answer, Prob]:
        words_in_mouth = " The answer is:\n\n("
        prompt = self.prompt.format(
            question=question.to_prompt(),
            context=context,
            answer_cases_short=", ".join(f"({a})" for a in question.answer_cases_short),
        )
        probs = await get_llm_probs_async(
            prompt=prompt,
            model=self.model,
            return_probs_for=question.answer_cases_short,
            words_in_mouth=words_in_mouth,
        )
        return {
            answer_case: Prob(probs[answer_case.short])
            for answer_case in question.answer_cases
        }

    def __init__(self, prompt: str = None, model: str = None):
        """Initialize basic / default Judge. Can be overriden in subclasses.

        Args:
            prompt (str): prompt for the judge. Default None.
            model (str): model for the judge. Default None.
        """
        if prompt is not None:
            self.prompt = prompt
        self.model = model

    prompt = (
        "{question}\n\n"
        "I want you to give me your most informed judgment on which answer "
        "is correct, based on any information I provide. Your response "
        "should simply be the option symbol of one of the items from "
        "POSSIBLE ANSWERS, i.e. {answer_cases_short}, and nothing else. "
        f"{TRUST_TOOL_USE_PROMPT}\n\n"
        "{context}\n\n"
    )


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
            words_in_mouth = (
                f" The probability that the answer is {answer_case.short} is:\n\n"
            )
            prompt = self.prompt.format(
                question=question.to_prompt(),
                context=context,
                answer_case=answer_case.short,
            )
            response = await get_llm_response_async(
                prompt=prompt,
                model=self.model,
                response_model=Prob,
                max_tokens=20,
                words_in_mouth=words_in_mouth,
            )
            return response

        probs_ = await parallelized_call(get_prob, question.answer_cases)
        total = sum(p.prob for p in probs_)  # for normalization
        return {a: Prob(p.prob / total) for a, p in zip(question.answer_cases, probs_)}

    def __init__(self, prompt: str = None, model: str = None):
        """Initialize basic / default Judge. Can be overriden in subclasses.

        Args:
            prompt (str): prompt for the judge. Default None.
            model (str): model for the judge. Default None.
        """
        if prompt is not None:
            self.prompt = prompt
        self.model = model

    prompt = (
        "{question}\n\n"
        "I want you to give me your most informed judgment on whether the answer "
        "is {answer_case}. Your response should simply be your credence/probability "
        "for the answer {answer_case}, expressed as a decimal number between 0 and 1, "
        "and nothing else. "
        f"{TRUST_TOOL_USE_PROMPT}\n\n"
        "{context}"
    )
