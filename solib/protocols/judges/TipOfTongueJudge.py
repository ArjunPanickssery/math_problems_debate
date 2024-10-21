import logging
from solib.datatypes import Question, Answer, Prob
from solib.tool_use.tool_rendering import TRUST_TOOL_USE_PROMPT
from solib.protocols.abstract import Judge

logger = logging.getLogger(__name__)


class TipOfTongueJudge(Judge):
    """TipOfTongueJudge's probabilities are extracted by asking it what the correct
    answer is and extracting the probability of it answering each option from the logits
    layer of the LLM."""

    async def __call__(
        self,
        question: Question,
        context: str,
    ) -> Question:

        prompt = self.prompt.format(
            question=question.to_prompt(),
            context=context,
            answer_cases_short=", ".join(f"({a})" for a in question.answer_cases_short),
        )
        probs = await self.get_probs_async(
            prompt=prompt,
            return_probs_for=question.answer_cases_short,
            words_in_mouth=self.words_in_mouth,
        )
        result = Question(
            question=question.question,
            answer_cases=[
                Answer.model_validate(
                    a.model_dump() | {"judge_prob": Prob(prob=probs[a.short])}
                )
                for a in question.answer_cases
            ],
        )
        assert result.is_elicited
        assert result.is_normalized
        return result

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
        "I want you to give me your most informed judgment on which answer "
        "is correct, based on any information I provide. Your response "
        "should simply be the option symbol of one of the items from "
        "POSSIBLE ANSWERS, i.e. {answer_cases_short}, and nothing else. "
        f"{TRUST_TOOL_USE_PROMPT}\n\n"
        "{context}\n\n"
    )

    words_in_mouth = " The answer is:\n\n("
