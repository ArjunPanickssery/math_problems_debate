import logging
from solib.datatypes import Question, Answer, Prob
from solib.protocols.abstract import Judge
from solib.utils.llm_utils import jinja_env

LOGGER = logging.getLogger(__name__)


class TipOfTongueJudge(Judge):
    """TipOfTongueJudge's probabilities are extracted by asking it what the correct
    answer is and extracting the probability of it answering each option from the logits
    layer of the LLM."""

    async def __call__(
        self,
        question: Question,
        context: str,
        cache_breaker: int = 0,
    ) -> Question:
        prompt = self.prompt_template.render(
            question=question.to_prompt(),
            context=context,
            answer_cases_short=", ".join(f"({a})" for a in question.answer_cases_short),
        )

        probs = await self.get_probs(
            prompt=prompt,
            return_probs_for=question.answer_cases_short,
            words_in_mouth=self.words_in_mouth,
            cache_breaker=cache_breaker,
            temperature=0.0,
        )

        result = Question(
            question=question.question,
            answer_cases=[
                Answer.model_validate(
                    a.model_dump() | {"judge_prob": Prob(prob=probs[a.short])}
                )
                for a in question.answer_cases
            ],
            transcript=question.transcript,
        )
        assert result.is_elicited
        assert result.is_normalized
        return result

    def __init__(
        self,
        model: str = None,
        tools: list[callable] | None = None,
        hf_quantization_config=None,
        prompt_file: str = "tot_judge.jinja",
        words_in_mouth: str = None,
    ):
        """Initialize basic / default Judge. Can be overriden in subclasses.

        Args:
            prompt (str): prompt for the judge. Default None.
            model (str): model for the judge. Default None.
        """

        self.prompt_template = jinja_env.get_template(prompt_file)

        if model.startswith(
            "hf:"
        ):  # only allow hf models to default to using words_in_mouth
            self.words_in_mouth = words_in_mouth or " The answer is:\n\n("
        else:
            self.words_in_mouth = words_in_mouth

        self.dict = {
            "model": model,
            "tools": tools,
            "prompt": jinja_env.get_source(prompt_file),
            "words_in_mouth": self.words_in_mouth,
        }
        super().__init__(
            model=model, tools=tools, hf_quantization_config=hf_quantization_config
        )
