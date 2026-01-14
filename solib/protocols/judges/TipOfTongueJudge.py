import logging
from pathlib import Path
from solib.datatypes import Prob, Question, Answer
from solib.protocols.abstract import Judge
from solib.utils.llm_utils import jinja_env

LOGGER = logging.getLogger(__name__)


class TipOfTongueJudge(Judge):
    """TipOfTongueJudge's probabilities are extracted by asking it what the correct
    answer is and extracting the probability of it answering each option from the logits
    layer of the LLM."""

    def __init__(
        self,
        model: str = None,
        tools: list[callable] | None = None,
        user_prompt_file: str = "judges/tot_judge.jinja",
        words_in_mouth: str | None = None,
    ):
        super().__init__(model=model, tools=tools)
        self.user_template = jinja_env.get_template(user_prompt_file)

        # Only allow HF models to default to using words_in_mouth
        if model and model.startswith("hf:"):
            self.words_in_mouth = words_in_mouth or " The answer is:\n\n("
        else:
            self.words_in_mouth = words_in_mouth

        self.dict = {
            "model": model,
            "tools": tools,
            "user_prompt": jinja_env.get_source(user_prompt_file),
            "words_in_mouth": self.words_in_mouth,
        }

    async def __call__(
        self,
        question: Question,
        context: str | None = None,
        write: Path | str | None = None,
        cache_breaker: str | int | None = None,
    ) -> Question:
        prompt_content = self.user_template.render(
            question=question.to_prompt(),
            context=context,
        )
        messages = [
            {
                "role": "user",
                "content": prompt_content,
            },
        ]

        probs = await self.get_probs(
            return_probs_for=[a.short for a in question.answer_cases],
            messages=messages,
            words_in_mouth=self.words_in_mouth,
            write=write,
            cache_breaker=cache_breaker,
        )

        # Format the prompt for display
        judge_prompt = f"=== USER ===\n{prompt_content}"

        result = Question(
            question=question.question,
            answer_cases=[
                Answer.model_validate(
                    a.model_dump() | {"judge_prob": Prob(prob=probs[a.short])}
                )
                for a in question.answer_cases
            ],
            transcript=question.transcript,
            judge_prompt=judge_prompt,
        )
        assert result.is_elicited
        return result
