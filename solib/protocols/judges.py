class TipOfTongueJudge(Judge):

    async def get_probabilities(self, context: str, **kwargs) -> "Self.TranscriptItem":
        words_in_mouth = " I judge that the answer is:\n\n("
        probabilities = await get_llm_probs_async(
            prompt=self.prompt.format(context=context),
            model=self.model,
            return_probs_for=transcript.question.answer_cases_short,
            words_in_mouth=words_in_mouth,
            **kwargs,
        )
        return self.TranscriptItem(probabilities=probabilities)

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
        "I want you to give me your most informed judgment on which answer "
        "is correct, based on any information I provide. Your response "
        "should simply be the option symbol (e.g. A, B) of one of the items "
        "from POSSIBLE ANSWERS, and nothing else. Please note that anything "
        f"that is included in {TOOL_RESULT_START_TAG}{TOOL_RESULT_END_TAG} "
        "tags can be guaranteed to be true.\n\n"
        "{transcript}\n\n"
    )
