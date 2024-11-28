import functools
from solib.datatypes import Question, Answer, TranscriptItem
from solib.protocols.abstract import Protocol, QA_Agent, Judge
from solib.globals import MAX_WORDS, jinja_env

# LOGGER = logging.get#LOGGER(__name__)


class Consultancy(Protocol):
    def __init__(
        self,
        consultant_prompt: str = None,
        client_prompt: str = None,
        num_turns: int = 2,
        consultant_goes_first: bool = False,
    ):
        self.consultant_prompt = consultant_prompt or self.load_consultant_prompt()
        self.client_prompt = client_prompt or self.load_client_prompt()
        self.num_turns = num_turns
        self.consultant_goes_first = consultant_goes_first
        super().__init__(
            consultant_prompt=self.consultant_prompt,
            client_prompt=self.client_prompt,
            num_turns=num_turns,
            consultant_goes_first=consultant_goes_first,
        )

    def load_consultant_prompt(self):
        template = jinja_env.get_template("consultant.jinja")
        return template.render(MAX_WORDS=MAX_WORDS)

    def load_client_prompt(self):
        template = jinja_env.get_template("client.jinja")
        return template.render(MAX_WORDS=MAX_WORDS)

    def end_communication(self, question: Question) -> bool:
        """Default end_communication method: return True when consultancy transcript
        exceeds length num_turns. Override in subclass."""
        return (
            question.transcript is not None
            and len(question.transcript) >= self.num_turns
        )

    def tsitem_to_prompt(self, item: TranscriptItem) -> str:
        if item.role == "client":
            return f"### Enquiry from judge:\n{item.content}\n"
        else:
            return super().tsitem_to_prompt(item)

    async def step(
        self,
        agent: QA_Agent,
        question: Question,
        answer_case: Answer,
        judge: Judge,
        cache_breaker: int = 0,
        temperature: float = 0.4,
    ) -> Question:
        consultant = functools.partial(
            agent,
            prompt=self.consultant_prompt,
            question=question,
            answer_case=answer_case,
            cache_breaker=cache_breaker,
            temperature=temperature,
        )
        client_agent = QA_Agent(
            model=judge.model,
            tools=judge.tools,
            hf_quantization_config=judge.hf_quantization_config,
        )
        client = functools.partial(
            client_agent,
            prompt=self.client_prompt,
            question=question,
            cache_breaker=cache_breaker,
            temperature=temperature,
        )
        if (question.transcript in [None, []] and self.consultant_goes_first) or (
            question.transcript and question.transcript[-1].role == "client"
        ):
            cons_resp = await consultant(context=self.ts_to_prompt(question))
            question.append(TranscriptItem(role=answer_case.short, content=cons_resp))
        else:
            client_resp = await client(context=self.ts_to_prompt(question))
            question.append(TranscriptItem(role="client", content=client_resp))
        return question
