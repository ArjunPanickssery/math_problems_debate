import json
import logging
import copy
import numpy as np
from typing import Any, Literal, Optional, Union, Callable, Self
from pydantic import BaseModel, field_validator, computed_field, model_validator, Field

LOGGER = logging.getLogger(__name__)


class Prob(BaseModel):
    prob: float = Field(description="Probability between 0 and 1.")

    @field_validator("prob")
    @classmethod
    def validate_prob(cls, v):
        assert 0.0 <= v <= 1.0, "Probability must be between 0 and 1."
        return v

    def __float__(self):
        return self.prob

    def __add__(self, other):
        return float(self) + float(other)

    def __mul__(self, other):
        return float(self) * float(other)

    def __truediv__(self, other):
        return float(self) / float(other)

    def __sub__(self, other):
        return float(self) - float(other)

    def __neg__(self):
        return -float(self)

    def __radd__(self, other):
        return float(other) + float(self)

    def __rmul__(self, other):
        return float(other) * float(self)

    def __rtruediv__(self, other):
        return float(other) / float(self)

    def __rsub__(self, other):
        return float(other) - float(self)

    def __array__(self):
        return np.array(float(self))

    def __lt__(self, other):
        return float(self) < float(other)

    def __le__(self, other):
        return float(self) <= float(other)

    def __gt__(self, other):
        return float(self) > float(other)

    def __ge__(self, other):
        return float(self) >= float(other)

    def __eq__(self, other):
        return float(self) == float(other)

    def pad(self, eps: float = 1e-4) -> "Prob":
        return Prob(prob=min(max(self.prob, eps), 1 - eps))


class Score(BaseModel):
    """
    Data class to store scores based on different metrics.
    """

    log: float | None = None
    brier: float | None = None
    logodds: float | None = None
    accuracy: float | None = None

    @classmethod
    def calc(
        cls, question: "Question", answer_case: "Answer", eps: float = 1e-4
    ) -> Optional["Score"]:
        """Score for answer_case based on probability assigned to it in Question."""
        # .censor() because e.g. question might be answer_case.case_probs, so
        # question.answer_cases will have probabilities while answer_case would not
        assert answer_case.censor() in question.censor().answer_cases
        assert question.is_elicited
        if not all(a.judge_prob is not None for a in question.answer_cases):
            return None
        if not question.is_normalized:
            LOGGER.warning("Calculating score based on unnormalized probabilities.")

        # find answer case in question
        answer_case_in_question = question.answer_cases_dict[answer_case.short]

        score_log = np.log(answer_case_in_question.judge_prob.pad(eps))
        score_brier = -np.sum(
            [
                (int(a.short == answer_case.short) - a.judge_prob) ** 2
                for a in question.answer_cases
            ]
        )
        score_logodds = np.log(answer_case_in_question.judge_prob.pad(eps)) - np.log(
            question.neg(answer_case_in_question).judge_prob.pad(eps)
        )
        score_accuracy = float(
            answer_case_in_question.judge_prob.pad(eps)
            > question.neg(answer_case_in_question).judge_prob.pad(eps)
        )
        return Score(
            log=score_log,
            brier=score_brier,
            logodds=score_logodds,
            accuracy=score_accuracy,
        )

    # Helper method for pointwise operations
    def _operate(self, other, op) -> "Score":
        # Function to apply an operation between two values
        def apply(x, y):
            if x is None or y is None:
                return None
            return op(x, y)

        if isinstance(other, Score):
            return Score(
                log=apply(self.log, other.log),
                brier=apply(self.brier, other.brier),
                logodds=apply(self.logodds, other.logodds),
                accuracy=apply(self.accuracy, other.accuracy),
            )
        elif isinstance(other, (int, float)):  # Scalar operations
            return Score(
                log=apply(self.log, other),
                brier=apply(self.brier, other),
                logodds=apply(self.logodds, other),
                accuracy=apply(self.accuracy, other),
            )
        else:
            return NotImplemented

    # Implement arithmetic magic methods
    def __add__(self, other):
        return self._operate(other, lambda x, y: x + y)

    def __sub__(self, other):
        return self._operate(other, lambda x, y: x - y)

    def __mul__(self, other):
        return self._operate(other, lambda x, y: x * y)

    def __truediv__(self, other):
        return self._operate(other, lambda x, y: x / y)

    def __floordiv__(self, other):
        return self._operate(other, lambda x, y: x // y)

    def __pow__(self, other):
        return self._operate(other, lambda x, y: x**y)

    # Implement reversed operators for commutative operations
    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    # helper function for array operations
    @classmethod
    def _arroperate(cls, scores: list["Score"], op) -> "Score":
        return Score(
            log=op([score.log for score in scores]),
            brier=op([score.brier for score in scores]),
            logodds=op([score.logodds for score in scores]),
            accuracy=op([score.accuracy for score in scores]),
        )

    @classmethod
    def mean(cls, scores: list["Score"]) -> "Score":
        return cls._arroperate(scores, np.mean)

    @classmethod
    def std(cls, scores: list["Score"]) -> "Score":
        return cls._arroperate(scores, np.std)

    def sqrt(self) -> "Score":
        """Return square root of each score metric."""
        return Score(
            log=None if self.log is None else np.sqrt(self.log),
            brier=None if self.brier is None else np.sqrt(self.brier),
            logodds=None if self.logodds is None else np.sqrt(self.logodds),
            accuracy=None if self.accuracy is None else np.sqrt(self.accuracy)
        )



class Stats(BaseModel):
    asd_mean: Score
    asd_std: Score
    jse_b0_mean: Score
    jse_b0_std: Score
    jse_b1_mean: Score
    jse_b1_std: Score
    jse_binf_mean: Score
    jse_binf_std: Score
    ase_b0_mean: Score
    ase_b0_std: Score
    ase_b1_mean: Score
    ase_b1_std: Score
    ase_binf_mean: Score
    ase_binf_std: Score
    # Verification stats (optional, only present if verification was enabled)
    verification_enabled: Optional[bool] = None
    verification_accepted_1try_rate: Optional[float] = None
    verification_accepted_2tries_rate: Optional[float] = None
    verification_accepted_3tries_rate: Optional[float] = None
    verification_never_accepted_rate: Optional[float] = None
    verification_avg_tries: Optional[float] = None


class Answer(BaseModel):
    """
    Generally only found as an item of a Question object.

    The Answer class may be used for the following purposes.

    ## `Answer.is_censored`
    To store a potential answer, with no indicator of whether it is true or not.
    Nothing else must be provided.

    ## `Answer.is_grounded`
    To store an answer, including its "value" (e.g. 1.0 for true, -1.0 for false).
    Additional attributes in this case:
        value (float): must be provided.

    ## `Answer.is_elicited`
    To store an answer and an elicited probability for it.
    Additional attributes in this case:
        judge_prob (Prob): must be provided.

    ## `Answer.is_argued`
    To store an answer, and some elicited probabilities for _all answers_ after
    "hearing an argument for this answer".
    Additional attributes in this case:
        case_probs (Question): must be provided. Should be a Question that is .is_elicited.
        agent_score (Score): computed. Measures how much the case_probs seem to be
            convinced by this answer.

    Attributes in all circumstances:
        short (str): must be provided.
        long (str): must be provided.

    Methods in all circumstances:
        censor() -> Answer: converts any Answer into the censored form. This is necessary
            before showing anything to an AI.
        uncensor(grounded: "Answer" | "Question") -> Answer: reattaches a value
            from a grounded float, Answer, or Question.
        to_prompt() -> str: a prompt representation of an Answer, containing only its censored
            form
    """

    short: str
    long: str
    value: Optional[float] = None
    judge_prob: Optional[Prob] = None
    case_probs: Optional["Question"] = None

    @property
    def id(self) -> tuple:
        return (self.short, self.long)

    @model_validator(mode="after")
    def makes_sense(self) -> Self:
        if self.is_argued:
            assert self.case_probs.is_elicited
            assert not self.case_probs.is_argued
        if self.is_elicited:
            assert not self.is_argued
        return self

    def censor(self) -> "Answer":
        return Answer(short=self.short, long=self.long)

    def uncensor(
        self,
        grounded: Union["Answer", "Question"],
    ) -> "Answer":
        # assert self.is_censored
        # # ^^we don't assert this because we still want to transfer values when
        # # self has some other attributes non-None
        # # could do this though:
        # assert self.value is None
        assert grounded.is_grounded
        if isinstance(grounded, Question):
            grounded_ans = grounded.answer_cases_dict[self.short]
            return Answer(
                short=self.short,
                long=self.long,
                value=grounded_ans.value,
                judge_prob=self.judge_prob or grounded_ans.judge_prob,
                case_probs=(
                    self.case_probs.uncensor(grounded)
                    if self.case_probs
                    else grounded_ans.case_probs
                ),
            )
        elif isinstance(grounded, Answer):
            assert (self.short, self.long) == (grounded.short, grounded.long)
            return Answer(
                short=self.short,
                long=self.long,
                value=grounded.value,
                judge_prob=self.judge_prob or grounded.judge_prob,
                case_probs=self.case_probs or grounded.case_probs,
            )
        else:
            raise TypeError(f"Invalid grounded: {grounded}")

    @property
    def is_censored(self) -> bool:
        return (
            self.value is None and self.judge_prob is None and self.case_probs is None
        )

    @property
    def is_grounded(self) -> bool:
        return self.value is not None

    @property
    def is_elicited(self) -> bool:
        return self.judge_prob is not None

    @property
    def is_argued(self) -> bool:
        return self.case_probs is not None

    @computed_field
    @property
    def item_type(self) -> list[str]:
        return [
            k
            for k, v in {
                "Answer": True,
                "is_censored": self.is_censored,
                "is_grounded": self.is_grounded,
                "is_elicited": self.is_elicited,
                "is_argued": self.is_argued,
            }.items()
            if v
        ]

    def to_prompt(self):
        return f"({self.short}): {self.long}"

    @computed_field
    @property
    def agent_score(self) -> Score:
        """Score for this answer case according to self.case_probs."""
        if not self.is_argued:
            return None
        return Score.calc(self.case_probs, self)


class Question(BaseModel):
    """
    The Question class may be used for the following purposes. (These purposes are
    inferred from the purposes of the question's answer_cases).

    ## `Question.is_censored`
    To store a question and its answer choices, with no info about ground truth.

    ## `Question.is_grounded`
    To store a question and its answer choices, each of which has a value (e.g. 1.0
    for the true answer, -1.0 for the false answer)
    Additional attributes in this case:
        true_answer: the answer_case with value = 1.0
        false_answer: the answer_case with value = -1.0
        best_answer: the answer_case with the highest value, generally same as true_answer
        worst_answer: the answer_case with the lowest value, generally same as false_answer

    ## `Question.is_elicited`
    To store a question and its answer choices, each with some elicited probability from
    the judge.
    Additional attributes in this case:
        total_prob (float): total probability of all answer cases.
        is_normalized (bool): checks if total_prob = 1.0.
        judge_score (Score): computed. Also requires .is_grounded. Scores the elicited
            probabilities against the true answer.
    Additional methods in this case:
        normalize_probs() -> Question: normalizes the probabilities so that they sum to 1.0.
            Not in-place.

    ## `Question.is_argued`
    To stores answers that are .is_argued, i.e. for each answer, store elicited probs after arguing
    for that answer.
    Additional attributes in this case:
        agent_score_diff (Score): computed. Also requires .is_grounded. Measures how much more the
            agent can convince the judge by arguing for the true answer compared to the false answer.
        judge_score_b0 (Score): computed. Also requires .is_grounded. Expected judge score with beta = 0.0.
        judge_score_b1 (Score): computed. Also requires .is_grounded. Expected judge score with beta = 1.0.
        judge_score_binf (Score): computed. Also requires .is_grounded. Expected judge score with beta = inf.
        agent_score_b0 (Score): computed. Expected agent score with beta = 0.0.
        agent_score_b1 (Score): computed. Expected agent score with beta = 1.0.
        agent_score_binf (Score): computed. Expected agent score with beta = inf.
    Additional methods in this case:
        open_probs(beta: float) -> dict[str, Prob]: Also requires .is_argued. Returns a dict mapping each answer
            case to its propensity of being argued for, given beta.
        judge_score_expected(beta: float) -> Score: Also requires .is_grounded. Scores the elicited
            probabilities against the true answer, given how likely the agent is to argue for each answer.
        agent_score_expected(beta: float) -> Score: Scores the agent's persuasiveness, given how likely the
            agent is to argue for each answer.

    ## `Question.has_transcript`
    To store a transcript en route to eliciting a probability.
    Additional attributes in this case:
        transcript (list[TranscriptItem]): must be provided / appended via .append()

    Attributes in all circumstances:
        question (str): must be provided.
        answer_cases (list[Answer]): must be provided.
        answer_cases_short (list[str]): calculated.
        answer_cases_dict (dict[str, Answer]): calculated.

    Methods in all circumstances:
        censor() -> Question: converts any Question into the censored form. This is
            necessary before showing anything to an AI.
        uncensor(grounded: Question) -> Question: reattaches values from a grounded
            Question.
        neg(a: Answer) -> Answer: return the other answer.
        append(t: TranscriptItem), inplace: adds a transcript item (creating
            the transcript for the first time if necessary)
        to_prompt() -> str: a prompt representation of a Question. Note: contains only
            question and answer cases; to promptify the transcript, make your own method
            in the Protocol.

    """

    question: str
    source_text: str | None = None
    answer_cases: list[Answer]
    transcript: list["TranscriptItem"] | None = None
    judge_prompt: str | None = None  # Full prompt used by judge to generate probabilities

    # not necessary, it's enough to validate this in Answer
    # @model_validator(mode="after")
    # def makes_sense(self) -> Self:
    #     if self.is_argued:
    #         assert not self.is_elicited
    #     if self.is_elicited:
    #         assert not self.is_argued
    #     return self

    @property
    def id(self) -> tuple:
        return (self.question, tuple(a.id for a in self.answer_cases))

    def censor(self) -> "Question":
        return Question(
            question=self.question,
            source_text=self.source_text,
            answer_cases=[answer.censor() for answer in self.answer_cases],
        )

    def uncensor(self, grounded: "Question") -> "Question":
        # assert self.is_censored
        # # ^^we don't assert this because we still want to transfer values when
        # # self has some other attributes non-None
        # NOTE: it is important that we pass a.uncensor(grounded) rather than
        # a.uncensor(grounded.answer_dict[a.short]), to ensure a is also updated
        # with all the case_probs from grounded
        assert grounded.is_grounded
        return Question(
            question=self.question,
            source_text=self.source_text or grounded.source_text,
            answer_cases=[a.uncensor(grounded) for a in self.answer_cases],
            transcript=self.transcript or grounded.transcript,
        )

    @property
    def is_censored(self) -> bool:
        return all(answer.is_censored for answer in self.answer_cases)

    @property
    def is_grounded(self) -> bool:
        return all(answer.is_grounded for answer in self.answer_cases)

    @property
    def is_elicited(self) -> bool:
        return all(answer.is_elicited for answer in self.answer_cases)

    @property
    def is_argued(self) -> bool:
        return all(answer.is_argued for answer in self.answer_cases)

    @computed_field
    @property
    def item_type(self) -> list[str]:
        return [
            k
            for k, v in {
                "Question": True,
                "is_censored": self.is_censored,
                "is_grounded": self.is_grounded,
                "is_elicited": self.is_elicited,
                "is_argued": self.is_argued,
                "has_source_text": self.source_text is not None,
            }.items()
            if v
        ]

    @property
    def has_transcript(self) -> bool:
        return self.transcript is not None

    def to_prompt(self):
        """Note: this only includes the question and answer cases,
        not the transcript. Implement a method in the Protocol for
        promptifying the transcript."""
        return (
            f"QUESTION: {self.question}\n"
            + "POSSIBLE ANSWERS:\n"
            + "\n".join(answer.to_prompt() for answer in self.answer_cases)
        )

    @property
    def answer_cases_short(self) -> list[str]:
        return [answer.short for answer in self.answer_cases]

    @property
    def answer_cases_dict(self) -> dict[str, Answer]:
        return {answer.short: answer for answer in self.answer_cases}

    @property
    def answer_cases_values(self) -> list[float]:
        assert self.is_grounded
        return [answer.value for answer in self.answer_cases]

    def neg(self, answer: Answer) -> Answer:
        assert len(self.answer_cases) == 2
        # NOTE this is fine for now, but could become a problem one day
        assert answer in self.answer_cases
        ans_index = self.answer_cases.index(answer)
        return self.answer_cases[1 - ans_index]

    @property
    def best_answer(self) -> Answer:
        assert self.is_grounded
        return max(self.answer_cases, key=lambda x: x.value)

    @property
    def worst_answer(self) -> Answer:
        assert self.is_grounded
        return min(self.answer_cases, key=lambda x: x.value)

    @property
    def true_answer(self) -> Answer:
        assert self.is_grounded
        true_answers = [answer for answer in self.answer_cases if answer.value == 1.0]
        assert len(true_answers) == 1
        return true_answers[0]

    @property
    def false_answer(self) -> Answer:
        assert self.is_grounded
        false_answers = [answer for answer in self.answer_cases if answer.value == -1.0]
        assert len(false_answers) == 1
        return false_answers[0]

    @property
    def total_prob(self) -> float:
        assert self.is_elicited
        return sum(answer.judge_prob for answer in self.answer_cases)

    @property
    def is_normalized(self) -> bool:
        assert self.is_elicited
        return abs(self.total_prob - 1.0) < 1e-3

    def normalize_probs(self) -> "Question":
        assert self.is_elicited
        if self.total_prob == 0.0:
            LOGGER.info(
                f"Total probability is 0.0 for question {self.question}. "
                "Normalizing probs to 1 / len(answer_cases)."
            )
        return Question(
            question=self.question,
            answer_cases=[
                Answer(
                    short=answer.short,
                    long=answer.long,
                    judge_prob=(
                        Prob(prob=answer.judge_prob / self.total_prob)
                        if self.total_prob != 0.0
                        else Prob(prob=1 / len(self.answer_cases))
                    ),
                )
                for answer in self.answer_cases
            ],
            source_text=self.source_text,
            transcript=self.transcript,
        )

    @computed_field
    @property
    def judge_score(self) -> Score:
        """
        Score judge_probs based on ground truth
        """
        if not (self.is_elicited and self.is_grounded):
            return None
        return Score.calc(self, self.true_answer)

    @computed_field
    @property
    def agent_score_diff(self) -> Score:
        """Advantage in answering for the truth."""
        if not (self.is_argued and self.is_grounded):
            return None
        return self.true_answer.agent_score - self.false_answer.agent_score

    def open_probs(self, beta: float) -> dict[str, Prob]:
        """Propensity of the agent to argue for each answer."""
        assert self.is_argued
        max_agent_score = max(answer.agent_score.log for answer in self.answer_cases)
        if beta == 0.0:
            return {
                answer.short: (
                    Prob(prob=1.0)
                    if answer.agent_score.log == max_agent_score
                    else Prob(prob=0.0)
                )
                for answer in self.answer_cases
            }
        elif beta == np.inf:
            return {
                answer.short: Prob(prob=1 / len(self.answer_cases))
                for answer in self.answer_cases
            }
        else:
            denom = sum(
                np.exp(answer.agent_score.log / beta) for answer in self.answer_cases
            )
            return {
                answer.short: Prob(prob=np.exp(answer.agent_score.log / beta) / denom)
                for answer in self.answer_cases
            }

    def judge_score_expected(self, beta: float) -> Score:
        if not (self.is_argued and self.is_grounded):
            return None
        return sum(
            self.open_probs(beta)[answer.short].prob * answer.case_probs.judge_score
            for answer in self.answer_cases
        )

    def agent_score_expected(self, beta: float) -> Score:
        if not self.is_argued:
            return None
        return sum(
            self.open_probs(beta)[answer.short].prob * answer.agent_score
            for answer in self.answer_cases
        )

    @computed_field
    @property
    def judge_score_b0(self) -> Score:
        return self.judge_score_expected(0.0)

    @computed_field
    @property
    def judge_score_b1(self) -> Score:
        return self.judge_score_expected(1.0)

    @computed_field
    @property
    def judge_score_binf(self) -> Score:
        return self.judge_score_expected(np.inf)

    @computed_field
    @property
    def agent_score_b0(self) -> Score:
        return self.agent_score_expected(0.0)

    @computed_field
    @property
    def agent_score_b1(self) -> Score:
        return self.agent_score_expected(1.0)

    @computed_field
    @property
    def agent_score_binf(self) -> Score:
        return self.agent_score_expected(np.inf)

    def _judge_score_expected_legacy(
        self,
        agent_arguing_for: Union[
            Callable[[Answer], Prob], list[float], Literal["max", "min", "uniform"]
        ],
    ) -> Score:
        """Expected judge score given how likely an agent is to argue for each answer.
        THIS IS BASICALLY USELESS and you should use judge_score_expected instead.

        Args:
            agent_arguing_for (Callable[[Answer], Prob] | list[float] | Literal["max", "min", "uniform"]):
                Callable: function mapping answer to probability of arguing for it.
                list[float]: list of probabilities of the agent arguing for each answer, in ascending order
                    of value (i.e. false first, true last)
                Literal["max", "min", "uniform"]: shortcut for most common distributions.
        """
        assert self.is_argued and self.is_grounded
        if agent_arguing_for == "max":
            agent_arguing_for = [0.0] * (len(self.answer_cases) - 1) + [1.0]
        elif agent_arguing_for == "min":
            agent_arguing_for = [1.0] + [0.0] * (len(self.answer_cases) - 1)
        elif agent_arguing_for == "uniform":
            agent_arguing_for = [1.0 / len(self.answer_cases)] * len(self.answer_cases)

        if isinstance(agent_arguing_for, list):
            assert len(agent_arguing_for) == len(self.answer_cases)
            agent_arguing_for_func = lambda a: agent_arguing_for[  # noqa
                sorted(self.answer_cases, key=lambda x: x.value).index(a)
            ]
            # we have to name agent_arguing_for_func differently because functions are
            # evaluated lazily, so when the function is called later on it will refer back
            # to the above line and attempt to subscript it. At least I think that's what
            # was happening.
        elif isinstance(agent_arguing_for, Callable):
            agent_arguing_for_func = agent_arguing_for
        else:
            raise TypeError(f"Invalid agent_arguing_for: {agent_arguing_for}")
        return sum(
            agent_arguing_for_func(answer) * answer.case_probs.judge_score
            for answer in self.answer_cases
        )

    @staticmethod
    def compute_stats(
        questions: list["Question"],
        exclude_unverified: bool = False,
    ) -> dict[str, Any]:
        """Compute stats for a list of .is_argued Questions.

        Args:
            questions: List of argued and grounded Questions
            exclude_unverified: If True, exclude questions where any argument
                               failed verification (was not aligned)
        """
        # Helper to check if all arguments in a question are verified as aligned
        def _all_arguments_verified(q: "Question") -> bool:
            if q.transcript is None:
                return True
            for item in q.transcript:
                if item.metadata and "verification" in item.metadata:
                    if not item.metadata["verification"].get("is_aligned", True):
                        return False
            return True

        # Filter if requested
        if exclude_unverified:
            questions = [q for q in questions if _all_arguments_verified(q)]

        assert all(
            question.is_argued and question.is_grounded for question in questions
        )
        asds = [question.agent_score_diff for question in questions]
        jse_b0s = [question.judge_score_b0 for question in questions]
        jse_b1s = [question.judge_score_b1 for question in questions]
        jse_binfs = [question.judge_score_binf for question in questions]
        ase_b0s = [question.agent_score_b0 for question in questions]
        ase_b1s = [question.agent_score_b1 for question in questions]
        ase_binfs = [question.agent_score_binf for question in questions]
        asd_mean = Score.mean(asds)
        asd_std = Score.std(asds)
        jse_b0_mean = Score.mean(jse_b0s)
        jse_b0_std = Score.std(jse_b0s)
        jse_b1_mean = Score.mean(jse_b1s)
        jse_b1_std = Score.std(jse_b1s)
        jse_binf_mean = Score.mean(jse_binfs)
        jse_binf_std = Score.std(jse_binfs)
        ase_b0_mean = Score.mean(ase_b0s)
        ase_b0_std = Score.std(ase_b0s)
        ase_b1_mean = Score.mean(ase_b1s)
        ase_b1_std = Score.std(ase_b1s)
        ase_binf_mean = Score.mean(ase_binfs)
        ase_binf_std = Score.std(ase_binfs)

        # Compute verification stats
        verification_stats = Question._compute_verification_stats(questions)

        return Stats(
            asd_mean=asd_mean,
            asd_std=asd_std,
            jse_b0_mean=jse_b0_mean,
            jse_b0_std=jse_b0_std,
            jse_b1_mean=jse_b1_mean,
            jse_b1_std=jse_b1_std,
            jse_binf_mean=jse_binf_mean,
            jse_binf_std=jse_binf_std,
            ase_b0_mean=ase_b0_mean,
            ase_b0_std=ase_b0_std,
            ase_b1_mean=ase_b1_mean,
            ase_b1_std=ase_b1_std,
            ase_binf_mean=ase_binf_mean,
            ase_binf_std=ase_binf_std,
            **verification_stats,
        )

    @staticmethod
    def _compute_verification_stats(questions: list["Question"]) -> dict:
        """Compute verification statistics across all questions."""
        all_verifications = []

        for q in questions:
            if q.transcript is None:
                continue
            for item in q.transcript:
                if item.metadata and "verification" in item.metadata:
                    all_verifications.append(item.metadata["verification"])

        if not all_verifications:
            return {"verification_enabled": False}

        n = len(all_verifications)
        accepted_1try = sum(1 for v in all_verifications if v.get("accepted_on_try") == 1)
        accepted_2tries = sum(1 for v in all_verifications if v.get("accepted_on_try") == 2)
        accepted_3tries = sum(1 for v in all_verifications if v.get("accepted_on_try") == 3)
        never_accepted = sum(1 for v in all_verifications if v.get("accepted_on_try") is None)

        return {
            "verification_enabled": True,
            "verification_accepted_1try_rate": accepted_1try / n,
            "verification_accepted_2tries_rate": accepted_2tries / n,
            "verification_accepted_3tries_rate": accepted_3tries / n,
            "verification_never_accepted_rate": never_accepted / n,
            "verification_avg_tries": sum(v["tries"] for v in all_verifications) / n,
        }

    # following methods apply for treating Question as a transcript

    def append(self, item: "TranscriptItem") -> "Question":
        question = copy.deepcopy(self)
        if self.transcript is None:
            question.transcript = []
        question.transcript.append(item)
        return question


class TranscriptItem(BaseModel):
    """
    Transcript item.
    """

    role: str  # e.g. answer_case_short, or "client"
    content: str
    metadata: dict | None = None
    prompt: str | None = None  # Full prompt used to generate this content


class BetterJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        elif isinstance(obj, BaseModel):
            return obj.model_dump()
        return super().default(obj)
