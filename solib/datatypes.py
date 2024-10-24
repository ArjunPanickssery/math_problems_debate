import json
import logging
import inspect
import warnings
import numpy as np
import copy
from functools import wraps
from typing import Any, Literal, Optional, Union, Callable, Self
from pydantic import BaseModel, field_validator, computed_field, model_validator

LOGGER = logging.getLogger(__name__)


class Prob(BaseModel):
    prob: float

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


class Score(BaseModel):
    """
    Data class to store scores based on different metrics.
    """

    log: float | None = None
    logodds: float | None = None
    accuracy: float | None = None

    @classmethod
    def calc(cls, question: "Question", answer_case: "Answer") -> Optional["Score"]:
        """Score for answer_case based on probability assigned to it in Question."""
        # .censor() because e.g. question might be answer_case.case_probs, so
        # question.answer_cases will have probabilities while answer_case would not
        assert answer_case.censor() in question.censor().answer_cases
        assert question.is_elicited
        if not all(a.judge_prob is not None for a in question.answer_cases):
            return None
        if not question.is_normalized:
            warnings.warn("Calculating score based on unnormalized probabilities.")

        # find answer case in question
        answer_case_in_question = question.answer_cases_dict[answer_case.short]

        score_log = np.log(answer_case_in_question.judge_prob)
        score_logodds = np.log(answer_case_in_question.judge_prob) - np.log(
            question.neg(answer_case_in_question).judge_prob
        )
        score_accuracy = float(
            answer_case_in_question.judge_prob
            > question.neg(answer_case_in_question).judge_prob
        )
        return Score(log=score_log, logodds=score_logodds, accuracy=score_accuracy)

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
                logodds=apply(self.logodds, other.logodds),
                accuracy=apply(self.accuracy, other.accuracy),
            )
        elif isinstance(other, (int, float)):  # Scalar operations
            return Score(
                log=apply(self.log, other),
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
            logodds=op([score.logodds for score in scores]),
            accuracy=op([score.accuracy for score in scores]),
        )

    @classmethod
    def mean(cls, scores: list["Score"]) -> "Score":
        return cls._arroperate(scores, np.mean)

    @classmethod
    def std(cls, scores: list["Score"]) -> "Score":
        return cls._arroperate(scores, np.std)


class Stats(BaseModel):
    asd_mean: Score
    asd_std: Score
    jsmax_mean: Score
    jsmax_std: Score
    jsmins_mean: Score
    jsmins_std: Score
    jsuni_mean: Score
    jsuni_std: Score


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
        judge_score_max (Score): computed. Also requires .is_grounded. Judge score if agent argues for
            the best answer.
        judge_score_min (Score): computed. Also requires .is_grounded. Judge score if agent argues for
            the worst answer.
        judge_score_uniform (Score): computed. Also requires .is_grounded. Judge score if agent argues for
            any answer with equal probability.
    Additional methods in this case:
        judge_score_expected(
            agent_arguing_for:
                Callable[[Answer], Prob]
                | list[float]
                | Literal["max", "min", "uniform"]
            ) -> Score:
            computed. Also requires .is_grounded. Scores the elicited probabilities against the true answer,
                given how likely the agent is to argue for each answer.

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
    answer_cases: list[Answer]
    transcript: list["TranscriptItem"] | None = None

    # not necessary, it's enough to validate this in Answer
    # @model_validator(mode="after")
    # def makes_sense(self) -> Self:
    #     if self.is_argued:
    #         assert not self.is_elicited
    #     if self.is_elicited:
    #         assert not self.is_argued
    #     return self

    def censor(self) -> "Question":
        return Question(
            question=self.question,
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
        return Question(
            question=self.question,
            answer_cases=[
                Answer(
                    short=answer.short,
                    long=answer.long,
                    judge_prob=Prob(prob=answer.judge_prob / self.total_prob),
                )
                for answer in self.answer_cases
            ],
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

    def judge_score_expected(
        self,
        agent_arguing_for: Union[
            Callable[[Answer], Prob], list[float], Literal["max", "min", "uniform"]
        ],
    ) -> Score:
        """Expected judge score given how likely an agent is to argue for each answer.

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
            agent_arguing_for_func = lambda a: agent_arguing_for[
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

    @computed_field
    @property
    def judge_score_max(self) -> Score | None:
        if not (self.is_argued and self.is_grounded):
            return None
        return self.judge_score_expected("max")

    @computed_field
    @property
    def judge_score_min(self) -> Score | None:
        if not (self.is_argued and self.is_grounded):
            return None
        return self.judge_score_expected("min")

    @computed_field
    @property
    def judge_score_uniform(self) -> Score | None:
        if not (self.is_argued and self.is_grounded):
            return None
        return self.judge_score_expected("uniform")

    @staticmethod
    def compute_stats(questions: list["Question"]) -> dict[str, Any]:
        """Compute stats for a list of .is_argued Questions."""
        assert all(
            question.is_argued and question.is_grounded for question in questions
        )
        asds = [question.agent_score_diff for question in questions]
        jsmaxs = [question.judge_score_max for question in questions]
        jsmins = [question.judge_score_min for question in questions]
        jsunis = [question.judge_score_uniform for question in questions]
        asd_mean = Score.mean(asds)
        asd_std = Score.std(asds)
        jsmax_mean = Score.mean(jsmaxs)
        jsmax_std = Score.std(jsmaxs)
        jsmins_mean = Score.mean(jsmins)
        jsmins_std = Score.std(jsmins)
        jsuni_mean = Score.mean(jsunis)
        jsuni_std = Score.std(jsunis)
        return Stats(
            asd_mean=asd_mean,
            asd_std=asd_std,
            jsmax_mean=jsmax_mean,
            jsmax_std=jsmax_std,
            jsmins_mean=jsmins_mean,
            jsmins_std=jsmins_std,
            jsuni_mean=jsuni_mean,
            jsuni_std=jsuni_std,
        )

    # following methods apply for treating Question as a transcript

    def append(self, item: "TranscriptItem") -> None:
        if self.transcript is None:
            self.transcript = []
        self.transcript.append(item)


class TranscriptItem(BaseModel):
    """
    Transcript item.
    """

    role: str  # e.g. answer_case_short, or "client"
    content: str
    metadata: dict | None = None


class BetterJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        elif isinstance(obj, BaseModel):
            return obj.model_dump()
        return super().default(obj)
