import logging
from pathlib import Path
from solib.datatypes import Question, Answer, TranscriptItem
from solib.protocols.abstract import Protocol, QA_Agent, Judge
from solib.data.loading import Dataset
from solib.utils import write_json, write_jsonl_async, parallelized_call
from solib.utils.verification import generate_argument_with_verification, verify_quotes_in_text

LOGGER = logging.getLogger(__name__)


def propaganda_to_market_making(question: Question) -> Question:
    """
    Transform a Propaganda result into a MarketMaking result.

    In Propaganda: answer.case_probs = probs after agent argued FOR that answer
    In MarketMaking: answer.case_probs = probs after adversary argued AGAINST that answer
                                       = probs after adversary argued FOR the opposite answer
                                       = Propaganda's opposite answer's case_probs
    """
    assert question.is_argued
    assert len(question.answer_cases) == 2, "MarketMaking only supports binary questions"

    return Question(
        question=question.question,
        answer_cases=[
            Answer(
                short=a.short,
                long=a.long,
                value=a.value,
                case_probs=question.neg(a).case_probs,  # swap case_probs
            )
            for a in question.answer_cases
        ],
        transcript=question.transcript,
    )


class Propaganda(Protocol):
    async def run(
        self,
        agent: QA_Agent,
        question: Question,
        answer_case: Answer,
        judge: Judge,
        temperature: float = 0.4,
        write: Path | str | None = None,
        cache_breaker: str | int | None = None,
        **rendering_components,
    ) -> Question:
        assert isinstance(question, Question)
        # Ensure answer_case_short is always passed to the template
        extra_user_renders = rendering_components.get("extra_user_renders") or {}
        extra_user_renders["answer_case_short"] = answer_case.short

        # Create callable for argument generation (accepts optional feedback and return_prompt)
        async def generate_argument(feedback: str = None, return_prompt: bool = False):
            return await agent(
                question=question,
                answer_case=answer_case,
                extra_user_renders=extra_user_renders,
                context=self.ts_to_prompt(question),
                feedback=feedback,
                cache_breaker=cache_breaker,
                temperature=temperature,
                write=write,
                return_prompt=return_prompt,
            )

        # Generate with verification (if enabled)
        agent_response, verification_metadata, agent_prompt = await generate_argument_with_verification(
            agent_callable=generate_argument,
            question=question,
            answer_case=answer_case,
            return_prompt=True,
        )
        # verify quotes
        quote_max_length = rendering_components.get("quote_max_length")
        agent_response = verify_quotes_in_text(agent_response, question.source_text, max_length=quote_max_length)

        question = question.append(
            TranscriptItem(
                role=answer_case.short,
                content=agent_response,
                metadata=verification_metadata if verification_metadata else None,
                prompt=agent_prompt,
            )
        )
        assert question.transcript is not None
        result = await judge(
            question=question,
            context=self.ts_to_prompt(question),
            cache_breaker=cache_breaker,
            write=write,
        )
        assert result.transcript is not None
        return result

    async def experiment(
        self,
        agent: QA_Agent,
        questions: Dataset,
        judge: Judge,
        write: Path | str | tuple[Path, Path] | None = None,
        continue_from_results: dict[tuple, Question] | None = None,
        **other_components,
    ) -> tuple[list[Question], dict]:
        """
        Override experiment to also compute and save MarketMaking results.

        Args:
            write: Can be a single path (only Propaganda saved) or a tuple of paths
                   (propaganda_path, market_making_path).
        """
        # Handle write paths
        if isinstance(write, tuple):
            propaganda_write, market_making_write = write
        else:
            propaganda_write = write
            market_making_write = None

        # Run the standard Propaganda experiment
        propaganda_results, propaganda_stats = await super().experiment(
            agent=agent,
            questions=questions,
            judge=judge,
            write=propaganda_write,
            continue_from_results=continue_from_results,
            **other_components,
        )

        # If we have a market_making path, compute and save those results too
        if market_making_write:
            Path(market_making_write).mkdir(parents=True, exist_ok=True)

            # Transform propaganda results to market-making results
            market_making_results = [
                propaganda_to_market_making(q) for q in propaganda_results
            ]

            # Compute stats for market-making
            market_making_stats = Question.compute_stats(market_making_results)

            # Write market-making results
            write_results_path = Path(market_making_write) / "results.jsonl"
            write_stats_path = Path(market_making_write) / "stats.json"
            write_config_path = Path(market_making_write) / "config.json"

            # Write config (same as propaganda but note it's derived)
            config = self.get_experiment_config(agent=agent, judge=judge, **other_components)
            config["derived_from"] = "Propaganda"
            write_json(config, path=write_config_path)

            # Write results
            for result in market_making_results:
                await write_jsonl_async(
                    result.model_dump(exclude_none=True),
                    path=write_results_path,
                    append=True,
                )

            # Write stats
            write_json(market_making_stats, path=write_stats_path)

            LOGGER.info(f"MarketMaking results saved to {market_making_write}")

        return propaganda_results, propaganda_stats
