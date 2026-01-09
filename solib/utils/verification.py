"""
Argument alignment verification module.

Provides scaffolding to verify that agent arguments are arguing for the intended
answer case, with retry logic and comprehensive tracking.
"""

import logging
import re
from typing import Callable, Any
from pydantic import BaseModel, Field

from solib.datatypes import Question, Answer
from solib.utils.globals import (
    VERIFY_ALIGNMENT,
    VERIFY_ALIGNMENT_N_TRIES,
    VERIFY_ALIGNMENT_MODEL,
    jinja_env,
)
from solib.utils.llm_utils import acompletion_wrapper

LOGGER = logging.getLogger(__name__)


class AlignmentCheck(BaseModel):
    """Structured output for alignment verification."""
    is_aligned: bool = Field(description="True if the argument is arguing for the intended answer, False otherwise.")


async def verify_argument_alignment(
    argument: str,
    question: Question,
    intended_answer: Answer,
) -> bool:
    """
    Use a small, cheap model to verify if the argument is arguing
    for the intended answer.

    Args:
        argument: The argument text to verify
        question: The question being answered
        intended_answer: The answer the argument should be arguing for

    Returns:
        True if aligned, False otherwise.
    """
    verification_template = jinja_env.get_template("verification/alignment_check.jinja")
    verification_prompt = verification_template.render(
        question=question.question,
        intended_answer_short=intended_answer.short,
        argument=argument,
    )

    result: AlignmentCheck = await acompletion_wrapper(
        model=VERIFY_ALIGNMENT_MODEL,
        response_format=AlignmentCheck,
        messages=[{"role": "user", "content": verification_prompt}],
    )

    return result.is_aligned


def verify_quotes_in_text(text: str, source_text: str | None) -> str:
    """
    Verify and format quotes in the text against the source text.
    Valid quotes <quote>X</quote> where X is in source_text become <quote>X</quote> (verified).
    Invalid quotes become [Invalid Quote: X].

    The output format uses XML-like tags that the Judge can be instructed to trust (or simply
    retains the <quote> tag which means 'verified' in this context, while removing/flagging invalid ones).

    Here, we will change <quote> to <verified_quote> if valid, and <invalid_quote> if not.
    """
    if not source_text:
        return text

    def replace_quote(match):
        content = match.group(1)
        # Check if content (or trimmed content) is in source_text
        if content in source_text:
             return f"<quote_verified>{content}</quote_verified>"
        if content.strip() in source_text:
             return f"<quote_verified>{content}</quote_verified>"

        return f"<quote_invalid>{content}</quote_invalid>"

    pattern = r"<quote>(.*?)</quote>"
    # re.DOTALL allows dot to match newlines inside the quote
    return re.sub(pattern, replace_quote, text, flags=re.DOTALL)


async def generate_argument_with_verification(
    agent_callable: Callable[..., Any],
    question: Question,
    answer_case: Answer,
    max_tries: int | None = None,
) -> tuple[str, dict]:
    """
    Generate an argument with alignment verification and retry logic.

    Args:
        agent_callable: Async function that generates argument string.
                       Should accept optional 'feedback' keyword argument.
        question: The question being answered
        answer_case: The answer to argue for
        max_tries: Override for VERIFY_ALIGNMENT_N_TRIES

    Returns:
        Tuple of (final_argument, verification_metadata)
        verification_metadata is empty dict if VERIFY_ALIGNMENT is False,
        otherwise contains:
        {
            "verification": {
                "is_aligned": bool,      # Whether final argument was acceptable
                "tries": int,            # Total tries (1-N)
                "accepted_on_try": int | None  # Which try was accepted (None if never)
            }
        }
    """
    if not VERIFY_ALIGNMENT:
        argument = await agent_callable()
        return argument, {}

    max_tries = max_tries or VERIFY_ALIGNMENT_N_TRIES

    for try_num in range(1, max_tries + 1):
        if try_num == 1:
            argument = await agent_callable()
        else:
            feedback = (
                f"[FEEDBACK: No, you're supposed to argue for answer "
                f"{answer_case.short}. Your previous response was arguing for "
                f"the wrong answer. In your response, don't apologize or respond "
                f"to the feedback, JUST give the proper argument taking this "
                f"feedback into account.]"
            )
            argument = await agent_callable(feedback=feedback)

        is_aligned = await verify_argument_alignment(
            argument=argument,
            question=question,
            intended_answer=answer_case,
        )

        if is_aligned:
            LOGGER.info(
                f"Argument alignment verified on try {try_num}/{max_tries}"
            )
            return argument, {
                "verification": {
                    "is_aligned": True,
                    "tries": try_num,
                    "accepted_on_try": try_num,
                }
            }
        else:
            LOGGER.warning(
                f"Argument alignment failed on try {try_num}/{max_tries}"
            )

    # All retries exhausted - do final check on the last argument
    final_aligned = await verify_argument_alignment(
        argument=argument,
        question=question,
        intended_answer=answer_case,
    )

    LOGGER.warning(
        f"All {max_tries} tries exhausted. Final argument aligned: {final_aligned}"
    )

    return argument, {
        "verification": {
            "is_aligned": final_aligned,
            "tries": max_tries,
            "accepted_on_try": None,  # Never accepted during retry loop
        }
    }
