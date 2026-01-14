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


def _truncate_at_word_boundary(text: str, max_length: int) -> str:
    """
    Truncate text to max_length characters, preferring word boundaries.
    
    Args:
        text: Text to truncate
        max_length: Maximum length in characters
        
    Returns:
        Truncated text with ellipsis if truncated
    """
    if len(text) <= max_length:
        return text
    
    # Reserve space for ellipsis
    effective_max = max_length - 3
    
    # Try to find a word boundary (space or newline) near the limit
    # Look backwards from effective_max for up to 20% of the length
    search_start = max(0, effective_max - max(1, effective_max // 5))
    boundary_pos = text.rfind(' ', search_start, effective_max)
    if boundary_pos == -1:
        boundary_pos = text.rfind('\n', search_start, effective_max)
    
    if boundary_pos > search_start:
        # Found a word boundary, truncate there
        return text[:boundary_pos] + "..."
    else:
        # No word boundary found, truncate at exact limit
        return text[:effective_max] + "..."


def verify_quotes_in_text(text: str, source_text: str | None, max_length: int | None = None) -> str:
    """
    Verify and format quotes in the text against the source text.
    Valid quotes <quote>X</quote> where X is in source_text become <quote_verified>X</quote_verified>.
    Invalid quotes become <quote_invalid>X</quote_invalid>.
    
    If max_length is specified, quotes exceeding this length will be truncated at word boundaries
    with ellipsis added.

    Args:
        text: Text containing quote tags to verify
        source_text: Source text to verify quotes against
        max_length: Maximum character length for quotes. If None, no limit is applied.

    Returns:
        Text with verified/invalid quote tags, and quotes truncated if necessary.
    """
    if not source_text:
        return text

    def replace_quote(match):
        content = match.group(1)
        
        # Truncate if max_length is specified and content exceeds it
        if max_length is not None and len(content) > max_length:
            content = _truncate_at_word_boundary(content, max_length)
        
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
    return_prompt: bool = False,
) -> tuple[str, dict, str | None] | tuple[str, dict]:
    """
    Generate an argument with alignment verification and retry logic.

    Args:
        agent_callable: Async function that generates argument string.
                       Should accept optional 'feedback' and 'return_prompt' keyword arguments.
        question: The question being answered
        answer_case: The answer to argue for
        max_tries: Override for VERIFY_ALIGNMENT_N_TRIES
        return_prompt: If True, also return the prompt used to generate the argument

    Returns:
        If return_prompt=False: Tuple of (final_argument, verification_metadata)
        If return_prompt=True: Tuple of (final_argument, verification_metadata, prompt_string)

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
    prompt_str = None

    if not VERIFY_ALIGNMENT:
        result = await agent_callable(return_prompt=return_prompt)
        if return_prompt:
            argument, prompt_str = result
            return argument, {}, prompt_str
        return result, {}

    max_tries = max_tries or VERIFY_ALIGNMENT_N_TRIES

    for try_num in range(1, max_tries + 1):
        if try_num == 1:
            result = await agent_callable(return_prompt=return_prompt)
            if return_prompt:
                argument, prompt_str = result
            else:
                argument = result
        else:
            feedback = (
                f"[FEEDBACK: No, you're supposed to argue for answer "
                f"{answer_case.short}. Your previous response was arguing for "
                f"the wrong answer. In your response, don't apologize or respond "
                f"to the feedback, JUST give the proper argument taking this "
                f"feedback into account.]"
            )
            result = await agent_callable(feedback=feedback, return_prompt=return_prompt)
            if return_prompt:
                argument, prompt_str = result
            else:
                argument = result

        is_aligned = await verify_argument_alignment(
            argument=argument,
            question=question,
            intended_answer=answer_case,
        )

        if is_aligned:
            LOGGER.info(
                f"Argument alignment verified on try {try_num}/{max_tries}"
            )
            metadata = {
                "verification": {
                    "is_aligned": True,
                    "tries": try_num,
                    "accepted_on_try": try_num,
                }
            }
            if return_prompt:
                return argument, metadata, prompt_str
            return argument, metadata
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

    metadata = {
        "verification": {
            "is_aligned": final_aligned,
            "tries": max_tries,
            "accepted_on_try": None,  # Never accepted during retry loop
        }
    }
    if return_prompt:
        return argument, metadata, prompt_str
    return argument, metadata
