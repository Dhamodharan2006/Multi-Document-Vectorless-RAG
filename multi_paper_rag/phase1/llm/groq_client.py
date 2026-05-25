"""
Groq Client Module
==================
Wraps the Groq API for fast querying (section selection + answer generation).
Uses the official groq Python SDK.
"""

import logging
import time

from groq import Groq

from multi_paper_rag.phase1.config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger("phase1.llm.groq_client")


class GroqClientError(Exception):
    """Raised when all Groq API call attempts fail."""


def _get_client() -> Groq:
    """
    Build a Groq client instance.

    Returns:
        Configured ``Groq`` client.
    """
    return Groq(api_key=GROQ_API_KEY)


def call_groq(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 2000,
) -> str:
    """
    Call the Groq API with retry logic.

    Args:
        system_prompt: System-level instruction for the LLM.
        user_prompt:   User-level prompt / content.
        temperature:   Sampling temperature (default 0.1 for determinism).
        max_tokens:    Max tokens in the response.

    Returns:
        The LLM response content string.

    Raises:
        GroqClientError: If all 3 retry attempts fail.
    """
    client = _get_client()
    last_error: Exception | None = None

    for attempt in range(1, 4):
        try:
            logger.debug(
                "Groq call attempt %d/3 — model=%s, temp=%.2f",
                attempt,
                GROQ_MODEL,
                temperature,
            )
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            if not content.strip():
                raise ValueError("Groq returned an empty response")
            logger.debug("Groq response length: %d chars", len(content))
            return content

        except Exception as exc:
            last_error = exc
            logger.warning("Groq call attempt %d/3 failed: %s", attempt, exc)
            if attempt < 3:
                time.sleep(2)

    error_msg = f"Groq API failed after 3 attempts: {last_error}"
    logger.error(error_msg)
    raise GroqClientError(error_msg)
