"""
NVIDIA NIM Client Module
=========================
Wraps the NVIDIA NIM API (OpenAI-compatible) for tree construction / reasoning.
Uses the openai SDK with a custom base_url.
"""

import logging
import time

from openai import OpenAI

from multi_paper_rag.phase1.config import NVIDIA_API_KEY, NVIDIA_BASE_URL, NVIDIA_MODEL

logger = logging.getLogger("phase1.llm.nvidia_client")


class NVIDIAClientError(Exception):
    """Raised when all NVIDIA NIM API call attempts fail."""


def _get_client() -> OpenAI:
    """
    Lazily build an OpenAI client pointed at NVIDIA NIM.

    Returns:
        Configured ``OpenAI`` client instance.
    """
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)


def call_nvidia(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 2000,
) -> str:
    """
    Call the NVIDIA NIM API with retry logic.

    Args:
        system_prompt: System-level instruction for the LLM.
        user_prompt:   User-level prompt / content.
        temperature:   Sampling temperature (default 0.1 for determinism).
        max_tokens:    Max tokens in the response.

    Returns:
        The LLM response content string.

    Raises:
        NVIDIAClientError: If all 3 retry attempts fail.
    """
    client = _get_client()
    last_error: Exception | None = None

    for attempt in range(1, 4):
        try:
            logger.debug(
                "NVIDIA call attempt %d/3 — model=%s, temp=%.2f",
                attempt,
                NVIDIA_MODEL,
                temperature,
            )
            response = client.chat.completions.create(
                model=NVIDIA_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            if not content.strip():
                raise ValueError("NVIDIA returned an empty response")
            logger.debug("NVIDIA response length: %d chars", len(content))
            return content

        except Exception as exc:
            last_error = exc
            logger.warning(
                "NVIDIA call attempt %d/3 failed: %s", attempt, exc
            )
            if attempt < 3:
                time.sleep(2)

    error_msg = f"NVIDIA NIM API failed after 3 attempts: {last_error}"
    logger.error(error_msg)
    raise NVIDIAClientError(error_msg)
