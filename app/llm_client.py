"""Small Qwen client used by the application layers."""

from __future__ import annotations

from collections.abc import Iterator

from openai import OpenAI

from app.config import DASHSCOPE_API_KEY, QWEN_BASE_URL, QWEN_MODEL
from app.errors import LLMClientError
from app.logging_config import configure_logging


client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url=QWEN_BASE_URL,
    timeout=30.0,
    max_retries=1,
)
logger = configure_logging()


SYSTEM_PROMPT = "You are a concise and accurate enterprise knowledge-base assistant."


def _messages(user_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def chat(user_prompt: str) -> str:
    """Send one non-streaming request and return its text."""
    if not user_prompt.strip():
        raise ValueError("user_prompt cannot be empty")

    logger.info("starting non-stream request model=%s prompt_chars=%d", QWEN_MODEL, len(user_prompt))
    try:
        response = client.chat.completions.create(
            model=QWEN_MODEL,
            messages=_messages(user_prompt),
            temperature=0.2,
        )
    except Exception as exc:
        logger.exception("non-stream request failed")
        raise LLMClientError(f"model request failed: {exc}") from exc

    if not response.choices:
        logger.error("model returned no choices")
        raise LLMClientError("model returned no choices")

    content = response.choices[0].message.content
    if not content:
        logger.error("model returned empty content")
        raise LLMClientError("model returned empty content")

    logger.info("non-stream request completed response_chars=%d", len(content))
    return content


def stream_chat(user_prompt: str) -> Iterator[str]:
    """Yield generated text fragments as Qwen produces them."""
    if not user_prompt.strip():
        raise ValueError("user_prompt cannot be empty")

    logger.info("starting stream request model=%s prompt_chars=%d", QWEN_MODEL, len(user_prompt))
    try:
        response = client.chat.completions.create(
            model=QWEN_MODEL,
            messages=_messages(user_prompt),
            temperature=0.2,
            stream=True,
        )
    except Exception as exc:
        logger.exception("stream request failed before receiving response")
        raise LLMClientError(f"stream request failed: {exc}") from exc

    yielded = False
    try:
        for chunk in response:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                yielded = True
                yield content
    except Exception as exc:
        logger.exception("stream request failed while receiving response")
        raise LLMClientError(f"stream response failed: {exc}") from exc

    if not yielded:
        logger.error("model returned no streamed content")
        raise LLMClientError("model returned no streamed content")
    print()
    logger.info("stream request completed")
