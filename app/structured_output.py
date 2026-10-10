"""Structured question classification built on top of the basic chat client."""

from __future__ import annotations

import json
from typing import Any

from app.llm_client import chat


ALLOWED_CATEGORIES = {
    "leave_policy",
    "reimbursement",
    "purchase",
    "it_support",
    "unknown",
}
REQUIRED_FIELDS = {"answer", "category", "need_retrieval", "confidence"}


def _parse_json_object(raw: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating a surrounding Markdown code fence."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        # Some models add a short sentence before the JSON. Keep the parser
        # forgiving while still requiring one complete JSON object.
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"model did not return valid JSON: {raw!r}") from None
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"model did not return valid JSON: {raw!r}") from exc

    if not isinstance(value, dict):
        raise ValueError("structured response must be a JSON object")
    return value


def _validate_result(value: dict[str, Any]) -> dict[str, Any]:
    missing = REQUIRED_FIELDS - value.keys()
    if missing:
        raise ValueError(f"structured response is missing fields: {sorted(missing)}")

    if not isinstance(value["answer"], str):
        raise ValueError("answer must be a string")
    if value["category"] not in ALLOWED_CATEGORIES:
        raise ValueError(f"category must be one of: {sorted(ALLOWED_CATEGORIES)}")
    if not isinstance(value["need_retrieval"], bool):
        raise ValueError("need_retrieval must be a boolean")
    if isinstance(value["confidence"], bool) or not isinstance(value["confidence"], (int, float)):
        raise ValueError("confidence must be a number between 0 and 1")
    if not 0 <= value["confidence"] <= 1:
        raise ValueError("confidence must be a number between 0 and 1")

    return value


def classify_question(user_prompt: str) -> dict[str, Any]:
    """Classify an enterprise question and return a validated Python dict."""
    if not user_prompt.strip():
        raise ValueError("user_prompt cannot be empty")

    instruction = f"""
你是企业知识库问答系统的路由器。请分析用户问题，并且只返回一个 JSON 对象，不能返回 Markdown、解释文字或代码块。

JSON 必须严格包含以下字段：
- answer: 对问题的简短回答；
- category: 只能是 leave_policy、reimbursement、purchase、it_support、unknown 之一；
- need_retrieval: 是否需要查询企业知识库，必须是 true 或 false；
- confidence: 0 到 1 之间的数字。

如果问题不属于企业知识库范围，category 使用 unknown，need_retrieval 使用 false。

用户问题：{user_prompt}
""".strip()

    raw = chat(instruction)
    return _validate_result(_parse_json_object(raw))
