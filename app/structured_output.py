"""结构化问答：校验分类结果，并可选地接入检索证据。"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from app.llm_client import chat, stream_chat
from app.rag import format_context
from app.retriever import SearchResult, search_jsonl


ALLOWED_CATEGORIES = {
    "leave_policy",
    "reimbursement",
    "purchase",
    "it_support",
    "unknown",
}
REQUIRED_FIELDS = {"answer", "category", "need_retrieval", "confidence"}


def _parse_json_object(raw: str) -> dict[str, Any]:
    """解析 JSON 对象，同时兼容模型包裹的 Markdown 代码块。"""

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
        # 模型偶尔会在 JSON 前后添加一句解释，截取完整对象再尝试解析。
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


def _validate_result(value: dict[str, Any], require_citations: bool = False) -> dict[str, Any]:
    """校验模型返回的字段类型和取值范围。"""

    missing = REQUIRED_FIELDS - value.keys()
    if missing:
        raise ValueError(f"structured response is missing fields: {sorted(missing)}")

    if not isinstance(value["answer"], str):
        raise ValueError("answer must be a string")
    if value["category"] not in ALLOWED_CATEGORIES:
        raise ValueError(f"category must be one of: {sorted(ALLOWED_CATEGORIES)}")
    if not isinstance(value["need_retrieval"], bool):
        raise ValueError("need_retrieval must be a boolean")
    if isinstance(value["confidence"], bool) or not isinstance(
        value["confidence"], (int, float)
    ):
        raise ValueError("confidence must be a number between 0 and 1")
    if not 0 <= value["confidence"] <= 1:
        raise ValueError("confidence must be a number between 0 and 1")

    if require_citations:
        citations = value.get("citations")
        if not isinstance(citations, list):
            raise ValueError("citations must be a list")
        for citation in citations:
            if not isinstance(citation, dict):
                raise ValueError("each citation must be an object")
            for field in ("source", "section", "chunk_id"):
                if not isinstance(citation.get(field), str) or not citation[field]:
                    raise ValueError(f"citation.{field} must be a non-empty string")

    return value


def _validate_citation_targets(
    citations: list[dict[str, Any]],
    results: list[SearchResult],
) -> None:
    """确保模型引用的文件、章节和 Chunk ID 都来自本次检索结果。"""

    allowed = {
        (
            str(result.chunk.get("source", "")),
            str(result.chunk.get("section", "")),
            str(result.chunk.get("chunk_id", "")),
        )
        for result in results
    }
    for citation in citations:
        target = (citation["source"], citation["section"], citation["chunk_id"])
        if target not in allowed:
            raise ValueError(f"citation does not match retrieved evidence: {target}")


def _classification_prompt(user_prompt: str) -> str:
    """构造不依赖知识库的分类 Prompt，保留旧接口的职责。"""

    return f"""
你是企业知识库问答系统的路由器。请分析用户问题，并且只返回一个 JSON 对象，不能返回 Markdown、解释文字或代码块。

JSON 必须严格包含以下字段：
- answer: 对问题的简短回答；
- category: 只能是 leave_policy、reimbursement、purchase、it_support、unknown 之一；
- need_retrieval: 是否需要查询企业知识库，必须是 true 或 false；
- confidence: 0 到 1 之间的数字。

如果问题不属于企业知识库范围，category 使用 unknown，need_retrieval 使用 false。

用户问题：
{user_prompt.strip()}
""".strip()


def build_structured_prompt(question: str, results: list[SearchResult]) -> str:
    """构造带检索证据和可机器解析引用的 JSON Prompt。"""

    if not question.strip():
        raise ValueError("question cannot be empty")

    return f"""
你是企业知识库问答助手。只能依据下方检索证据回答，不得补充证据之外的制度信息。
只返回一个 JSON 对象，不要返回 Markdown、解释文字或代码块。

JSON 必须严格包含：
- answer: 简洁回答；没有足够证据时必须说明“知识库中没有找到足够依据”；
- category: leave_policy、reimbursement、purchase、it_support、unknown 之一；
- need_retrieval: true 或 false；
- confidence: 0 到 1 之间的数字；
- citations: 引用的证据列表，每项必须包含 source、section、chunk_id；无证据时返回空列表。

用户问题：
{question.strip()}

检索证据：
{format_context(results)}
""".strip()


def classify_question(user_prompt: str) -> dict[str, Any]:
    """保留原有的非检索分类接口。"""

    if not user_prompt.strip():
        raise ValueError("user_prompt cannot be empty")
    return _validate_result(_parse_json_object(chat(_classification_prompt(user_prompt))))


def classify_question_stream(
    user_prompt: str,
    chunks_path: Path | None = None,
    top_k: int = 3,
) -> Iterator[str]:
    """流式分类；传入 chunks_path 时额外使用知识库证据。"""

    if not user_prompt.strip():
        raise ValueError("user_prompt cannot be empty")

    if chunks_path is None:
        prompt = _classification_prompt(user_prompt)
    else:
        results = search_jsonl(chunks_path, user_prompt, top_k=top_k)
        prompt = build_structured_prompt(user_prompt, results)
    yield from stream_chat(prompt)


def classify_question_stream_result(
    user_prompt: str,
    chunks_path: Path | None = None,
    top_k: int = 3,
    on_fragment: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """收集流式结果后解析；有 chunks_path 时同时校验 citations。"""

    # 结果校验阶段也读取同一批证据，防止模型伪造来源。
    results = search_jsonl(chunks_path, user_prompt, top_k=top_k) if chunks_path else None
    fragments: list[str] = []
    for fragment in classify_question_stream(user_prompt, chunks_path, top_k):
        fragments.append(fragment)
        if on_fragment is not None:
            on_fragment(fragment)

    result = _validate_result(
        _parse_json_object("".join(fragments)),
        require_citations=chunks_path is not None,
    )
    if results is not None:
        _validate_citation_targets(result["citations"], results)
    return result
