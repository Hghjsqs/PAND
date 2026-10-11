"""关键词检索基线：从已处理的知识库 Chunk 中找出相关原文。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable


# 第一版先不安装分词库：英文/数字按连续单词处理，中文按单字处理。
# 这样实现简单、没有额外依赖，而且可以直接观察“哪些词命中了”。
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
# 高频虚词通常不能说明问题是否相关，因此不把它们作为检索证据。
STOPWORDS = set("的了是就在有和与及或怎么如何今天")


@dataclass(frozen=True)
class SearchResult:
    """一次检索命中的 Chunk，以及用于排序的证据。"""

    chunk: dict[str, Any]
    score: float
    matched_terms: tuple[str, ...]


def tokenize(text: str) -> list[str]:
    """把问题或文档转换成可比较的词项列表。"""

    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if token not in STOPWORDS
    ]


def load_chunks(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL：每一行是一个 Chunk 对象。"""

    chunks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                # 提前报告具体行号，便于定位文档处理阶段产生的坏数据。
                raise ValueError(f"invalid JSON on line {line_number}: {path}") from exc
            if not isinstance(value, dict) or not value.get("text"):
                # 没有正文的记录无法参与检索，直接阻止错误数据继续传播。
                raise ValueError(f"chunk on line {line_number} has no text: {path}")
            chunks.append(value)
    return chunks


def _score(query: str, text: str) -> tuple[float, tuple[str, ...]]:
    """计算一个简单的词项重合分数，并返回实际命中的词。"""

    query_terms = set(tokenize(query))
    document_terms = set(tokenize(text))
    matched = tuple(sorted(query_terms & document_terms))
    if not query_terms:
        return 0.0, matched

    # 用命中词数除以问题词数，避免长问题天然得到更大的绝对分数。
    score = len(matched) / len(query_terms)
    if query.strip().lower() in text.lower():
        # 整个问题原样出现在 Chunk 中时，额外加分。
        score += 0.5
    return score, matched


def search(
    query: str,
    chunks: Iterable[dict[str, Any]],
    top_k: int = 3,
) -> list[SearchResult]:
    """返回有词项重合的 Chunk，并按分数从高到低排序。"""

    if not query.strip():
        raise ValueError("query cannot be empty")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    results: list[SearchResult] = []
    for chunk in chunks:
        # 只使用正文参与匹配，来源、章节等元数据暂时用于展示和引用。
        score, matched = _score(query, str(chunk["text"]))
        if score > 0:
            results.append(SearchResult(chunk, score, matched))

    # 分数相同时用稳定的 chunk_id 排序，保证同一个问题每次结果顺序一致。
    results.sort(
        key=lambda result: (-result.score, str(result.chunk.get("chunk_id", "")))
    )
    return results[:top_k]


def search_jsonl(path: Path, query: str, top_k: int = 3) -> list[SearchResult]:
    """组合“读取 Chunk + 检索”两个步骤，提供更方便的调用入口。"""

    return search(query, load_chunks(path), top_k=top_k)


def main() -> None:
    """命令行演示：手动输入问题并查看命中的原文。"""

    project_root = Path(__file__).resolve().parents[1]
    chunks_path = project_root / "data" / "processed" / "chunks.jsonl"
    query = input("请输入问题：").strip()
    for index, result in enumerate(search_jsonl(chunks_path, query), start=1):
        chunk = result.chunk
        print(f"\n[{index}] score={result.score:.3f} matched={result.matched_terms}")
        print(f"source={chunk['source']} section={chunk['section']}")
        print(chunk["text"])


if __name__ == "__main__":
    main()
