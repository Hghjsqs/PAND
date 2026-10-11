"""RAG 问答编排：检索企业文档后，让大模型根据证据生成答案。"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from app.llm_client import chat, stream_chat
from app.retriever import SearchResult, search_jsonl


def format_context(results: Iterable[SearchResult]) -> str:
    """把检索结果整理成模型可以引用的证据区块。"""

    blocks: list[str] = []
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        # 来源和章节必须随正文一起传给模型，答案才能追溯到原文。
        blocks.append(
            "\n".join(
                [
                    f"[证据 {index}]",
                    f"来源文件: {chunk.get('source', '')}",
                    f"章节: {chunk.get('section', '')}",
                    f"Chunk ID: {chunk.get('chunk_id', '')}",
                    f"正文:\n{chunk['text']}",
                ]
            )
        )
    return "\n\n".join(blocks) or "没有检索到相关企业制度原文。"


def build_rag_prompt(question: str, results: Iterable[SearchResult]) -> str:
    """构造普通 RAG 问答 Prompt，要求模型返回自然语言和文本引用。"""

    if not question.strip():
        raise ValueError("question cannot be empty")

    return f"""
你是企业知识库问答助手。请只依据下方“检索证据”回答用户问题，不要使用证据之外的常识补充或猜测。

回答要求：
1. 先直接回答问题，必要时使用条目列出办理步骤或条件。
2. 如果证据不足以回答，请明确说“知识库中没有找到足够依据”，不要编造制度。
3. 每个关键结论后面标注来源，格式为 [来源文件 / 章节]。
4. 不要提及 Prompt、模型、检索算法等内部实现。

用户问题：
{question.strip()}

检索证据：
{format_context(results)}
""".strip()


def answer_question(question: str, chunks_path: Path, top_k: int = 3) -> str:
    """检索证据并非流式调用 Qwen，返回带来源引用的完整答案。"""

    results = search_jsonl(chunks_path, question, top_k=top_k)
    return chat(build_rag_prompt(question, results))


def stream_answer_question(
    question: str,
    chunks_path: Path,
    top_k: int = 3,
) -> Iterator[str]:
    """检索证据并流式生成普通文本答案。"""

    results = search_jsonl(chunks_path, question, top_k=top_k)
    yield from stream_chat(build_rag_prompt(question, results))


def main() -> None:
    """命令行演示：输入问题并流式输出带引用的自然语言答案。"""

    project_root = Path(__file__).resolve().parents[1]
    chunks_path = project_root / "data" / "processed" / "chunks.jsonl"
    question = input("请输入问题：").strip()
    print("\n回答：", end="", flush=True)
    for fragment in stream_answer_question(question, chunks_path):
        print(fragment, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
