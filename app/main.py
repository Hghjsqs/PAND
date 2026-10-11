"""企业知识库问答系统的命令行入口。"""

from __future__ import annotations

from pathlib import Path

from app.structured_output import classify_question_stream_result


def print_fragment(fragment: str) -> None:
    """模型返回一个片段后立即打印，形成流式输出效果。"""

    print(fragment, end="", flush=True)


def main() -> None:
    """输入一个问题，调用 RAG 流程并输出结构化结果。"""

    project_root = Path(__file__).resolve().parents[1]
    chunks_path = project_root / "data" / "processed" / "chunks.jsonl"
    question = input("请输入问题：").strip()

    print("\n流式回答：", end="", flush=True)
    result = classify_question_stream_result(
        question,
        chunks_path=chunks_path,
        top_k=3,
        on_fragment=print_fragment,
    )

    print("\n结构化结果：")
    print(result)


if __name__ == "__main__":
    main()
