from pathlib import Path
import tempfile

from app.document_processor import load_markdown_documents, write_jsonl
import app.rag as rag
from app.rag import build_rag_prompt, format_context, stream_answer_question
from app.retriever import search_jsonl


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    chunks = list(load_markdown_documents(project_root / "data" / "raw"))

    with tempfile.TemporaryDirectory() as temp_dir:
        chunks_path = Path(temp_dir) / "chunks.jsonl"
        write_jsonl(chunks, chunks_path)
        results = search_jsonl(chunks_path, "年假怎么申请", top_k=3)

    context = format_context(results)
    prompt = build_rag_prompt("年假怎么申请", results)

    # Prompt 必须同时包含问题、原文和引用所需的来源信息。
    assert "年假怎么申请" in prompt
    assert "leave_policy.md" in prompt
    assert "请假申请流程" in prompt
    assert "[证据 1]" in context
    assert "Chunk ID" in context

    try:
        build_rag_prompt("", results)
    except ValueError as exc:
        assert str(exc) == "question cannot be empty"
    else:
        raise AssertionError("empty question should fail")

    # 用本地假数据替代 Qwen，测试流式函数本身而不是网络。
    captured: list[str] = []

    def fake_stream_chat(prompt: str):
        assert "年假怎么申请" in prompt
        assert "leave_policy.md" in prompt
        captured.extend(["第一段", "第二段"])
        yield "第一段"
        yield "第二段"

    original_stream_chat = rag.stream_chat
    rag.stream_chat = fake_stream_chat
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            chunks_path = Path(temp_dir) / "chunks.jsonl"
            write_jsonl(chunks, chunks_path)
            fragments = list(stream_answer_question("年假怎么申请", chunks_path))
    finally:
        rag.stream_chat = original_stream_chat

    assert fragments == ["第一段", "第二段"]
    assert "".join(fragments) == "第一段第二段"
    assert captured == fragments

    print("RAG Prompt 测试通过")


if __name__ == "__main__":
    main()
