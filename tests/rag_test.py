from pathlib import Path
import tempfile

from app.document_processor import load_markdown_documents, write_jsonl
from app.rag import build_rag_prompt, format_context
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

    print("RAG Prompt 测试通过")


if __name__ == "__main__":
    main()
