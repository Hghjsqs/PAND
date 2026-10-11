from pathlib import Path
import tempfile

from app.document_processor import load_markdown_documents, write_jsonl
from app.retriever import search_jsonl


def main() -> None:
    # 测试使用临时 JSONL，避免修改项目中的正式处理结果。
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    chunks = list(load_markdown_documents(raw_dir))

    with tempfile.TemporaryDirectory() as temp_dir:
        chunks_path = Path(temp_dir) / "chunks.jsonl"
        write_jsonl(chunks, chunks_path)

        # 有依据的问题应该命中对应制度文件。
        leave_results = search_jsonl(chunks_path, "年假怎么申请", top_k=3)
        assert leave_results
        assert leave_results[0].chunk["source"] == "leave_policy.md"
        assert leave_results[0].score > 0

        it_results = search_jsonl(chunks_path, "电脑无法连接公司网络", top_k=3)
        assert it_results
        assert it_results[0].chunk["source"] == "it_support.md"

        # 范围外问题不应仅因为命中“的、怎么”等虚词而返回结果。
        unrelated_results = search_jsonl(chunks_path, "今天的天气怎么样", top_k=3)
        assert unrelated_results == []

    print("检索测试通过")


if __name__ == "__main__":
    main()
