import json
from pathlib import Path
import tempfile

from app.document_processor import (
    load_markdown_documents,
    parse_markdown,
    write_jsonl,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    chunks = list(load_markdown_documents(raw_dir))

    assert len(chunks) > 10
    assert all(chunk.source.endswith(".md") for chunk in chunks)
    assert all(chunk.document_id for chunk in chunks)
    assert all(chunk.version == "v1.0" for chunk in chunks)
    assert all(chunk.text.strip() for chunk in chunks)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert any(chunk.section == "2. 请假申请流程" for chunk in chunks)

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "chunks.jsonl"
        count = write_jsonl(chunks, output_path)
        rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
        assert count == len(chunks)
        assert len(rows) == len(chunks)
        assert rows[0]["chunk_id"] == chunks[0].chunk_id

    sample = "# Title\n\n版本：v2.0\n\n## Section\n\nA short paragraph."
    sample_chunks = parse_markdown(sample, "sample.md")
    assert sample_chunks[0].title == "Title"
    assert sample_chunks[0].version == "v2.0"
    assert sample_chunks[0].section == "Section"

    print(f"文档处理测试通过: chunks={len(chunks)}")


if __name__ == "__main__":
    main()
